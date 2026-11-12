"""Tests for the Phase 2 data pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbanepi.city.etl import clean_records
from urbanepi.city.injector import OutbreakSpec, make_outbreak_records, sample_spec, write_ground_truth
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config

CFG = load_config()
INSTITUTIONS = list(CFG["institutions"].values())


def raw_row(**changes) -> dict:
    row = {
        "ID_REGISTRO": "a", "ORIGEN": "1", "SECTOR": "4", "ENTIDAD_RES": "09", "MUNICIPIO_RES": "007",
        "TIPO_PACIENTE": "1", "FECHA_INGRESO": "2024-06-01", "NEUMONIA": "2", "EDAD": "30", "RESULTADO_PCR": "5",
        "RESULTADO_PCR_COINFECCION": "5", "CLASIFICACION_FINAL_COVID": "7", "CLASIFICACION_FINAL_FLU": "7",
    }
    row.update({column: "2" for column in CFG["conditions"].values()})
    row.update(changes)
    return row


@pytest.fixture()
def events() -> pd.DataFrame:
    rows = [
        raw_row(ID_REGISTRO="a", CLASIFICACION_FINAL_FLU="3", EDAD="70", DIABETES="1", TIPO_PACIENTE="2"),
        raw_row(ID_REGISTRO="b", CLASIFICACION_FINAL_COVID="3", EDAD="4", SECTOR="15", NEUMONIA="1"),
        raw_row(ID_REGISTRO="c", FECHA_INGRESO="2025-04-16"),
        raw_row(ID_REGISTRO="d", SECTOR="3"),
        raw_row(ID_REGISTRO="e", MUNICIPIO_RES="999"),
    ]
    table, _ = clean_records(pd.DataFrame(rows), CFG)
    return table


def test_exclusions_are_counted():
    rows = [raw_row(ID_REGISTRO="a"), raw_row(ID_REGISTRO="c", FECHA_INGRESO="2025-04-16"),
            raw_row(ID_REGISTRO="d", SECTOR="3"), raw_row(ID_REGISTRO="e", MUNICIPIO_RES="999")]
    table, report = clean_records(pd.DataFrame(rows), CFG)
    assert table["record_id"].tolist() == ["a"]
    assert report["excluded_outside_period"] == 1
    assert report["excluded_small_institution"] == 1
    assert report["excluded_unknown_borough"] == 1


def test_fields_are_mapped(events):
    first = events[events["record_id"] == "a"].iloc[0]
    assert first["flu_confirmed"] and not first["covid_confirmed"]
    assert first["age_band"] == "65+" and first["diabetes"] and first["hospitalized"]
    second = events[events["record_id"] == "b"].iloc[0]
    assert second["covid_confirmed"] and second["age_band"] == "0-4"
    assert second["institution"] == "IMSS_BIENESTAR_OPD" and second["pneumonia"]


def test_pcr_codes_become_pathogens_including_coinfections():
    rows = [raw_row(ID_REGISTRO="a", RESULTADO_PCR="3", RESULTADO_PCR_COINFECCION="34"), raw_row(ID_REGISTRO="b", RESULTADO_PCR="5", RESULTADO_PCR_COINFECCION="5"),
            raw_row(ID_REGISTRO="c", RESULTADO_PCR="11", RESULTADO_PCR_COINFECCION="998")]
    table = clean_records(pd.DataFrame(rows), CFG)[0].set_index("record_id")
    assert list(table.loc["a", "pathogens"]) == ["COVID", "FLU_A_H3"]
    assert list(table.loc["b", "pathogens"]) == []
    assert list(table.loc["c", "pathogens"]) == ["RSV"]


def test_repeated_ids_are_rejected():
    with pytest.raises(ValueError):
        clean_records(pd.DataFrame([raw_row(), raw_row()]), CFG)


def test_window_grid_covers_the_horizon_exactly():
    grid = WindowGrid.from_config(CFG)
    assert grid.size == 157
    assert grid.first_day(156) + pd.Timedelta(days=2) == pd.Timestamp("2025-04-15")
    sizes = [len(grid.split_windows(a, b)) for a, b in CFG["study"]["splits"].values()]
    assert sum(sizes) == grid.size


def test_cells_keep_every_record(events):
    grid = WindowGrid.from_config(CFG)
    cells = count_cells(events, grid, ["007"], INSTITUTIONS)
    assert cells.sum() == len(events)


def test_unknown_borough_is_rejected(events):
    with pytest.raises(ValueError):
        count_cells(events, WindowGrid.from_config(CFG), ["002"], INSTITUTIONS)


def source_records(boroughs=("002", "003")) -> pd.DataFrame:
    rows = [raw_row(ID_REGISTRO=f"{b}-{i}", MUNICIPIO_RES=b, SECTOR=s)
            for b in boroughs for i, s in enumerate(["4", "12"] * 5)]
    return clean_records(pd.DataFrame(rows), CFG)[0]


@pytest.mark.parametrize("excess", [1, 10, 320])
@pytest.mark.parametrize("duration", [7, 21])
def test_outbreak_has_exact_size_place_and_time(excess, duration):
    spec = OutbreakSpec("t", ("002", "003"), "2025-01-01", duration, excess)
    source = source_records()
    before = source.copy()
    added = make_outbreak_records(np.random.default_rng(1), spec, source, {"002": 100, "003": 300}, 0.6)
    assert len(added) == excess
    assert added["injected"].all() and added["record_id"].is_unique
    assert set(added["borough"]) <= {"002", "003"}
    assert added["admitted"].min() >= pd.Timestamp("2025-01-01")
    assert added["admitted"].max() < pd.Timestamp("2025-01-01") + pd.Timedelta(days=duration)
    pd.testing.assert_frame_equal(source, before)


def test_outbreak_is_reproducible():
    spec = OutbreakSpec("t", ("002",), "2025-01-01", 14, 40)
    make = lambda: make_outbreak_records(np.random.default_rng(9), spec, source_records(), {"002": 1, "003": 1}, 0.6)
    pd.testing.assert_frame_equal(make(), make())


def test_sampled_outbreaks_are_adjacent_and_inside_test():
    adjacency = {"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []}
    adjacency = {k: v for k, v in adjacency.items() if v}
    rng = np.random.default_rng(3)
    for i in range(200):
        spec = sample_spec(rng, CFG, adjacency, str(i), 10, 21, "2024-11-02", "2025-04-15")
        assert 1 <= len(spec.boroughs) <= 3
        if len(spec.boroughs) > 1:
            assert all(set(adjacency[b]) & set(spec.boroughs) for b in spec.boroughs)
        assert pd.Timestamp(spec.start) + pd.Timedelta(days=20) <= pd.Timestamp("2025-04-15")


def test_ground_truth_is_written_to_its_own_folder(tmp_path):
    path = write_ground_truth([OutbreakSpec("t", ("002",), "2025-01-01", 7, 10)], tmp_path / "truth", "run")
    assert path.parent.name == "truth"
    assert json.loads(path.read_text(encoding="utf8"))[0]["outbreak_id"] == "t"


PUBLIC = Path(CFG["data"]["boroughs_path"])
EVENTS = Path(CFG["data"]["events_path"])


@pytest.mark.skipif(not PUBLIC.exists(), reason="public metadata not built")
def test_public_metadata_is_consistent():
    boroughs = pd.read_csv(PUBLIC, dtype={"code": str})
    assert len(boroughs) == 16 and boroughs["population_2020"].sum() == 9_209_944
    adjacency = json.loads(Path(CFG["data"]["adjacency_path"]).read_text(encoding="utf8"))
    assert set(adjacency) == set(boroughs["code"])
    assert all(a in adjacency[b] for a, others in adjacency.items() for b in others), "adjacency must be symmetric"
    reached, frontier = set(), ["007"]
    while frontier:
        code = frontier.pop()
        if code not in reached:
            reached.add(code)
            frontier += adjacency[code]
    assert reached == set(adjacency), "every borough must be reachable"


@pytest.mark.skipif(not EVENTS.exists(), reason="events table not built")
def test_real_events_match_the_design():
    events = pd.read_parquet(EVENTS)
    assert events["record_id"].is_unique
    assert events["admitted"].min() >= pd.Timestamp("2024-01-01")
    assert events["admitted"].max() <= pd.Timestamp("2025-04-15")
    assert events["borough"].nunique() == 16
    assert set(events["institution"]) == set(INSTITUTIONS)
    assert not events["injected"].any()
    assert events["age_band"].isin(CFG["age_bands"]["labels"]).all()
    boroughs = sorted(events["borough"].unique())
    assert count_cells(events, WindowGrid.from_config(CFG), boroughs, INSTITUTIONS).sum() == len(events)
