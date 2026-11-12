"""Turn the Mexico open-data file into the Mexico City events table."""
import json
import zipfile
from pathlib import Path

import pandas as pd

YES = "1"
HOSPITALIZED = "2"
RAW_COLUMNS = [
    "ID_REGISTRO", "ORIGEN", "SECTOR", "ENTIDAD_RES", "MUNICIPIO_RES", "TIPO_PACIENTE", "FECHA_INGRESO",
    "NEUMONIA", "EDAD", "RESULTADO_PCR", "RESULTADO_PCR_COINFECCION",
    "CLASIFICACION_FINAL_COVID", "CLASIFICACION_FINAL_FLU",
]


def pathogen_lookup(ontology_path: str) -> dict[str, str]:
    """Map each PCR result code to its pathogen, which is the leaf syndrome's name."""
    syndromes = json.loads(Path(ontology_path).read_text(encoding="utf8"))["syndromes"]
    return {code: name for name, entry in syndromes.items() for code in entry["pcr_codes"]}


def read_state_records(cfg: dict) -> pd.DataFrame:
    """Read all raw records for the configured state, with every value as text."""
    columns = RAW_COLUMNS + list(cfg["conditions"].values())
    with zipfile.ZipFile(cfg["data"]["source_zip"]) as archive:
        with archive.open(cfg["data"]["source_member"]) as handle:
            raw = pd.read_csv(handle, usecols=columns, dtype=str, encoding="latin-1")
    return raw[raw["ENTIDAD_RES"] == cfg["data"]["state_code"]].copy()


def clean_records(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the study filters and return the events table with a count of each exclusion."""
    report = {"state_records": int(len(raw))}
    admitted = pd.to_datetime(raw["FECHA_INGRESO"], errors="raise")
    start, end = pd.Timestamp(cfg["study"]["start"]), pd.Timestamp(cfg["study"]["end"])

    in_period = admitted.between(start, end)
    report["excluded_outside_period"] = int((~in_period).sum())
    known_institution = raw["SECTOR"].isin(cfg["institutions"])
    report["excluded_small_institution"] = int((in_period & ~known_institution).sum())
    known_borough = raw["MUNICIPIO_RES"].str.fullmatch(r"0(0[2-9]|1[0-7])")
    report["excluded_unknown_borough"] = int((in_period & known_institution & ~known_borough).sum())

    keep = in_period & known_institution & known_borough
    kept, admitted = raw[keep], admitted[keep]
    if kept["ID_REGISTRO"].duplicated().any():
        raise ValueError("ID_REGISTRO repeats within the study period")

    lookup = pathogen_lookup(cfg["data"]["ontology_path"])
    confirmed = cfg["classification"]["confirmed_code"]
    bands = cfg["age_bands"]
    events = pd.DataFrame({
        "record_id": kept["ID_REGISTRO"],
        "admitted": admitted,
        "borough": kept["MUNICIPIO_RES"],
        "institution": kept["SECTOR"].map(cfg["institutions"]),
        "age_band": pd.cut(kept["EDAD"].astype(int), bins=bands["edges"], labels=bands["labels"], right=False).astype(str),
        "flu_confirmed": kept["CLASIFICACION_FINAL_FLU"] == confirmed,
        "covid_confirmed": kept["CLASIFICACION_FINAL_COVID"] == confirmed,
        "pneumonia": kept["NEUMONIA"] == YES,
        "hospitalized": kept["TIPO_PACIENTE"] == HOSPITALIZED,
        "pathogens": [sorted({lookup[c] for c in codes if c in lookup})
                      for codes in zip(kept["RESULTADO_PCR"], kept["RESULTADO_PCR_COINFECCION"])],
        "pcr_code": kept["RESULTADO_PCR"],
        "pcr_coinfection_code": kept["RESULTADO_PCR_COINFECCION"],
        "injected": False,
    })
    for name, column in cfg["conditions"].items():
        events[name] = kept[column] == YES
    events = events.sort_values(["admitted", "record_id"]).reset_index(drop=True)
    report["events"] = int(len(events))
    return events, report
