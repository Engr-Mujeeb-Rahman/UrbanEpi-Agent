"""Follow-up to Go/No-Go 1: why was detection weak, even with no privacy noise?

Validation data only. Tries the longer window from the phases.md rule, and the design's noise-aware score.
The alert-rate target and the pass rule are unchanged from the first run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from urbanepi.aggregator.stream import fast_noisy_table
from urbanepi.city.injector import make_outbreak_records, sample_spec
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config
from urbanepi.gate.c2 import alert_onsets, c2_scores
from urbanepi.gate.naeg import adjacency_matrix, naeg_scores
from urbanepi.privacy.accounting import StreamBudget

cfg = load_config()
check = cfg["go_no_go_1"]
events = pd.read_parquet(cfg["data"]["events_path"])
table = pd.read_csv(cfg["data"]["boroughs_path"], dtype={"code": str})
boroughs = table["code"].tolist()
populations = dict(zip(boroughs, table["population_2020"]))
adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
institutions = list(cfg["institutions"].values())
position = {code: i for i, code in enumerate(boroughs)}
neighbors = adjacency_matrix(boroughs, adjacency)
SLACK, SPATIAL = 0.5, 0.5
first, last = cfg["study"]["splits"]["validation"]
sizes = check["excess_cases"]


def outbreaks(rng: np.random.Generator) -> dict[int, list[dict]]:
    source = events[(events["admitted"] >= first) & (events["admitted"] <= last)]
    plans: dict[int, list[dict]] = {}
    for size in sizes:
        plans[size] = []
        for trial in range(check["trials_per_size"]):
            duration = int(rng.choice(check["duration_days"]))
            spec = sample_spec(rng, cfg, adjacency, f"{size}-{trial}", size, duration, first, last)
            added = pd.DataFrame({"admitted": pd.Series(dtype="datetime64[ns]"), "borough": pd.Series(dtype=str)})
            if size:
                added = make_outbreak_records(rng, spec, source, populations, cfg["injection"]["lognormal_sigma"])[["admitted", "borough"]]
            plans[size].append({"spec": spec, "added": added, "places": [position[b] for b in spec.boroughs]})
    return plans


def evaluate(window_days: int, detector: str, unit: str, epsilon: float | None, plans: dict, rng: np.random.Generator) -> dict:
    grid = WindowGrid.from_config(cfg, window_days)
    truth = count_cells(events, grid, boroughs, institutions).sum(axis=2).astype(float)
    windows = grid.split_windows(first, last)
    period_days = len(windows) * window_days
    end = int(windows[-1]) + 1
    sigma = 0.0 if epsilon is None else StreamBudget.from_config(cfg, epsilon, unit).borough_sigma
    baseline = max(round(check["baseline_windows"] * 3 / window_days), 6)
    guard = max(round(check["guard_windows"] * 3 / window_days), 1)

    def score(values: np.ndarray) -> np.ndarray:
        noisy = fast_noisy_table(values, sigma, rng) if sigma else values
        if detector == "c2":
            return c2_scores(noisy[:end], baseline, guard)
        return naeg_scores(noisy[:end], sigma, baseline, guard, SLACK, SPATIAL, neighbors)

    runs = [score(truth) for _ in range(check["calibration_seeds"] if sigma else 1)]
    threshold = float("inf")
    for candidate in np.arange(0.5, 40.01, 0.1):
        if np.mean([alert_onsets(r, candidate)[windows].sum() for r in runs]) / (period_days / 30) <= check["alerts_per_30_days"]:
            threshold = float(candidate)
            break

    pod, narrow = {}, {}
    for size in sizes:
        hits, single, single_hits = 0, 0, 0
        for outbreak in plans[size]:
            spec = outbreak["spec"]
            values = truth.copy()
            added = outbreak["added"]
            if len(added):
                np.add.at(values, (grid.index_of(added["admitted"]), [position[b] for b in added["borough"]]), 1)
            above = np.nan_to_num(score(values), nan=-np.inf) >= threshold
            begin = grid.index_of(pd.Series([pd.Timestamp(spec.start)]))[0]
            stop = grid.index_of(pd.Series([pd.Timestamp(spec.start) + pd.Timedelta(days=spec.duration_days - 1)]))[0]
            hit = bool(above[begin:stop + 1][:, outbreak["places"]].any())
            hits += hit
            if len(spec.boroughs) == 1:
                single += 1
                single_hits += hit
        pod[size] = hits / len(plans[size])
        narrow[size] = single_hits / max(single, 1)
    passing = [s for s in sizes if s and pod[s] >= check["target_pod"]]
    result = {"window_days": window_days, "detector": detector, "unit": unit, "epsilon": epsilon, "sigma": round(sigma, 2),
              "threshold": round(threshold, 1), "pod": pod, "pod_single_borough": narrow, "mdo": passing[0] if passing else None}
    print(f"dt={window_days} {detector:4s} {unit:6s} eps={epsilon} sigma={sigma:5.2f} tau={threshold:5.1f} MDO={result['mdo']} | "
          + " ".join(f"{s}:{p:.2f}" for s, p in pod.items()) + " | single " + " ".join(f"{s}:{p:.2f}" for s, p in narrow.items()), flush=True)
    return result


seeds = np.random.SeedSequence([cfg["seed"], 2]).spawn(100)
plans = outbreaks(np.random.default_rng(seeds[0]))
settings = [("none", None), ("record", 1), ("record", 2), ("record", 4), ("person", 2), ("person", 4)]
results = []
counter = 1
for window_days in (3, 7):
    for detector in ("c2", "naeg"):
        for unit, epsilon in settings:
            results.append(evaluate(window_days, detector, unit, epsilon, plans, np.random.default_rng(seeds[counter])))
            counter += 1
Path("experiments/results/go_no_go_1b.json").write_text(json.dumps(
    {"note": "Validation data. Feasibility diagnosis, not a paper result.", "results": results}, indent=1), encoding="utf8")
