"""Go/No-Go 1: can plausible outbreaks be seen through privacy noise?

Uses validation data only. The test period stays untouched. This is a quick feasibility check
with a simple detector, not a paper result.
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
from urbanepi.privacy.accounting import StreamBudget

cfg = load_config()
check = cfg["go_no_go_1"]
events = pd.read_parquet(cfg["data"]["events_path"])
boroughs_table = pd.read_csv(cfg["data"]["boroughs_path"], dtype={"code": str})
boroughs = boroughs_table["code"].tolist()
populations = dict(zip(boroughs, boroughs_table["population_2020"]))
adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
institutions = list(cfg["institutions"].values())
grid = WindowGrid.from_config(cfg)
position = {code: i for i, code in enumerate(boroughs)}

first, last = cfg["study"]["splits"]["validation"]
windows = grid.split_windows(first, last)
period_start = grid.first_day(int(windows[0]))
period_end = grid.first_day(int(windows[-1]) + 1) - pd.Timedelta(days=1)
source = events[(events["admitted"] >= period_start) & (events["admitted"] <= period_end)]
truth = count_cells(events, grid, boroughs, institutions).sum(axis=2).astype(float)
days_in_period = (period_end - period_start).days + 1
visible_end = int(windows[-1]) + 1
sigma_rng = np.random.SeedSequence([cfg["seed"], 1])


def score_table(table: np.ndarray) -> np.ndarray:
    return c2_scores(table[:visible_end], check["baseline_windows"], check["guard_windows"])


def calibrate(sigma: float, rng: np.random.Generator) -> float:
    """Find the lowest threshold meeting the alert-rate target on validation data with no injection."""
    runs = [score_table(fast_noisy_table(truth, sigma, rng) if sigma else truth)
            for _ in range(check["calibration_seeds"] if sigma else 1)]
    for threshold in np.arange(1.0, 15.01, 0.05):
        rate = np.mean([alert_onsets(s, threshold)[windows].sum() for s in runs]) / (days_in_period / 30)
        if rate <= check["alerts_per_30_days"]:
            return float(threshold)
    return float("inf")


def draw_outbreaks(rng: np.random.Generator) -> dict[int, list[dict]]:
    """Draw the same outbreak places and dates for every privacy setting, so settings stay paired."""
    plans = {}
    for size in check["excess_cases"]:
        plans[size] = []
        for trial in range(check["trials_per_size"]):
            duration = int(rng.choice(check["duration_days"]))
            spec = sample_spec(rng, cfg, adjacency, f"{size}-{trial}", size, duration,
                               str(period_start.date()), str(period_end.date()))
            added = np.zeros_like(truth)
            if size:
                records = make_outbreak_records(rng, spec, source, populations, cfg["injection"]["lognormal_sigma"])
                added = count_cells(records, grid, boroughs, institutions).sum(axis=2).astype(float)
            start = grid.index_of(pd.Series([pd.Timestamp(spec.start)]))[0]
            end = grid.index_of(pd.Series([pd.Timestamp(spec.start) + pd.Timedelta(days=duration - 1)]))[0]
            plans[size].append({"added": added, "start": int(start), "end": int(end),
                                "places": [position[b] for b in spec.boroughs]})
    return plans


def probability_of_detection(plan: list[dict], sigma: float, threshold: float, rng: np.random.Generator) -> float:
    hits = 0
    for outbreak in plan:
        table = truth + outbreak["added"]
        table = fast_noisy_table(table, sigma, rng) if sigma else table
        above = np.nan_to_num(score_table(table), nan=-np.inf) >= threshold
        hits += bool(above[outbreak["start"]:outbreak["end"] + 1][:, outbreak["places"]].any())
    return hits / len(plan)


streams = sigma_rng.spawn(40)
plans = draw_outbreaks(np.random.default_rng(streams[0]))
settings = [("none", None)] + [(unit, e) for unit in ("record", "person") for e in cfg["privacy"]["epsilons"]]
results = []
for index, (unit, epsilon) in enumerate(settings, start=1):
    sigma = 0.0 if unit == "none" else StreamBudget.from_config(cfg, epsilon, unit).borough_sigma
    rng = np.random.default_rng(streams[index])
    threshold = calibrate(sigma, rng)
    pod = {size: probability_of_detection(plans[size], sigma, threshold, rng) for size in check["excess_cases"]}
    passing = [s for s in check["excess_cases"] if s and pod[s] >= check["target_pod"]]
    results.append({"unit": unit, "epsilon": epsilon, "sigma": round(sigma, 2), "threshold": round(threshold, 2),
                    "pod": pod, "mdo": passing[0] if passing else None})
    print(f"{unit:7s} eps={epsilon} sigma={sigma:6.2f} tau={threshold:5.2f} MDO={results[-1]['mdo']} "
          + " ".join(f"{s}:{p:.2f}" for s, p in pod.items()), flush=True)

decision = next(r for r in results if r["unit"] == "record" and r["epsilon"] == check["decision_epsilon"])
outcome = "continue" if decision["mdo"] and decision["mdo"] <= check["plausible_excess_cases"] else (
    "adjust windows or areas" if decision["mdo"] else "rethink scope")
print("Decision at epsilon", check["decision_epsilon"], "record level:", outcome)
Path("experiments/results/go_no_go_1.json").write_text(json.dumps(
    {"note": "Validation data, simple C2 detector. Feasibility check, not a paper result.",
     "decision": outcome, "results": results}, indent=1), encoding="utf8")
