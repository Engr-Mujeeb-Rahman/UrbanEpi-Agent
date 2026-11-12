"""Second follow-up to Go/No-Go 1: what limits detection, and how much do the stream design and zones matter?

Validation data only. Uses the simple C2 detector at 3-day windows. Feasibility diagnosis, not a paper result.
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
from urbanepi.privacy.accounting import StreamBudget, sigma_for

cfg = load_config()
check = cfg["go_no_go_1"]
events = pd.read_parquet(cfg["data"]["events_path"])
table = pd.read_csv(cfg["data"]["boroughs_path"], dtype={"code": str})
boroughs = table["code"].tolist()
names = dict(zip(boroughs, table["name"]))
populations = dict(zip(boroughs, table["population_2020"]))
adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
position = {code: i for i, code in enumerate(boroughs)}
grid = WindowGrid.from_config(cfg)
first, last = cfg["study"]["splits"]["validation"]
windows = grid.split_windows(first, last)
end = int(windows[-1]) + 1
period_days = len(windows) * grid.window_days
truth = count_cells(events, grid, boroughs, list(cfg["institutions"].values())).sum(axis=2).astype(float)
source = events[(events["admitted"] >= first) & (events["admitted"] <= last)]
BASE, GUARD, TRIALS = check["baseline_windows"], check["guard_windows"], check["trials_per_size"]
SIZES = [0, 40, 80, 160, 320]
ZONES = {"north": ["002", "005"], "center": ["006", "014", "015", "016", "017"], "east": ["007", "011"],
         "south": ["003", "009", "012", "013"], "west": ["004", "008", "010"]}
zone_index = {b: i for i, codes in enumerate(ZONES.values()) for b in codes}
seeds = np.random.SeedSequence([cfg["seed"], 3]).spawn(50)
rng = np.random.default_rng(seeds[0])


def to_zones(values: np.ndarray) -> np.ndarray:
    merged = np.zeros((values.shape[0], len(ZONES)))
    for code, zone in zone_index.items():
        merged[:, zone] += values[:, position[code]]
    return merged


def scores(values: np.ndarray, sigma: float, generator: np.random.Generator, zones: bool = False) -> np.ndarray:
    data = to_zones(values) if zones else values
    noisy = fast_noisy_table(data, sigma, generator) if sigma else data
    return c2_scores(noisy[:end], BASE, GUARD)


def threshold_for(runs: list[np.ndarray]) -> float:
    for candidate in np.arange(0.5, 40.01, 0.05):
        if np.mean([alert_onsets(r, candidate)[windows].sum() for r in runs]) / (period_days / 30) <= check["alerts_per_30_days"]:
            return float(candidate)
    return float("inf")


plans = {}
for size in SIZES:
    plans[size] = []
    for trial in range(TRIALS):
        duration = int(rng.choice(check["duration_days"]))
        spec = sample_spec(rng, cfg, adjacency, f"{size}-{trial}", size, duration, first, last)
        added = np.zeros_like(truth)
        if size:
            records = make_outbreak_records(rng, spec, source, populations, cfg["injection"]["lognormal_sigma"])
            np.add.at(added, (grid.index_of(records["admitted"]), [position[b] for b in records["borough"]]), 1)
        begin = grid.index_of(pd.Series([pd.Timestamp(spec.start)]))[0]
        stop = grid.index_of(pd.Series([pd.Timestamp(spec.start) + pd.Timedelta(days=duration - 1)]))[0]
        plans[size].append({"added": added, "begin": begin, "stop": stop, "codes": spec.boroughs})


def pod(sizes: list[int], sigma: float, threshold: float, generator: np.random.Generator, zones: bool = False,
        only=lambda outbreak: True) -> dict:
    out = {}
    for size in sizes:
        chosen = [o for o in plans[size] if only(o)]
        hits = 0
        for outbreak in chosen:
            above = np.nan_to_num(scores(truth + outbreak["added"], sigma, generator, zones), nan=-np.inf) >= threshold
            places = sorted({zone_index[c] for c in outbreak["codes"]}) if zones else [position[c] for c in outbreak["codes"]]
            hits += bool(above[outbreak["begin"]:outbreak["stop"] + 1][:, places].any())
        out[size] = round(hits / max(len(chosen), 1), 2)
    return out


report: dict = {"note": "Validation data, simple C2 detector. Feasibility diagnosis, not a paper result."}
real = c2_scores(truth[:end], BASE, GUARD)
tau = threshold_for([real])
print("A. Where the no-noise z-scores come from. Threshold for 1 alert per 30 days:", round(tau, 2))
onsets_at_3 = alert_onsets(real, 3.0)
print("Alert onsets at z >= 3 by month:", pd.Series([grid.first_day(int(t)).strftime("%Y-%m") for t in np.where(onsets_at_3[windows].any(axis=1))[0] + windows[0]]).value_counts().sort_index().to_dict())
print("Total onsets at z >= 3:", int(onsets_at_3[windows].sum()), "per 30 days:", round(onsets_at_3[windows].sum() / (period_days / 30), 1))
print("Peak z by borough, with validation mean count:")
for code in boroughs:
    i = position[code]
    print(f"  {names[code]:28s} mean {truth[windows, i].mean():6.1f}  variance/mean {truth[windows, i].var(ddof=1) / max(truth[windows, i].mean(), .1):5.1f}  peak z {np.nanmax(real[windows, i]):5.1f}")

print("\nB. Single-borough outbreaks of 80 cases, no noise, by borough size")
single = {}
for code in sorted(boroughs, key=lambda c: truth[windows, position[c]].mean()):
    single[code] = pod([80, 160], 0.0, tau, np.random.default_rng(seeds[1]), only=lambda o, c=code: o["codes"] == (c,))
    print(f"  {names[code]:28s} mean {truth[windows, position[code]].mean():6.1f} POD {single[code]}")
report["single_borough"] = single

print("\nC. Threshold from a null model with no real waves (design section 6.3)")
level = np.median(truth[windows], axis=0).clip(min=0.5)
dispersion = 2.3
null_rng = np.random.default_rng(seeds[2])
for label, sigma in (("none", 0.0), ("record eps=2", StreamBudget.from_config(cfg, 2, "record").borough_sigma)):
    runs = []
    for _ in range(check["calibration_seeds"]):
        success = 1 / dispersion
        simulated = null_rng.negative_binomial(level * success / (1 - success), success, size=(end, len(boroughs))).astype(float)
        runs.append(scores(simulated, sigma, null_rng))
    null_tau = threshold_for(runs)
    observed = np.mean([alert_onsets(scores(truth, sigma, null_rng), null_tau)[windows].sum() for _ in range(10)]) / (period_days / 30)
    result = pod(SIZES, sigma, null_tau, np.random.default_rng(seeds[3]))
    report[f"null_{label}"] = {"threshold": null_tau, "real_alerts_per_30_days": observed, "pod": result}
    print(f"  {label:14s} tau={null_tau:.2f} real alerts/30d={observed:.1f} POD {result}")

print("\nD. Privacy designs at epsilon = 2, same alert target on real validation data")
budget = StreamBudget.from_config(cfg, 2, "record")
person = StreamBudget.from_config(cfg, 2, "person")
per_window = sigma_for(budget.borough_rho / grid.size, 1.0)
designs = {"no noise": 0.0, "record, distributed": budget.borough_sigma, "person L=4, distributed": person.borough_sigma,
           "person L=4, per-institution": person.borough_sigma * 5 ** 0.5, "per-window, distributed": per_window,
           "per-window, per-institution": per_window * 5 ** 0.5}
report["designs"] = {}
for index, (label, sigma) in enumerate(designs.items()):
    generator = np.random.default_rng(seeds[10 + index])
    threshold = threshold_for([scores(truth, sigma, generator) for _ in range(check["calibration_seeds"] if sigma else 1)])
    result = pod(SIZES, sigma, threshold, generator)
    report["designs"][label] = {"sigma": round(sigma, 1), "threshold": threshold, "pod": result}
    print(f"  {label:30s} sigma={sigma:6.1f} tau={threshold:5.2f} POD {result}")

print("\nE. Five zones instead of 16 boroughs, same alert target")
report["zones"] = {}
for index, (label, sigma) in enumerate((("none", 0.0), ("record eps=4", StreamBudget.from_config(cfg, 4, "record").borough_sigma),
                                        ("record eps=2", budget.borough_sigma))):
    generator = np.random.default_rng(seeds[20 + index])
    threshold = threshold_for([scores(truth, sigma, generator, zones=True) for _ in range(check["calibration_seeds"] if sigma else 1)])
    result = pod(SIZES, sigma, threshold, generator, zones=True)
    report["zones"][label] = {"threshold": threshold, "pod": result}
    print(f"  {label:14s} tau={threshold:5.2f} POD {result}")

Path("experiments/results/go_no_go_1c.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf8")
