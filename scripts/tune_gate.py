"""Choose the gate score on validation data. The selection rule is fixed in configs/base.yaml before this runs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from urbanepi.aggregator.stream import fast_noisy_table
from urbanepi.city.injector import make_outbreak_records, sample_spec
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config
from urbanepi.gate.c2 import alert_onsets, c2_scores
from urbanepi.gate.naeg import GateScoreSettings, adjacency_matrix, estimate_null, gate_scores, simulate_null
from urbanepi.privacy.accounting import StreamBudget

cfg = load_config()
tune = cfg["gate_tuning"]
events = pd.read_parquet(cfg["data"]["events_path"])
table = pd.read_csv(cfg["data"]["boroughs_path"], dtype={"code": str})
boroughs = table["code"].tolist()
populations = dict(zip(boroughs, table["population_2020"]))
adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
position = {code: i for i, code in enumerate(boroughs)}
neighbors = adjacency_matrix(boroughs, adjacency)
grid = WindowGrid.from_config(cfg)
first, last = cfg["study"]["splits"]["validation"]
windows = grid.split_windows(first, last)
end = int(windows[-1]) + 1
period_days = len(windows) * grid.window_days
truth = count_cells(events, grid, boroughs, list(cfg["institutions"].values())).sum(axis=2).astype(float)[:end]
source = events[(events["admitted"] >= first) & (events["admitted"] <= last)]
seeds = np.random.SeedSequence([cfg["seed"], 5]).spawn(200)
rng = np.random.default_rng(seeds[0])

plans = {}
for size in tune["excess_cases"]:
    plans[size] = []
    for trial in range(tune["trials_per_size"]):
        duration = int(rng.choice(cfg["injection"]["duration_days"]))
        spec = sample_spec(rng, cfg, adjacency, f"{size}-{trial}", size, duration, first, last)
        added = np.zeros_like(truth)
        if size:
            records = make_outbreak_records(rng, spec, source, populations, cfg["injection"]["lognormal_sigma"])
            np.add.at(added, (grid.index_of(records["admitted"]), [position[b] for b in records["borough"]]), 1)
        begin = int(grid.index_of(pd.Series([pd.Timestamp(spec.start)]))[0])
        stop = int(grid.index_of(pd.Series([pd.Timestamp(spec.start) + pd.Timedelta(days=duration - 1)]))[0])
        plans[size].append((added, begin, stop, [position[b] for b in spec.boroughs]))

base = {"baseline": tune["baseline_windows"], "guard": tune["guard_windows"], "slack": tune["slack"]}
variants = {"c2": None}
for name, memory, weight in [("z only", 1, 0.0), ("sum 2", 2, 0.0), ("sum 3", 3, 0.0), ("sum 4", 4, 0.0),
                             ("sum 2 + spatial", 2, 0.5), ("sum 3 + spatial", 3, 0.5), ("cusum + spatial", None, 0.5)]:
    variants[name] = GateScoreSettings(memory=memory, spatial_weight=weight, **base)
if not all(k in variants for k in ("c2", "z only")):
    raise ValueError("Variants are missing")


def score(values: np.ndarray, sigma: float, settings: GateScoreSettings | None) -> np.ndarray:
    if settings is None:
        return c2_scores(values, base["baseline"], base["guard"])
    return gate_scores(values, sigma, settings, neighbors)


def null_threshold(sigma: float, settings: GateScoreSettings | None, generator: np.random.Generator) -> tuple[float, float]:
    released = fast_noisy_table(truth, sigma, generator) if sigma else truth
    level, dispersion = estimate_null(released[windows], sigma)
    runs = []
    for _ in range(tune["null_seeds"]):
        simulated = simulate_null(generator, level, dispersion, end)
        runs.append(score(fast_noisy_table(simulated, sigma, generator) if sigma else simulated, sigma, settings))
    for candidate in np.arange(0.5, 60.01, 0.05):
        rate = np.mean([alert_onsets(r, candidate)[windows].sum() for r in runs]) / (period_days / 30)
        if rate <= tune["null_alerts_per_30_days"]:
            return float(candidate), dispersion
    return float("inf"), dispersion


results = []
counter = 1
for unit, epsilon in tune["settings"]:
    sigma = 0.0 if unit == "none" else StreamBudget.from_config(cfg, epsilon, unit).borough_sigma
    for name, settings in variants.items():
        generator = np.random.default_rng(seeds[counter])
        counter += 1
        threshold, dispersion = null_threshold(sigma, settings, generator)
        real = np.mean([alert_onsets(score(fast_noisy_table(truth, sigma, generator) if sigma else truth, sigma, settings), threshold)[windows].sum()
                        for _ in range(10 if sigma else 1)]) / (period_days / 30)
        pod = {}
        for size, plan in plans.items():
            hits = 0
            for added, begin, stop, places in plan:
                values = truth + added
                noisy = fast_noisy_table(values, sigma, generator) if sigma else values
                above = np.nan_to_num(score(noisy, sigma, settings), nan=-np.inf) >= threshold
                hits += bool(above[begin:stop + 1][:, places].any())
            pod[size] = round(hits / len(plan), 3)
        row = {"unit": unit, "epsilon": epsilon, "sigma": round(sigma, 2), "variant": name, "threshold": round(threshold, 2),
               "null_dispersion": round(dispersion, 2), "real_alerts_per_30_days": round(float(real), 2), "pod": pod}
        results.append(row)
        print(f"{unit:6s} eps={epsilon} {name:16s} tau={threshold:5.2f} disp={dispersion:4.2f} real/30d={real:4.1f} "
              + " ".join(f"{s}:{p:.2f}" for s, p in pod.items()), flush=True)

selection_unit, selection_epsilon = tune["selection_setting"]
pool = [r for r in results if r["unit"] == selection_unit and r["epsilon"] == selection_epsilon and r["variant"] != "c2"]
allowed = [r for r in pool if r["pod"][0] <= tune["max_chance_pod"] and r["real_alerts_per_30_days"] <= tune["max_real_alerts_per_30_days"]]
key = tune["selection_excess_cases"]
chosen = max(allowed, key=lambda r: (r["pod"][key], -r["pod"][0])) if allowed else None
print("Allowed:", [r["variant"] for r in allowed], "Chosen:", chosen and chosen["variant"])
Path("experiments/results/gate_tuning.json").write_text(json.dumps(
    {"note": "Validation data only.", "chosen": chosen, "results": results}, indent=1, default=str), encoding="utf8")
