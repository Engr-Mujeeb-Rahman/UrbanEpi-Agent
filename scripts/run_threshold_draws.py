"""Added for the supervisor's comment 25. Curves for Figures 2 to 4 averaged over 15 threshold draws, with their range.

Each draw recalibrates the threshold on a fresh noisy validation release, as in the main runs. Outbreaks and the standard-normal
noise draws on the test period are shared across draws and arms, so the band shows how much the threshold draw alone moves each curve.
Post hoc: no frozen setting is changed.
"""
import json
from pathlib import Path

import numpy as np

DOC = __doc__
source = Path("scripts/run_detection_experiments.py").read_text(encoding="utf8")
exec(source[:source.index("started = time.time()")])

DRAWS = 15
TRIALS = setup["trials_per_size"]
SIZES = list(setup["excess_cases"])
K = len(context.institutions)


def z_draw(size: int, trial: int, shape: tuple) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], 95, size, trial])).standard_normal(shape)


def gate_arm(label: str, sigma: float, table: np.ndarray, holder: int | None = None, scorer=naeg_scorer) -> dict:
    taus = [threshold_for(sigma, scorer, stream(f"draws-{label}-{d}"), table=table) for d in range(DRAWS)]
    pod = {}
    calls = []
    for size in SIZES:
        hits = np.zeros((TRIALS, DRAWS), dtype=bool)
        for i, outbreak in enumerate(plans[size][:TRIALS]):
            if holder is not None:
                keep = outbreak.holders == holder
                outbreak = type(outbreak)(int(keep.sum()), outbreak.boroughs, outbreak.day_start, outbreak.duration, outbreak.days[keep], outbreak.places[keep], outbreak.holders[keep])
            clean = add_outbreak(table, outbreak, grid)
            values = clean + sigma * z_draw(size, i, clean.shape) if sigma else clean
            scores = scorer(values, sigma)
            for d, tau in enumerate(taus):
                alerts = run_gate(scores, test, context.boroughs, context.adjacency, tau, gate_cfg)
                hits[i, d] = first_hit(alerts, outbreak, grid) is not None
                if size == 0:
                    calls.append((d, len(alerts)))
        pod[str(size)] = hits.mean(axis=0)
    per_draw_calls = np.array([np.mean([c for d2, c in calls if d2 == d]) for d in range(DRAWS)])
    return {"thresholds": [round(t, 3) for t in taus], "pod_by_draw": {s: [round(float(x), 4) for x in v] for s, v in pod.items()},
            "pod_mean": {s: round(float(v.mean()), 4) for s, v in pod.items()}, "pod_min": {s: round(float(v.min()), 4) for s, v in pod.items()},
            "pod_max": {s: round(float(v.max()), 4) for s, v in pod.items()}, "calls_by_draw": [round(float(c), 2) for c in per_draw_calls]}


results = {"federation": {}, "designs": {}, "calls": {}}
record2 = StreamBudget.from_config(cfg, 2, "record").borough_sigma
results["federation"]["all"] = gate_arm("fed-all", record2, base)
print("federation all", results["federation"]["all"]["pod_mean"], flush=True)
for k, name in enumerate(context.institutions):
    results["federation"][name] = gate_arm(f"fed-{name}", record2, base_cells[:, :, k], holder=k)
    print("federation", name, results["federation"][name]["pod_mean"], flush=True)
    Path("experiments/results/final/threshold_draws.json").write_text(json.dumps({"note": DOC, "draws": DRAWS, "trials_per_size": TRIALS, "results": results}, indent=1), encoding="utf8")

for epsilon in (1, 2, 4):
    record = StreamBudget.from_config(cfg, epsilon, "record")
    person = StreamBudget.from_config(cfg, epsilon, "person")
    per_window = sigma_for(record.borough_rho / grid.size, 1.0)
    designs = {"record_distributed": record.borough_sigma, "person_distributed": person.borough_sigma, "person_per_institution": person.borough_sigma * math.sqrt(K),
               "per_window_distributed": per_window, "per_window_per_institution": per_window * math.sqrt(K)}
    for name, sigma in designs.items():
        results["designs"][f"eps{epsilon}/{name}"] = gate_arm(f"design-{name}-{epsilon}", sigma, base)
        print("design", epsilon, name, results["designs"][f"eps{epsilon}/{name}"]["pod_mean"], flush=True)
        Path("experiments/results/final/threshold_draws.json").write_text(json.dumps({"note": DOC, "draws": DRAWS, "trials_per_size": TRIALS, "results": results}, indent=1), encoding="utf8")

# Figure 4: calls against POD at epsilon 2, per draw.
sigma = record2
running = lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, None, 0.0), neighbors)
naeg = results["designs"]["eps2/record_distributed"]
results["calls"]["naeg"] = {"calls_by_draw": naeg["calls_by_draw"], "pod_80": naeg["pod_by_draw"]["80"], "pod_160": naeg["pod_by_draw"]["160"]}
run = gate_arm("calls-running", sigma, base, scorer=running)
results["calls"]["running_cusum"] = {"calls_by_draw": run["calls_by_draw"], "pod_80": run["pod_by_draw"]["80"], "pod_160": run["pod_by_draw"]["160"]}
print("running cusum", run["pod_mean"], flush=True)
naeg_taus = naeg["thresholds"]
reset_taus = [_resetting_threshold(sigma, stream(f"draws-reset-{d}")) for d in range(DRAWS)]
cells = {"every_alert_cell_no_merging": [], "resetting_cusum": [], "periodic_same_calls": [], "random_same_calls": []}
rng = stream("draws-random")
pods = {k: {80: np.zeros((TRIALS, DRAWS), bool), 160: np.zeros((TRIALS, DRAWS), bool)} for k in cells}
counts = {"every_alert_cell_no_merging": np.zeros((TRIALS, DRAWS)), "resetting_cusum": np.zeros((TRIALS, DRAWS))}
for size in (0, 80, 160):
    for i, outbreak in enumerate(plans[size][:TRIALS]):
        clean = add_outbreak(base, outbreak, grid)
        values = clean + sigma * z_draw(size, i, clean.shape)
        scores = np.nan_to_num(naeg_scorer(values, sigma), nan=-np.inf)
        z = z_scores(values, sigma, settings)
        begin, stop = outbreak_windows(outbreak, grid)
        places = [context.position[b] for b in outbreak.boroughs]
        for d in range(DRAWS):
            above = scores >= naeg_taus[d]
            alarms = resetting_cusum_alarms(z, settings.slack, reset_taus[d])
            n_calls = max(1, int(round(naeg["calls_by_draw"][d])))
            chosen = test[::max(1, len(test) // n_calls)]
            picks = rng.choice(test, size=len(chosen), replace=False)
            if size == 0:
                counts["every_alert_cell_no_merging"][i, d] = above[test].sum()
                counts["resetting_cusum"][i, d] = alarms[test].sum()
                continue
            pods["every_alert_cell_no_merging"][size][i, d] = above[begin:stop + 1][:, places].any()
            pods["resetting_cusum"][size][i, d] = alarms[begin:stop + 1][:, places].any()
            pods["periodic_same_calls"][size][i, d] = any(begin <= t <= stop and above[t, places].any() for t in chosen)
            pods["random_same_calls"][size][i, d] = any(begin <= t <= stop and above[t, places].any() for t in picks)
for name in cells:
    row = {"pod_80": [round(float(x), 4) for x in pods[name][80].mean(axis=0)], "pod_160": [round(float(x), 4) for x in pods[name][160].mean(axis=0)]}
    if name in counts:
        row["calls_by_draw"] = [round(float(x), 2) for x in counts[name].mean(axis=0)]
    else:
        row["calls_by_draw"] = [float(len(test[::max(1, len(test) // max(1, int(round(c))))])) for c in naeg["calls_by_draw"]]
    results["calls"][name] = row
    print(name, {k: round(float(np.mean(v)), 3) for k, v in row.items()}, flush=True)
Path("experiments/results/final/threshold_draws.json").write_text(json.dumps({"note": DOC, "draws": DRAWS, "trials_per_size": TRIALS, "results": results}, indent=1), encoding="utf8")
