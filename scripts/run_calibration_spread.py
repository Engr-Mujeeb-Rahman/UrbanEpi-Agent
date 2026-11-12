"""Post-hoc check, not pre-registered: how much do results move when the one validation release behind each threshold changes?

Each draw makes a new noisy validation release, sets a new threshold, and runs the same outbreaks. Settings stay frozen.
"""
import json
from pathlib import Path

import numpy as np

DOC = __doc__
source = Path("scripts/run_detection_experiments.py").read_text(encoding="utf8")
exec(source[:source.index("started = time.time()")])

DRAWS, TRIALS, SIZES = 15, 80, (0, 80, 160)
ARMS = {
    "naeg": naeg_scorer,
    "c2_no_noise_floor": lambda v, s: c2_scores(v, settings.baseline, settings.guard),
    "minus_floor": lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, settings.memory, 0.0, noise_floor=False), neighbors),
}
CONFIGS = [(1, "record"), (2, "record"), (4, "record"), (2, "person")]


def summary(values: list[float]) -> dict:
    return {"mean": round(float(np.mean(values)), 3), "min": round(float(np.min(values)), 3), "max": round(float(np.max(values)), 3), "sd": round(float(np.std(values, ddof=1)), 3)}


results = {}
for epsilon, unit in CONFIGS:
    sigma = StreamBudget.from_config(cfg, epsilon, unit).borough_sigma
    for name in [*ARMS, "resetting_cusum"]:
        if unit == "person" and name != "naeg":
            continue
        taus, pods, reaches = [], {s: [] for s in SIZES}, []
        for draw in range(DRAWS):
            rng = stream(f"spread-{unit}-{epsilon}-{name}-{draw}")
            tau = _resetting_threshold(sigma, rng) if name == "resetting_cusum" else threshold_for(sigma, ARMS[name], rng)
            taus.append(tau)
            for size in SIZES:
                hits = []
                for outbreak in plans[size][:TRIALS]:
                    values = noisy(add_outbreak(base, outbreak, grid), sigma, rng)
                    if name == "resetting_cusum":
                        alarms = resetting_cusum_alarms(z_scores(values, sigma, settings), settings.slack, tau)
                        begin, stop = outbreak_windows(outbreak, grid)
                        hits.append(bool(alarms[begin:stop + 1][:, [context.position[b] for b in outbreak.boroughs]].any()))
                    else:
                        alerts = run_gate(ARMS[name](values, sigma), test, context.boroughs, context.adjacency, tau, gate_cfg)
                        hits.append(first_hit(alerts, outbreak, grid) is not None)
                pods[size].append(float(np.mean(hits)))
            reaches.append(pods[80][-1] >= target_pod)
        key = f"{unit}_eps{epsilon}_{name}"
        results[key] = {"threshold": summary(taus), "pod": {s: summary(v) for s, v in pods.items()}, "share_of_draws_reaching_target_at_80": round(float(np.mean(reaches)), 3), "pod_by_draw": {s: [round(x, 3) for x in v] for s, v in pods.items()}}
        print(key, json.dumps({k: results[key][k] for k in ("threshold", "pod", "share_of_draws_reaching_target_at_80")}), flush=True)
OUT.joinpath("calibration_spread.json").write_text(json.dumps({"note": DOC, "draws": DRAWS, "trials_per_size": TRIALS, "results": results}, indent=1), encoding="utf8")
