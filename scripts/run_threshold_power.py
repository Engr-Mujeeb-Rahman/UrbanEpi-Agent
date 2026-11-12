"""Post-hoc: separates threshold effects from trial sampling. Uses more trials per draw than calibration_spread and saves each draw's threshold."""
import json
from pathlib import Path

import numpy as np

source = Path("scripts/run_detection_experiments.py").read_text(encoding="utf8")
exec(source[:source.index("started = time.time()")])

DRAWS, SIZE = 15, 80
results = {}
for epsilon, unit in ((1, "record"), (2, "record"), (2, "person")):
    sigma = StreamBudget.from_config(cfg, epsilon, unit).borough_sigma
    taus, pods, chance = [], [], []
    for draw in range(DRAWS):
        rng = stream(f"power-{unit}-{epsilon}-{draw}")
        tau = threshold_for(sigma, naeg_scorer, rng)
        rates = {}
        for size in (0, SIZE):
            hits = []
            for outbreak in plans[size]:
                values = noisy(add_outbreak(base, outbreak, grid), sigma, rng)
                alerts = run_gate(naeg_scorer(values, sigma), test, context.boroughs, context.adjacency, tau, gate_cfg)
                hits.append(first_hit(alerts, outbreak, grid) is not None)
            rates[size] = float(np.mean(hits))
        taus.append(float(tau)); pods.append(rates[SIZE]); chance.append(rates[0])
        print(unit, epsilon, draw, round(tau, 2), rates, flush=True)
    n = len(plans[SIZE])
    mean = float(np.mean(pods))
    results[f"{unit}_eps{epsilon}"] = {"trials_per_draw": n, "thresholds": [round(t, 3) for t in taus], "pod_80": [round(p, 4) for p in pods], "pod_0": [round(p, 4) for p in chance],
        "pod_80_mean": round(mean, 3), "pod_80_min": round(min(pods), 3), "pod_80_max": round(max(pods), 3), "pod_80_sd": round(float(np.std(pods, ddof=1)), 4),
        "binomial_sd": round(float(np.sqrt(mean * (1 - mean) / n)), 4), "corr_threshold_pod": round(float(np.corrcoef(taus, pods)[0, 1]), 3), "corr_threshold_chance": round(float(np.corrcoef(taus, chance)[0, 1]), 3),
        "pod_0_mean": round(float(np.mean(chance)), 3), "pod_0_min": round(min(chance), 3), "pod_0_max": round(max(chance), 3), "share_reaching_target": round(float(np.mean(np.array(pods) >= target_pod)), 3)}
    print(results[f"{unit}_eps{epsilon}"], flush=True)
    OUT.joinpath("threshold_power.json").write_text(json.dumps({"note": __doc__, "results": results}, indent=1), encoding="utf8")
