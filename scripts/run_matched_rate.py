"""Added for the November 2026 review (M2, M3, m8). Compares detectors and budget designs at matched chance overlap.

Every arm sees the same outbreaks and the same standard-normal noise draws, scaled by its own sigma, so arms are paired.
For each arm, a grid of thresholds around its validation-calibrated value traces POD against chance overlap.
POD at a fixed chance overlap and MDO interpolated in log size are then read from these curves.
The grid is an evaluation device on the test period, not a tuning step: no setting is changed by it.
"""
import json
from pathlib import Path

import numpy as np

DOC = __doc__
source = Path("scripts/run_detection_experiments.py").read_text(encoding="utf8")
exec(source[:source.index("started = time.time()")])

SIZES = (0, 40, 80, 160, 320)
FACTORS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.85, 2.2])
CHANCE_TARGETS = (0.10, 0.15, 0.20)
TRIALS = setup["trials_per_size"]
RECORD2 = StreamBudget.from_config(cfg, 2, "record").borough_sigma
PERSON2 = StreamBudget.from_config(cfg, 2, "person").borough_sigma
PER_WINDOW2 = RECORD2 * math.sqrt(grid.size)


def z_draw(size: int, trial: int, shape: tuple) -> np.ndarray:
    """The same standard-normal draw for a given outbreak in every arm."""
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], 95, size, trial])).standard_normal(shape)


def cusum_scorer(values, sigma):
    return z_scores(values, sigma, settings)


ARMS = {
    "noiseless": (0.0, naeg_scorer, None),
    "record_eps1": (StreamBudget.from_config(cfg, 1, "record").borough_sigma, naeg_scorer, None),
    "record_eps2": (RECORD2, naeg_scorer, None),
    "record_eps4": (StreamBudget.from_config(cfg, 4, "record").borough_sigma, naeg_scorer, None),
    "person_eps2": (PERSON2, naeg_scorer, None),
    "per_window_eps2": (PER_WINDOW2, naeg_scorer, None),
    "record_eps2_c2": (RECORD2, lambda v, s: c2_scores(v, settings.baseline, settings.guard), None),
    "record_eps2_no_floor": (RECORD2, lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, settings.memory, 0.0, noise_floor=False), neighbors), None),
    "record_eps2_resetting_cusum": (RECORD2, cusum_scorer, None),
}
for k, name in enumerate(context.institutions):
    ARMS[f"single_{name}_noiseless"] = (0.0, naeg_scorer, k)
    ARMS[f"single_{name}_eps2"] = (RECORD2, naeg_scorer, k)


def hits_for(name: str, sigma: float, scorer, k, taus: np.ndarray) -> dict[int, np.ndarray]:
    table = base if k is None else base_cells[:, :, k]
    cusum = name.endswith("resetting_cusum")
    out = {}
    for size in SIZES:
        hits = np.zeros((len(plans[size][:TRIALS]), len(taus)), dtype=bool)
        for i, outbreak in enumerate(plans[size][:TRIALS]):
            if k is not None:
                keep = outbreak.holders == k
                outbreak = type(outbreak)(int(keep.sum()), outbreak.boroughs, outbreak.day_start, outbreak.duration, outbreak.days[keep], outbreak.places[keep], outbreak.holders[keep])
            clean = add_outbreak(table, outbreak, grid)
            values = clean + sigma * z_draw(size, i, clean.shape) if sigma else clean
            begin, stop = outbreak_windows(outbreak, grid)
            places = [context.position[b] for b in outbreak.boroughs]
            if cusum:
                z = scorer(values, sigma)
                for j, tau in enumerate(taus):
                    alarms = resetting_cusum_alarms(z, settings.slack, tau)
                    hits[i, j] = bool(alarms[begin:stop + 1][:, places].any())
            else:
                scores = scorer(values, sigma)
                for j, tau in enumerate(taus):
                    alerts = run_gate(scores, test, context.boroughs, context.adjacency, tau, gate_cfg)
                    hits[i, j] = first_hit(alerts, outbreak, grid) is not None
        out[size] = hits
    return out


def mdo_interpolated(pod_by_size: dict[int, float]) -> float | None:
    """Smallest size where POD reaches the target, interpolating linearly in log2(size)."""
    sizes = [s for s in SIZES if s]
    for lower, upper in zip(sizes, sizes[1:]):
        if pod_by_size[lower] >= target_pod:
            return float(lower) if lower == sizes[0] else None
        if pod_by_size[upper] >= target_pod:
            a, b = pod_by_size[lower], pod_by_size[upper]
            share = (target_pod - a) / (b - a)
            return float(2 ** (np.log2(lower) + share * (np.log2(upper) - np.log2(lower))))
    return None


results = {}
for name, (sigma, scorer, k) in ARMS.items():
    table = base if k is None else base_cells[:, :, k]
    rng = stream(f"matched-{name}")
    tau0 = _resetting_threshold(sigma, rng) if name.endswith("resetting_cusum") else threshold_for(sigma, scorer, rng, table=table)
    taus = tau0 * FACTORS
    hits = hits_for(name, sigma, scorer, k, taus)
    curve = {str(size): [round(float(x), 4) for x in hits[size].mean(axis=0)] for size in SIZES}
    chance = np.array(curve["0"])
    at_target = {}
    for c in CHANCE_TARGETS:
        order = np.argsort(chance)
        pods = {}
        for size in SIZES[1:]:
            pods[size] = round(float(np.interp(c, chance[order], np.array(curve[str(size)])[order])), 3)
        at_target[str(c)] = {"pod": {str(s): v for s, v in pods.items()}, "mdo_interpolated": mdo_interpolated(pods),
                             "within_grid": bool(chance.min() <= c <= chance.max())}
    base_index = int(np.argmin(np.abs(FACTORS - 1.0)))
    c0 = float(chance[base_index])
    adjusted = {str(s): round((curve[str(s)][base_index] - c0) / (1 - c0), 3) for s in SIZES[1:]}
    results[name] = {"sigma": round(sigma, 2), "calibrated_threshold": round(float(tau0), 3), "threshold_factors": FACTORS.tolist(), "curve": curve,
                     "at_calibrated": {"chance": round(c0, 3), "pod": {str(s): curve[str(s)][base_index] for s in SIZES[1:]}, "chance_adjusted_pod": adjusted},
                     "at_matched_chance": at_target}
    print(name, json.dumps({"tau0": results[name]["calibrated_threshold"], "c0": c0, "adj": adjusted, "c=0.15": at_target["0.15"], "c=0.2": at_target["0.2"]}), flush=True)
    OUT.joinpath("matched_rate.json").write_text(json.dumps({"note": DOC, "trials_per_size": TRIALS, "sizes": SIZES, "results": results}, indent=1), encoding="utf8")

relative = {}
expected = np.mean(base[test], axis=0)
for size in SIZES[1:]:
    ratios = []
    for outbreak in plans[size][:TRIALS]:
        begin, stop = outbreak_windows(outbreak, grid)
        places = [context.position[b] for b in outbreak.boroughs]
        ratios.append(size / (expected[places].sum() * (stop - begin + 1)))
    relative[str(size)] = {"median_excess_over_expected_background": round(float(np.median(ratios)), 3),
                           "p10": round(float(np.percentile(ratios, 10)), 3), "p90": round(float(np.percentile(ratios, 90)), 3)}
print(json.dumps(relative))
OUT.joinpath("matched_rate.json").write_text(json.dumps({"note": DOC, "trials_per_size": TRIALS, "sizes": SIZES, "relative_size": relative, "results": results}, indent=1), encoding="utf8")
