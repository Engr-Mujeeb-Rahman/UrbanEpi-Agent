"""Added for the supervisor's comments of 12 November 2026 (19, 20, 21, 22, 23, 24). Post hoc: no frozen setting is changed.

Every arm uses the paired outbreaks and the shared standard-normal noise draws of run_matched_rate.py, so the main arms reproduce
its curves exactly. For each arm this script saves:
- POD and MDO at matched chance overlap with 95% bootstrap intervals (1,000 resamples of trials within each size) (19);
- alerts per 30 days on the real test background, at the calibrated threshold and at the threshold matching chance 0.15 (21);
- a threshold chosen on validation data, with placebo outbreaks in the validation period, for chance 0.15, applied unchanged to test (20).
It adds two standard baselines, run at matched chance overlap like the others (22):
- a Farrington-type detector: per borough, a quasi-Poisson log-linear fit with trend on the 20 baseline windows before the guard,
  the trend dropped when not significant, the known DP noise variance added, and the 2/3-power exceedance score;
- a prospective space-time scan statistic (Kulldorff type, Poisson): clusters of 1 to 3 adjacent boroughs over the last 1 to 7 windows,
  expected counts from the same baseline; the most likely cluster in each window alarms if its log-likelihood ratio passes the threshold.
Both are simplified versions written for this study (no outlier reweighting or seasonal terms, and the scan threshold is set by the same
null simulation as every other detector, not by Monte Carlo replication).
It also repeats the main comparison with the test period ending on 31 March 2025 (23) and with IMSS removed from the background (24).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

DOC = __doc__
source = Path("scripts/run_detection_experiments.py").read_text(encoding="utf8")
exec(source[:source.index("started = time.time()")])

SIZES = (0, 40, 80, 160, 320)
FACTORS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.85, 2.2])
WIDE = np.geomspace(0.25, 6.0, 20)
TRIALS = setup["trials_per_size"]
BOOT = 1000
TARGET_CHANCE = 0.15
RECORD = {e: StreamBudget.from_config(cfg, e, "record").borough_sigma for e in (1, 2, 4)}
PERSON2 = StreamBudget.from_config(cfg, 2, "person").borough_sigma
PER_WINDOW2 = RECORD[2] * math.sqrt(grid.size)
BASELINE, GUARD = settings.baseline, settings.guard
TEST_DAYS = len(test) * grid.window_days


def z_draw(size: int, trial: int, shape: tuple, tag: int = 95) -> np.ndarray:
    """The same standard-normal draw for a given outbreak in every arm (tag 95 matches run_matched_rate.py)."""
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], tag, size, trial])).standard_normal(shape)


# ---------- two added baselines ----------

def farrington_scores(values: np.ndarray, sigma: float) -> np.ndarray:
    """Farrington-type exceedance score per (window, borough); NaN where the baseline is incomplete."""
    T, D = values.shape
    out = np.full((T, D), np.nan)
    x = np.arange(BASELINE, dtype=float) - (BASELINE - 1) / 2
    X = np.column_stack([np.ones(BASELINE), x])
    for t in range(max(BASELINE + GUARD, int(validation[0])), T):
        y = np.clip(values[t - GUARD - BASELINE:t - GUARD], 0, None)
        mu_rows = np.empty(D)
        var_fit = np.empty(D)
        phi = np.empty(D)
        for d in range(D):
            yd = y[:, d]
            m = max(yd.mean(), 0.5)
            beta_ = np.array([np.log(m), 0.0])
            for _ in range(8):
                eta = X @ beta_
                mu = np.exp(eta)
                W = mu
                z = eta + (yd - mu) / mu
                XtW = X.T * W
                beta_ = np.linalg.solve(XtW @ X + 1e-9 * np.eye(2), XtW @ z)
            mu = np.exp(X @ beta_)
            p = 2
            ph = max(1.0, float(((yd - mu) ** 2 / mu).sum() / (BASELINE - p)))
            cov = ph * np.linalg.inv((X.T * mu) @ X)
            if abs(beta_[1]) / np.sqrt(cov[1, 1]) < 1.96:
                m = max(yd.mean(), 0.5)
                ph = max(1.0, float(((yd - m) ** 2 / m).sum() / (BASELINE - 1)))
                mu_rows[d], var_fit[d], phi[d] = m, ph * m / BASELINE, ph
            else:
                xt = BASELINE - 1 + GUARD + 1 - (BASELINE - 1) / 2
                xv = np.array([1.0, xt])
                mu_rows[d] = float(np.exp(xv @ beta_))
                var_fit[d] = float(mu_rows[d] ** 2 * (xv @ cov @ xv))
                phi[d] = ph
        mu_rows = np.clip(mu_rows, 0.5, None)
        var_y = phi * mu_rows + var_fit + sigma ** 2
        y_now = np.clip(values[t], 0, None)
        out[t] = (y_now ** (2 / 3) - mu_rows ** (2 / 3)) / ((2 / 3) * mu_rows ** (-1 / 3) * np.sqrt(var_y))
    return out


def _clusters() -> np.ndarray:
    """Membership matrix of connected sets of 1 to 3 boroughs."""
    pos = context.position
    nbr = {b: set(context.adjacency.get(b, [])) for b in context.boroughs}
    sets = {frozenset([b]) for b in context.boroughs}
    for b in context.boroughs:
        for c in nbr[b]:
            sets.add(frozenset([b, c]))
            for e in nbr[b] | nbr[c]:
                if e not in (b, c):
                    sets.add(frozenset([b, c, e]))
    members = np.zeros((len(sets), len(context.boroughs)))
    for i, s in enumerate(sorted(sets, key=lambda s: sorted(s))):
        members[i, [pos[b] for b in s]] = 1
    return members


CLUSTERS = _clusters()
DURATIONS = range(1, 8)


def scan_scores(values: np.ndarray, sigma: float) -> np.ndarray:
    """Score per (window, borough): the LLR of the most likely cluster ending at that window, on its boroughs; 0 elsewhere."""
    T, D = values.shape
    y = np.clip(values, 0, None)
    out = np.full((T, D), np.nan)
    csum = np.vstack([np.zeros(D), np.cumsum(y, axis=0)])
    for t in range(BASELINE + GUARD + 6, T):
        mu = np.clip(y[t - GUARD - BASELINE:t - GUARD].mean(axis=0), 0.5, None)
        best, where = 0.0, None
        for w in DURATIONS:
            observed = CLUSTERS @ (csum[t + 1] - csum[t + 1 - w])
            expected = CLUSTERS @ (mu * w)
            llr = np.where(observed > expected, observed * np.log(np.maximum(observed, 1e-9) / expected) - (observed - expected), 0.0)
            i = int(np.argmax(llr))
            if llr[i] > best:
                best, where = float(llr[i]), i
        out[t] = 0.0
        if where is not None:
            out[t, CLUSTERS[where] > 0] = best
    return out


# ---------- arms ----------

def cusum_alarms(values, sigma, tau):
    return resetting_cusum_alarms(z_scores(values, sigma, settings), settings.slack, tau)


ARMS = {
    "noiseless": (0.0, naeg_scorer, None, "gate"),
    "record_eps1": (RECORD[1], naeg_scorer, None, "gate"),
    "record_eps2": (RECORD[2], naeg_scorer, None, "gate"),
    "record_eps4": (RECORD[4], naeg_scorer, None, "gate"),
    "person_eps2": (PERSON2, naeg_scorer, None, "gate"),
    "per_window_eps2": (PER_WINDOW2, naeg_scorer, None, "gate"),
    "record_eps2_c2": (RECORD[2], lambda v, s: c2_scores(v, settings.baseline, settings.guard), None, "gate"),
    "record_eps2_no_floor": (RECORD[2], lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, settings.memory, 0.0, noise_floor=False), neighbors), None, "gate"),
    "record_eps2_resetting_cusum": (RECORD[2], None, None, "cusum"),
    "noiseless_farrington": (0.0, farrington_scores, None, "cells"),
    "record_eps2_farrington": (RECORD[2], farrington_scores, None, "cells"),
    "noiseless_scan": (0.0, scan_scores, None, "cells"),
    "record_eps2_scan": (RECORD[2], scan_scores, None, "cells"),
}


def alarms_at(kind, scores, values, sigma, tau):
    """Return either a list of (window, boroughs) gate alerts or a boolean (T, D) alarm matrix."""
    if kind == "gate":
        return run_gate(scores, test_windows, context.boroughs, context.adjacency, tau, gate_cfg)
    if kind == "cusum":
        return cusum_alarms(values, sigma, tau)
    return np.nan_to_num(scores, nan=-np.inf) >= tau


def hit(kind, alarm, outbreak, windows) -> bool:
    if kind == "gate":
        return first_hit(alarm, outbreak, grid) is not None
    begin, stop = outbreak_windows(outbreak, grid)
    places = [context.position[b] for b in outbreak.boroughs]
    rows = [t for t in range(begin, stop + 1) if t in windows]
    return bool(alarm[rows][:, places].any()) if rows else False


def count_alarms(kind, alarm, windows) -> int:
    if kind == "gate":
        return len(alarm)
    return int(alarm[windows].sum())


def calibrate_wide(sigma, scorer, rng, table):
    """The same null-simulation rule as calibrate(), over a wider threshold grid for scores on other scales."""
    from urbanepi.gate.c2 import alert_onsets
    from urbanepi.gate.naeg import estimate_null, simulate_null
    released = noisy(table[:validation_end], sigma, rng)
    level, dispersion = estimate_null(released[validation], sigma)
    runs = []
    for _ in range(gate_cfg["null_seeds"]):
        simulated = simulate_null(rng, level, dispersion, validation_end)
        runs.append(scorer(noisy(simulated, sigma, rng), sigma))
    days = len(validation) * grid.window_days
    for candidate in np.geomspace(0.05, 1e5, 2000):
        if np.mean([alert_onsets(r, candidate)[validation].sum() for r in runs]) / (days / 30) <= gate_cfg["null_alerts_per_30_days"]:
            return float(candidate)
    return float("inf")


def tau_for(name, sigma, scorer, kind, table):
    rng = stream(f"matched-{name}")
    if kind == "cusum":
        return _resetting_threshold(sigma, rng)
    if kind == "cells":
        return calibrate_wide(sigma, scorer, rng, table)
    return threshold_for(sigma, scorer, rng, table=table)


def curves(name, sigma, scorer, kind, table, taus, plan, windows, tag=95):
    """Hits per (trial, threshold) for every size."""
    out = {}
    for size in SIZES:
        trials = plan[size][:TRIALS]
        hits = np.zeros((len(trials), len(taus)), dtype=bool)
        for i, outbreak in enumerate(trials):
            clean = add_outbreak(table, outbreak, grid)
            values = clean + sigma * z_draw(size, i, clean.shape, tag) if sigma else clean
            scores = scorer(values, sigma) if scorer is not None else None
            for j, tau in enumerate(taus):
                hits[i, j] = hit(kind, alarms_at(kind, scores, values, sigma, tau), outbreak, windows)
        out[size] = hits
    return out


def interp_at_chance(chance, values, c=TARGET_CHANCE):
    order = np.argsort(chance)
    return float(np.interp(c, chance[order], values[order]))


def mdo_from(pods: dict) -> float:
    sizes = [s for s in SIZES if s]
    for lower, upper in zip(sizes, sizes[1:]):
        if pods[lower] >= target_pod:
            return float(lower) if lower == sizes[0] else float("nan")
        if pods[upper] >= target_pod:
            a, b = pods[lower], pods[upper]
            share = (target_pod - a) / (b - a)
            return float(2 ** (np.log2(lower) + share * (np.log2(upper) - np.log2(lower))))
    return float("inf")


def summarize(hits: dict, rng: np.random.Generator) -> dict:
    curve = {s: hits[s].mean(axis=0) for s in SIZES}
    pods = {s: interp_at_chance(curve[0], curve[s]) for s in SIZES[1:]}
    point = {"pod_80": round(pods[80], 3), "mdo": mdo_from(pods)}
    boot_pod, boot_mdo = [], []
    for _ in range(BOOT):
        c = {s: hits[s][rng.integers(0, len(hits[s]), len(hits[s]))].mean(axis=0) for s in SIZES}
        p = {s: interp_at_chance(c[0], c[s]) for s in SIZES[1:]}
        boot_pod.append(p[80])
        boot_mdo.append(mdo_from(p))
    boot_mdo = np.array(boot_mdo)
    finite = np.where(np.isnan(boot_mdo), np.inf, boot_mdo)
    point["pod_80_ci"] = [round(float(np.percentile(boot_pod, 2.5)), 3), round(float(np.percentile(boot_pod, 97.5)), 3)]
    point["mdo_ci"] = [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
    point["within_grid"] = bool(curve[0].min() <= TARGET_CHANCE <= curve[0].max())
    point["curve"] = {str(s): [round(float(x), 4) for x in curve[s]] for s in SIZES}
    return point


def threshold_at_chance(taus, chance, c=TARGET_CHANCE):
    """Threshold whose chance overlap is c, interpolating chance (which falls with tau) in log tau."""
    order = np.argsort(chance)
    return float(np.exp(np.interp(c, chance[order], np.log(taus)[order])))


def background_alarms(name, sigma, scorer, kind, table, tau, windows, draws=20) -> float:
    """Alarms per 30 days on the real background with no injected outbreak, averaged over noise draws."""
    counts = []
    for k in range(draws if sigma else 1):
        values = table + sigma * z_draw(9999, k, table.shape, 97) if sigma else table
        scores = scorer(values, sigma) if scorer is not None else None
        counts.append(count_alarms(kind, alarms_at(kind, scores, values, sigma, tau), windows))
    return round(float(np.mean(counts)) / (len(windows) * grid.window_days / 30), 2)


def validation_chance(name, sigma, scorer, kind, table, taus, plan_val) -> np.ndarray:
    """Chance overlap on validation placebo outbreaks, using only data up to the end of validation."""
    global test_windows
    saved = test_windows
    test_windows = validation
    end = validation_end
    hits = np.zeros((len(plan_val[0]), len(taus)), dtype=bool)
    for i, outbreak in enumerate(plan_val[0]):
        clean = table[:end].copy()
        values = clean + sigma * z_draw(0, i, clean.shape, 96) if sigma else clean
        scores = scorer(values, sigma) if scorer is not None else None
        for j, tau in enumerate(taus):
            hits[i, j] = hit(kind, alarms_at(kind, scores, values, sigma, tau), outbreak, set(validation.tolist()))
    test_windows = saved
    return hits.mean(axis=0)


test_windows = test
plan_val = context.draw_outbreaks(np.random.default_rng(np.random.SeedSequence([cfg["seed"], 98])), [0], TRIALS, split="validation")
results = {}
only = set(sys.argv[1:])
for name, (sigma, scorer, k, kind) in ARMS.items():
    if only and name not in only:
        continue
    table = base
    tau0 = tau_for(name, sigma, scorer, kind, table)
    taus = tau0 * (WIDE if kind == "cells" else FACTORS)
    hits = curves(name, sigma, scorer, kind, table, taus, plans, set(test.tolist()))
    row = {"sigma": round(sigma, 2), "kind": kind, "calibrated_threshold": round(float(tau0), 3), "thresholds": [round(float(t), 4) for t in taus]}
    row["matched"] = summarize(hits, np.random.default_rng(np.random.SeedSequence([cfg["seed"], 99, *name.encode()])))
    base_index = int(np.argmin(np.abs(taus - tau0)))
    chance_curve = np.array(row["matched"]["curve"]["0"])
    row["at_calibrated"] = {"chance": round(float(chance_curve[base_index]), 3), "pod_80": row["matched"]["curve"]["80"][base_index]}
    tau15 = threshold_at_chance(taus, chance_curve) if row["matched"]["within_grid"] else None
    row["threshold_at_chance_0.15"] = round(tau15, 4) if tau15 else None
    row["background_alarms_per_30_days"] = {"at_calibrated": background_alarms(name, sigma, scorer, kind, table, tau0, test),
                                            "at_chance_0.15": background_alarms(name, sigma, scorer, kind, table, tau15, test) if tau15 else None}
    val_chance = validation_chance(name, sigma, scorer, kind, table, taus, plan_val)
    row["validation_chance_curve"] = [round(float(x), 4) for x in val_chance]
    if val_chance.min() <= TARGET_CHANCE <= val_chance.max():
        tau_v = threshold_at_chance(taus, val_chance)
        logt = np.log(taus)
        test_at = {str(s): round(float(np.interp(np.log(tau_v), logt, np.array(row["matched"]["curve"][str(s)]))), 3) for s in SIZES}
        row["validation_chosen"] = {"threshold": round(tau_v, 4), "test_chance": test_at["0"], "test_pod": {s: test_at[s] for s in test_at if s != "0"},
                                    "test_mdo": mdo_from({int(s): v for s, v in test_at.items() if s != "0"})}
    else:
        row["validation_chosen"] = None
    results[name] = row
    print(name, json.dumps({k2: v for k2, v in row.items() if k2 not in ("thresholds", "validation_chance_curve")} | {"matched": {k3: v3 for k3, v3 in row["matched"].items() if k3 != "curve"}}), flush=True)
    OUT.joinpath("supervisor_checks.json").write_text(json.dumps({"note": DOC, "trials_per_size": TRIALS, "bootstrap": BOOT, "results": results}, indent=1), encoding="utf8")

# ---------- sensitivity: shorter test period (23) and IMSS removed (24) ----------
if not only or "sensitivity" in only:
    sensitivity = {}
    cutoff = int(grid.split_windows("2024-11-01", "2025-03-31")[-1])
    short_test = test[test <= cutoff]
    short_plans = {s: [o for o in plans[s][:TRIALS] if outbreak_windows(o, grid)[1] <= cutoff] for s in SIZES}
    without_imss = base - base_cells[:, :, context.institutions.index("IMSS")]
    cases = {"test_to_2025_03_31": (base, short_test, short_plans), "without_imss": (without_imss, test, plans)}
    for label, (table, windows, plan) in cases.items():
        test_windows = windows
        for name in ("noiseless", "record_eps2"):
            sigma, scorer, _, kind = ARMS[name]
            tau0 = threshold_for(sigma, scorer, stream(f"matched-{name}"), table=table)
            taus = tau0 * FACTORS
            hits = curves(name, sigma, scorer, kind, table, taus, plan, set(windows.tolist()))
            row = summarize(hits, np.random.default_rng(np.random.SeedSequence([cfg["seed"], 100, *label.encode(), *name.encode()])))
            row["trials_per_size"] = {str(s): len(plan[s][:TRIALS]) for s in SIZES}
            row["test_windows"] = len(windows)
            sensitivity[f"{label}/{name}"] = row
            print(label, name, json.dumps({k2: v for k2, v in row.items() if k2 != "curve"}), flush=True)
    test_windows = test
    old = json.loads(OUT.joinpath("supervisor_checks.json").read_text(encoding="utf8")) if OUT.joinpath("supervisor_checks.json").exists() else {"results": results}
    old["sensitivity"] = sensitivity
    old["note"], old["trials_per_size"], old["bootstrap"] = DOC, TRIALS, BOOT
    OUT.joinpath("supervisor_checks.json").write_text(json.dumps(old, indent=1), encoding="utf8")
