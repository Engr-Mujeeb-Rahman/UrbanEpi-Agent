"""Final detection experiments on the frozen test period. Usage: run_detection_experiments.py e3 e4 e5 e7 e8

Settings and the selection of gate parameters were frozen before these runs (experiments/frozen/).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from urbanepi.city.windows import WindowGrid
from urbanepi.config import load_config
from urbanepi.evaluation.detection import (DetectionContext, add_outbreak, calibrate, exact_mcnemar, first_hit, holm, minimum_detectable,
                                           outbreak_windows, run_gate, wilson)
from urbanepi.gate.c2 import c2_scores
from urbanepi.gate.naeg import GateScoreSettings, adjacency_matrix, gate_scores, resetting_cusum_alarms, z_scores
from urbanepi.privacy.accounting import StreamBudget, rho_from_epsilon, sigma_for

cfg = load_config()
gate_cfg, setup = cfg["gate"], cfg["final_detection"]
OUT = Path("experiments/results/final")
OUT.mkdir(parents=True, exist_ok=True)
context = DetectionContext(cfg)
grid = WindowGrid.from_config(cfg)
base_cells = context.counts(grid)
base = base_cells.sum(axis=2)
validation = context.windows_of(grid, "validation")
test = context.windows_of(grid, "test")
validation_end = int(validation[-1]) + 1
neighbors = adjacency_matrix(context.boroughs, context.adjacency)
settings = GateScoreSettings(gate_cfg["baseline_windows"], gate_cfg["guard_windows"], gate_cfg["slack"], gate_cfg["memory_windows"], gate_cfg["spatial_weight"])
master = np.random.SeedSequence([cfg["seed"], 60])
sizes = setup["excess_cases"]
plans = context.draw_outbreaks(np.random.default_rng(master.spawn(1)[0]), sizes, setup["trials_per_size"])
target_pod = setup["target_pod"]


def stream(name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], 61, *name.encode()]))


def noisy(values: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return values + rng.normal(0, sigma, values.shape) if sigma else values


def threshold_for(sigma: float, scorer, rng: np.random.Generator, table: np.ndarray = base, windows: np.ndarray = validation, window_days: int = 3) -> float:
    end = int(windows[-1]) + 1
    released = noisy(table[:end], sigma, rng)
    return calibrate(released, windows, sigma, scorer, gate_cfg["null_alerts_per_30_days"], window_days, gate_cfg["null_seeds"], rng)


def naeg_scorer(values: np.ndarray, sigma: float) -> np.ndarray:
    return gate_scores(values, sigma, settings, neighbors)


def references(tau: float) -> list[tuple[int, set[str]]]:
    """Alerts from the same detector on true counts, standing in for real waves."""
    return run_gate(naeg_scorer(base, 0.0), test, context.boroughs, context.adjacency, tau, gate_cfg)


def unmatched_rate(alerts: list, refs: list) -> float:
    """Alerts per 30 days matching no reference alert in the same borough within two windows."""
    count = sum(not any(abs(t - r) <= 2 and where & other for r, other in refs) for t, where in alerts)
    return count / (len(test) * grid.window_days / 30)


def evaluate(sigma: float, tau: float, rng: np.random.Generator, refs: list, table: np.ndarray = base, scorer=naeg_scorer) -> dict:
    """Run the full gate over paired outbreaks. Returns POD, delay, calls, and unmatched alerts."""
    pod, ci, delay, calls, hits_by_size = {}, {}, {}, {}, {}
    unmatched = []
    for size in sizes:
        hits = np.zeros(len(plans[size]), dtype=bool)
        delays, used = [], []
        for i, outbreak in enumerate(plans[size]):
            values = noisy(add_outbreak(table, outbreak, grid), sigma, rng)
            alerts = run_gate(scorer(values, sigma), test, context.boroughs, context.adjacency, tau, gate_cfg)
            used.append(len(alerts))
            found = first_hit(alerts, outbreak, grid)
            if found is not None:
                hits[i] = True
                delays.append(found - outbreak_windows(outbreak, grid)[0])
            if size == 0:
                unmatched.append(unmatched_rate(alerts, refs))
        pod[size], ci[size] = round(float(hits.mean()), 3), wilson(int(hits.sum()), len(hits))
        delay[size] = round(float(np.mean(delays)), 2) if delays else None
        calls[size] = round(float(np.mean(used)), 2)
        hits_by_size[size] = hits
    return {"sigma": round(sigma, 2), "threshold": round(tau, 2), "pod": pod, "ci": ci, "mdo": minimum_detectable(pod, target_pod), "delay_windows": delay,
            "calls_per_test_period": calls, "unmatched_alerts_per_30_days": round(float(np.mean(unmatched)), 2), "_hits": hits_by_size}


def save(name: str, result: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items() if str(k) != "_hits"}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(clean({"note": "Frozen test period.", "trials_per_size": setup["trials_per_size"], "seed": cfg["seed"], **result}), indent=1), encoding="utf8")


def line(label: str, row: dict) -> None:
    print(f"{label:34s} s={row['sigma']:6.2f} t={row['threshold']:5.2f} MDO={row['mdo']} unm={row['unmatched_alerts_per_30_days']:4.1f} calls={row['calls_per_test_period'][0]:4.1f} "
          + " ".join(f"{s}:{row['pod'][s]:.2f}" for s in sizes), flush=True)


def e3() -> None:
    results, rng = {}, stream("e3")
    reference_tau = threshold_for(0.0, naeg_scorer, stream("reference"))
    refs = references(reference_tau)
    results["central_no_noise"] = evaluate(0.0, reference_tau, rng, refs)
    line("no noise", results["central_no_noise"])
    for unit in ("record", "person"):
        for epsilon in cfg["privacy"]["epsilons"]:
            sigma = StreamBudget.from_config(cfg, epsilon, unit).borough_sigma
            tau = threshold_for(sigma, naeg_scorer, stream(f"e3-{unit}-{epsilon}"))
            results[f"{unit}_eps{epsilon}"] = evaluate(sigma, tau, rng, refs)
            line(f"{unit} eps={epsilon}", results[f"{unit}_eps{epsilon}"])
    sigma = StreamBudget.from_config(cfg, 2, "record").borough_sigma
    for k, name in enumerate(context.institutions):
        table = base_cells[:, :, k]
        tau = threshold_for(sigma, naeg_scorer, stream(f"e3-{name}"), table=table)
        row = {}
        pod, ci = {}, {}
        for size in sizes:
            hits = []
            for outbreak in plans[size]:
                keep = outbreak.holders == k
                single = type(outbreak)(int(keep.sum()), outbreak.boroughs, outbreak.day_start, outbreak.duration, outbreak.days[keep], outbreak.places[keep], outbreak.holders[keep])
                values = noisy(add_outbreak(table, single, grid), sigma, rng)
                alerts = run_gate(naeg_scorer(values, sigma), test, context.boroughs, context.adjacency, tau, gate_cfg)
                hits.append(first_hit(alerts, outbreak, grid) is not None)
            pod[size], ci[size] = round(float(np.mean(hits)), 3), wilson(int(np.sum(hits)), len(hits))
        row = {"sigma": round(sigma, 2), "threshold": round(tau, 2), "pod": pod, "ci": ci, "mdo": minimum_detectable(pod, target_pod)}
        results[f"single_{name}_eps2"] = row
        print(f"single {name:22s} t={tau:5.2f} MDO={row['mdo']} " + " ".join(f"{s}:{pod[s]:.2f}" for s in sizes), flush=True)

    borough_rng = stream("e3-borough")
    tau = threshold_for(sigma, naeg_scorer, stream("e3-record-2"))
    by_borough = {}
    for code in context.boroughs:
        pod = {}
        for size in setup["borough_sizes"]:
            outbreaks = context.single_borough_outbreaks(borough_rng, code, size, setup["trials_per_borough"])
            hits = 0
            for outbreak in outbreaks:
                values = noisy(add_outbreak(base, outbreak, grid), sigma, borough_rng)
                hits += first_hit(run_gate(naeg_scorer(values, sigma), test, context.boroughs, context.adjacency, tau, gate_cfg), outbreak, grid) is not None
            pod[size] = round(hits / len(outbreaks), 3)
        by_borough[code] = {"test_mean_per_window": round(float(base[test, context.position[code]].mean()), 1), "pod": pod,
                            "mdo": minimum_detectable(pod, target_pod)}
        print(code, by_borough[code], flush=True)
    results["per_borough_record_eps2"] = by_borough
    save("e3", {"results": results})


def e4() -> None:
    results, comparisons = {}, {}
    rng = stream("e4")
    refs = references(threshold_for(0.0, naeg_scorer, stream("reference")))
    institutions = len(context.institutions)
    for epsilon in setup["e4_epsilons"]:
        record = StreamBudget.from_config(cfg, epsilon, "record")
        person = StreamBudget.from_config(cfg, epsilon, "person")
        per_window = sigma_for(record.borough_rho / grid.size, 1.0)
        designs = {"record_distributed": record.borough_sigma, "record_per_institution": record.borough_sigma * math.sqrt(institutions),
                   "person_distributed": person.borough_sigma, "person_per_institution": person.borough_sigma * math.sqrt(institutions),
                   "per_window_distributed": per_window, "per_window_per_institution": per_window * math.sqrt(institutions)}
        rows = {}
        for name, sigma in designs.items():
            tau = threshold_for(sigma, naeg_scorer, stream(f"e4-{name}-{epsilon}"))
            rows[name] = evaluate(sigma, tau, rng, refs)
            line(f"eps={epsilon} {name}", rows[name])
        p_values = {}
        for size in setup["comparison_sizes"]:
            for other in ("per_window_distributed", "per_window_per_institution", "person_per_institution"):
                ours = "person_distributed"
                p_values[f"{ours} vs {other}, A={size}"] = exact_mcnemar(rows[ours]["_hits"][size], rows[other]["_hits"][size])
        comparisons[str(epsilon)] = holm(p_values)
        results[f"eps{epsilon}"] = rows
        print("Holm", comparisons[str(epsilon)], flush=True)
    budget = StreamBudget.from_config(cfg, 2, "record")
    local = {code: round(budget.borough_sigma * math.sqrt(context.populations[code]), 0) for code in context.boroughs}
    save("e4", {"results": results, "holm_mcnemar": comparisons,
                "local_dp_analytic_sd_at_eps2": {"note": "Each resident adds noise, so the SD grows with the root of population. Not simulated.", "values": local}})


def e5() -> None:
    results = {}
    rng = stream("e5")
    refs = references(threshold_for(0.0, naeg_scorer, stream("reference")))
    arms = {
        "naeg": naeg_scorer,
        "c2_no_noise_floor": lambda v, s: c2_scores(v, settings.baseline, settings.guard),
        "minus_floor": lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, settings.memory, 0.0, noise_floor=False), neighbors),
        "minus_memory": lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, 1, 0.0), neighbors),
        "plus_spatial": lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, settings.memory, 0.5), neighbors),
        "running_cusum": lambda v, s: gate_scores(v, s, GateScoreSettings(settings.baseline, settings.guard, settings.slack, None, 0.0), neighbors),
    }
    for epsilon in setup["e5_epsilons"]:
        sigma = StreamBudget.from_config(cfg, epsilon, "record").borough_sigma
        rows = {}
        for name, scorer in arms.items():
            tau = threshold_for(sigma, scorer, stream(f"e5-{name}-{epsilon}"))
            rows[name] = evaluate(sigma, tau, rng, refs, scorer=scorer)
            line(f"eps={epsilon} {name}", rows[name])
        naeg, tau = rows["naeg"], rows["naeg"]["threshold"]

        no_gate, cusum, periodic, random_calls = {}, {}, {}, {}
        calls = max(1, int(round(naeg["calls_per_test_period"][0])))
        every = max(1, len(test) // calls)
        chosen = test[::every]
        cusum_tau = _resetting_threshold(sigma, stream(f"e5-reset-{epsilon}"))
        call_counts = {"no_gate": [], "resetting_cusum": []}
        for size in sizes:
            hit_all, hit_reset, hit_periodic, hit_random = [], [], [], []
            for outbreak in plans[size]:
                values = noisy(add_outbreak(base, outbreak, grid), sigma, rng)
                scores = naeg_scorer(values, sigma)
                begin, stop = outbreak_windows(outbreak, grid)
                places = [context.position[b] for b in outbreak.boroughs]
                above = np.nan_to_num(scores, nan=-np.inf) >= tau
                hit_all.append(bool(above[begin:stop + 1][:, places].any()))
                picks = rng.choice(test, size=len(chosen), replace=False)
                hit_periodic.append(any(begin <= t <= stop and above[t, places].any() for t in chosen))
                hit_random.append(any(begin <= t <= stop and above[t, places].any() for t in picks))
                alarms = resetting_cusum_alarms(z_scores(values, sigma, settings), settings.slack, cusum_tau)
                hit_reset.append(bool(alarms[begin:stop + 1][:, places].any()))
                if size == 0:
                    call_counts["no_gate"].append(int(above[test].sum()))
                    call_counts["resetting_cusum"].append(int(alarms[test].sum()))
            no_gate[size], cusum[size] = round(float(np.mean(hit_all)), 3), round(float(np.mean(hit_reset)), 3)
            periodic[size], random_calls[size] = round(float(np.mean(hit_periodic)), 3), round(float(np.mean(hit_random)), 3)
        drop = {s: round(no_gate[s] - naeg["pod"][s], 3) for s in sizes if s}
        results[f"eps{epsilon}"] = {
            **rows,
            "always_on": {"calls_per_test_period": len(test), "note": "One LLM call per window; has no detector of its own."},
            "every_alert_cell_no_merging": {"pod": no_gate, "calls_without_outbreak": round(float(np.mean(call_counts["no_gate"])), 2), "threshold": round(tau, 2)},
            "resetting_cusum": {"pod": cusum, "threshold": round(cusum_tau, 2), "calls_without_outbreak": round(float(np.mean(call_counts["resetting_cusum"])), 2)},
            "periodic_same_calls": {"pod": periodic, "calls": len(chosen)},
            "random_same_calls": {"pod": random_calls, "calls": len(chosen)},
            "call_reduction_vs_always_on": round(1 - naeg["calls_per_test_period"][0] / len(test), 3),
            "pod_drop_vs_no_merging": drop,
        }
        print(f"eps={epsilon} reduction={results[f'eps{epsilon}']['call_reduction_vs_always_on']} drop={drop}\n  no gate {no_gate}\n  reset cusum t={cusum_tau:.2f} {cusum}\n  periodic {periodic}\n  random {random_calls}", flush=True)
    save("e5", {"results": results})


def _resetting_threshold(sigma: float, rng: np.random.Generator) -> float:
    """Same alert-rate target as every other detector, for the resetting CUSUM."""
    from urbanepi.gate.naeg import estimate_null, simulate_null
    released = noisy(base[:validation_end], sigma, rng)
    level, dispersion = estimate_null(released[validation], sigma)
    runs = []
    for _ in range(gate_cfg["null_seeds"]):
        simulated = simulate_null(rng, level, dispersion, validation_end)
        runs.append(z_scores(noisy(simulated, sigma, rng), sigma, settings))
    days = len(validation) * grid.window_days
    for candidate in np.arange(0.5, 40.01, 0.1):
        rate = np.mean([resetting_cusum_alarms(z, settings.slack, candidate)[validation].sum() for z in runs]) / (days / 30)
        if rate <= gate_cfg["null_alerts_per_30_days"]:
            return float(candidate)
    return float("inf")


def e7() -> None:
    results = {}
    rng = stream("e7")
    refs = references(threshold_for(0.0, naeg_scorer, stream("reference")))
    keep_only = setup["e7_sizes"]
    saved = dict(plans)
    globals()["plans"] = {s: saved[s][:setup["e7_trials"]] for s in keep_only}
    globals()["sizes"] = keep_only
    for assumed in setup["e7_assumed_records"]:
        changed = {**cfg, "privacy": {**cfg["privacy"], "assumed_records_per_person": assumed}}
        sigma = StreamBudget.from_config(changed, 2, "person").borough_sigma
        tau = threshold_for(sigma, naeg_scorer, stream(f"e7-L{assumed}"))
        results[f"person_L{assumed}"] = evaluate(sigma, tau, rng, refs)
        line(f"L={assumed}", results[f"person_L{assumed}"])
    person = StreamBudget.from_config(cfg, 2, "person").borough_sigma
    tau = threshold_for(person * 2, naeg_scorer, stream("e7-m2"))
    results["person_L4_two_institutions_per_person"] = evaluate(person * 2, tau, rng, refs)
    line("m_max=2", results["person_L4_two_institutions_per_person"])

    record = StreamBudget.from_config(cfg, 2, "record").borough_sigma
    without = base - base_cells[:, :, context.institutions.index("PEMEX")]
    tau = threshold_for(record, naeg_scorer, stream("e7-pemex"), table=without)
    results["record_without_pemex"] = evaluate(record, tau, rng, refs, table=without)
    line("without PEMEX", results["record_without_pemex"])

    for days in setup["e7_window_days"]:
        other = WindowGrid.from_config(cfg, days)
        table = context.counts(other).sum(axis=2)
        windows_v, windows_t = context.windows_of(other, "validation"), context.windows_of(other, "test")
        scale = 3 / days
        local = GateScoreSettings(max(round(settings.baseline * scale), 6), max(round(settings.guard * scale), 1), settings.slack, max(round(settings.memory * scale), 1), 0.0)
        scorer = lambda v, s, st=local: gate_scores(v, s, st, neighbors)
        tau = threshold_for(record, scorer, stream(f"e7-dt{days}"), table=table, windows=windows_v, window_days=days)
        pod = {}
        for size in keep_only:
            hits = []
            for outbreak in plans[size]:
                values = noisy(add_outbreak(table, outbreak, other), record, rng)
                alerts = run_gate(scorer(values, record), windows_t, context.boroughs, context.adjacency, tau, gate_cfg)
                hits.append(first_hit(alerts, outbreak, other) is not None)
            pod[size] = round(float(np.mean(hits)), 3)
        results[f"window_{days}_days"] = {"threshold": round(tau, 2), "pod": pod, "mdo": minimum_detectable(pod, target_pod)}
        print(f"window {days} days t={tau:.2f} MDO={results[f'window_{days}_days']['mdo']} {pod}", flush=True)
    globals()["plans"], globals()["sizes"] = saved, setup["excess_cases"]
    save("e7", {"results": results, "trials_per_size": setup["e7_trials"]})


def e8() -> None:
    """Does privacy delay detection of the real waves? The reference is the same gate on true counts."""
    rng = stream("e8")
    reference_tau = threshold_for(0.0, naeg_scorer, stream("reference"))
    refs = references(reference_tau)
    distinct: list[tuple[int, set[str]]] = []
    for t, where in sorted(refs, key=lambda r: r[0]):
        if not any(where & other and t - first <= setup["wave_gap_windows"] for first, other in distinct):
            distinct.append((t, where))
    results = {"reference_threshold": round(reference_tau, 2), "reference_alerts": [{"window_start": str(grid.first_day(t).date()), "boroughs": sorted(w)} for t, w in distinct]}
    print("reference waves", len(distinct), [(str(grid.first_day(t).date()), len(w)) for t, w in distinct], flush=True)
    for unit, epsilon in setup["e8_settings"]:
        sigma = StreamBudget.from_config(cfg, epsilon, unit).borough_sigma
        tau = threshold_for(sigma, naeg_scorer, stream(f"e8-{unit}-{epsilon}"))
        delays, found = [], np.zeros((setup["e8_seeds"], len(distinct)), dtype=bool)
        for seed in range(setup["e8_seeds"]):
            alerts = run_gate(naeg_scorer(noisy(base, sigma, rng), sigma), test, context.boroughs, context.adjacency, tau, gate_cfg)
            for i, (t, where) in enumerate(distinct):
                matching = [a for a, other in alerts if other & where and t - setup["wave_gap_windows"] <= a <= t + setup["wave_gap_windows"]]
                if matching:
                    found[seed, i] = True
                    delays.append(min(matching, key=lambda a: abs(a - t)) - t)
        row = {"sigma": round(sigma, 2), "threshold": round(tau, 2), "share_found": round(float(found.mean()), 3),
               "per_wave_share_found": [round(float(x), 2) for x in found.mean(axis=0)],
               "mean_delay_windows": round(float(np.mean(delays)), 2) if delays else None, "median_delay_windows": float(np.median(delays)) if delays else None}
        results[f"{unit}_eps{epsilon}"] = row
        print(unit, epsilon, row, flush=True)
    save("e8", {"results": results, "trials_per_size": None})


started = time.time()
for name in sys.argv[1:]:
    print(f"\n=== {name} ===", flush=True)
    globals()[name]()
    print(f"{name} finished at {round((time.time() - started) / 60, 1)} min", flush=True)
