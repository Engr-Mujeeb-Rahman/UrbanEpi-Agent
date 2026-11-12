"""Does triage tell known-false alerts from known-true ones? Validation-level check, not a paper result.

Both groups use the same simulated outbreak-free background, so labels are certain:
- known false: alerts on background alone
- known true: alerts overlapping an outbreak injected into that background
Usage: triage_check.py [prompt_id]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config
from urbanepi.gate.events import EscalationGate
from urbanepi.gate.naeg import GateScoreSettings, adjacency_matrix, estimate_null, gate_scores, simulate_null, z_scores
from urbanepi.gate.thresholds import null_threshold
from urbanepi.llm.adapter import CachedLLM
from urbanepi.llm.gemini import make_provider_call
from urbanepi.llm.triage import build_evidence, triage_event
from urbanepi.privacy.accounting import StreamBudget

cfg = load_config()
check, gate_cfg, llm_cfg = cfg["triage_check"], cfg["gate"], cfg["llm"]
prompt_id = sys.argv[1] if len(sys.argv) > 1 else llm_cfg["triage_prompt"]
vocabulary = Vocabulary.from_config(cfg)
grid = WindowGrid.from_config(cfg)
boroughs = vocabulary.codes
adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
neighbors = adjacency_matrix(boroughs, adjacency)
first, last = cfg["study"]["splits"]["validation"]
windows = grid.split_windows(first, last)
end = int(windows[-1]) + 1
events = pd.read_parquet(cfg["data"]["events_path"])
events = events[events["admitted"] < grid.first_day(end)]
budget = StreamBudget.from_config(cfg, check["epsilon"], check["unit"])
sigma = budget.borough_sigma
settings = GateScoreSettings(gate_cfg["baseline_windows"], gate_cfg["guard_windows"], gate_cfg["slack"], gate_cfg["memory_windows"], gate_cfg["spatial_weight"])
evidence_settings = {"confidence_z": cfg["agent"]["confidence_z"], "baseline": settings.baseline, "guard": settings.guard}
rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 8]))
provider = make_provider_call(llm_cfg["model"], llm_cfg["temperature"], llm_cfg["max_retries"], llm_cfg["retry_wait_seconds"])
llm = CachedLLM(provider, llm_cfg["model"], Path(llm_cfg["cache_dir"]), check["max_llm_calls"])

truth = count_cells(events, grid, boroughs, list(cfg["institutions"].values())).sum(axis=2).astype(float)[:end]
released_real = truth + rng.normal(0, sigma, truth.shape)
level, dispersion = estimate_null(released_real[windows], sigma)
tau = null_threshold(released_real, windows, sigma, settings, neighbors, gate_cfg["null_alerts_per_30_days"], grid.window_days, gate_cfg["null_seeds"], rng)
window_index = grid.index_of(events["admitted"])
city = {name: np.bincount(window_index[events[flag].to_numpy()], minlength=end).astype(float)[:end]
        for name, flag in (("FLU", "flu_confirmed"), ("COVID", "covid_confirmed"))}


def released_from(values: np.ndarray) -> ReleasedTable:
    released = ReleasedTable(boroughs, grid.size, sigma, budget.citywide_sigma)
    for t in range(end):
        released.add_window(values[t], {n: city[n][t] + rng.normal(0, budget.citywide_sigma) for n in city})
    return released


def alerts(values: np.ndarray):
    scores = gate_scores(values, sigma, settings, neighbors)
    gate = EscalationGate(boroughs, adjacency, tau, tau * gate_cfg["tau_lo_ratio"], gate_cfg["cooldown_windows"], gate_cfg["max_events_per_window"])
    for t in windows:
        for event in gate.step(int(t), scores[t]).escalate:
            yield int(t), event


rows, false_alerts, true_alerts = [], 0, 0
attempts = 0
while (false_alerts < check["alerts_per_group"] or true_alerts < check["alerts_per_group"]) and attempts < check["max_runs"]:
    attempts += 1
    background = simulate_null(rng, level, dispersion, end)
    if false_alerts < check["alerts_per_group"]:
        values = background + rng.normal(0, sigma, background.shape)
        released = released_from(values)
        z = z_scores(values, sigma, settings)
        for t, event in alerts(values):
            if false_alerts >= check["alerts_per_group"]:
                break
            false_alerts += 1
            evidence = build_evidence(event, t, released, z, grid, vocabulary, evidence_settings, adjacency)
            result = triage_event(llm, event, t, evidence, prompt_id)
            rows.append({"group": "known_false", "verdict": result.verdict, "grounded": result.grounded, "first_try": result.grounded_first_try})
            print("known_false", len(rows), result.verdict, result.grounded, flush=True)
    if true_alerts < check["alerts_per_group"]:
        begin = int(rng.choice(windows[:-check["outbreak_windows"]]))
        place = int(rng.integers(len(boroughs)))
        injected = background.copy()
        weights = rng.dirichlet(np.ones(check["outbreak_windows"]) * 3)
        injected[begin:begin + check["outbreak_windows"], place] += np.round(weights * check["excess_cases"])
        values = injected + rng.normal(0, sigma, injected.shape)
        released = released_from(values)
        z = z_scores(values, sigma, settings)
        for t, event in alerts(values):
            matched = begin <= t < begin + check["outbreak_windows"] and boroughs[place] in event.boroughs
            if not matched or true_alerts >= check["alerts_per_group"]:
                continue
            true_alerts += 1
            evidence = build_evidence(event, t, released, z, grid, vocabulary, evidence_settings, adjacency)
            result = triage_event(llm, event, t, evidence, prompt_id)
            rows.append({"group": "known_true", "verdict": result.verdict, "grounded": result.grounded, "first_try": result.grounded_first_try})
            print("known_true", len(rows), result.verdict, result.grounded, flush=True)

table = pd.DataFrame(rows)
known_true, known_false = table[table["group"] == "known_true"], table[table["group"] == "known_false"]
report = {
    "note": "Simulated background, validation-level check. Not a paper result.",
    "prompt": prompt_id, "model": llm_cfg["model"], "epsilon": check["epsilon"], "unit": check["unit"], "threshold": round(tau, 2), "dispersion": round(dispersion, 2),
    "verdicts_by_group": pd.crosstab(table["group"], table["verdict"]).to_dict(),
    "sensitivity_confirm": round(float((known_true["verdict"] == "confirm").mean()), 3),
    "sensitivity_not_dismissed": round(float((known_true["verdict"] != "dismiss").mean()), 3),
    "benign_rejection_dismiss": round(float((known_false["verdict"] == "dismiss").mean()), 3),
    "benign_not_confirmed": round(float((known_false["verdict"] != "confirm").mean()), 3),
    "confirm_everything_baseline": {"sensitivity": 1.0, "benign_not_confirmed": 0.0},
    "grounded": round(float(table["grounded"].mean()), 3), "grounded_first_try": round(float(table["first_try"].mean()), 3),
    "live_calls": llm.calls, "cache_hits": llm.cache_hits, "runs": attempts,
}
print(json.dumps(report, indent=1, default=str))
Path(f"experiments/results/triage_check_{prompt_id}.json").write_text(json.dumps({"report": report, "rows": rows}, indent=1, default=str), encoding="utf8")
