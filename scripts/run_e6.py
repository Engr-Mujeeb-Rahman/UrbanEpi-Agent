"""E6: triage value and faithfulness on test-period dates, with known labels from a simulated background.

Known false: alerts on outbreak-free background. Known true: alerts overlapping an injected outbreak.
Real-data alerts have no labels, so they're reported only as a verdict mix.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

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
setup, gate_cfg, llm_cfg = cfg["e6"], cfg["gate"], cfg["llm"]
vocabulary = Vocabulary.from_config(cfg)
grid = WindowGrid.from_config(cfg)
boroughs = vocabulary.codes
adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
neighbors = adjacency_matrix(boroughs, adjacency)
validation = grid.split_windows(*cfg["study"]["splits"]["validation"])
test = grid.split_windows(*cfg["study"]["splits"]["test"])
end = int(test[-1]) + 1
events = pd.read_parquet(cfg["data"]["events_path"])
budget = StreamBudget.from_config(cfg, setup["epsilon"], setup["unit"])
sigma = budget.borough_sigma
settings = GateScoreSettings(gate_cfg["baseline_windows"], gate_cfg["guard_windows"], gate_cfg["slack"], gate_cfg["memory_windows"], gate_cfg["spatial_weight"])
evidence_settings = {"confidence_z": cfg["agent"]["confidence_z"], "baseline": settings.baseline, "guard": settings.guard}
rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 80]))
provider = make_provider_call(llm_cfg["model"], llm_cfg["temperature"], llm_cfg["max_retries"], llm_cfg["retry_wait_seconds"])
llm = CachedLLM(provider, llm_cfg["model"], Path(llm_cfg["cache_dir"]), setup["max_llm_calls"])

truth = count_cells(events, grid, boroughs, list(cfg["institutions"].values())).sum(axis=2).astype(float)
validation_end = int(validation[-1]) + 1
released_validation = truth[:validation_end] + rng.normal(0, sigma, (validation_end, len(boroughs)))
level, dispersion = estimate_null(released_validation[validation], sigma)
tau = null_threshold(released_validation, validation, sigma, settings, neighbors, gate_cfg["null_alerts_per_30_days"], grid.window_days, gate_cfg["null_seeds"], rng)
window_index = grid.index_of(events["admitted"])
city = {name: np.bincount(window_index[events[flag].to_numpy()], minlength=grid.size).astype(float)
        for name, flag in (("FLU", "flu_confirmed"), ("COVID", "covid_confirmed"))}


def released_from(values: np.ndarray) -> ReleasedTable:
    released = ReleasedTable(boroughs, grid.size, sigma, budget.citywide_sigma)
    for t in range(end):
        released.add_window(values[t], {n: city[n][t] + rng.normal(0, budget.citywide_sigma) for n in city})
    return released


def alerts(values: np.ndarray):
    scores = gate_scores(values, sigma, settings, neighbors)
    gate = EscalationGate(boroughs, adjacency, tau, tau * gate_cfg["tau_lo_ratio"], gate_cfg["cooldown_windows"], gate_cfg["max_events_per_window"])
    for t in test:
        for event in gate.step(int(t), scores[t]).escalate:
            yield int(t), event


def simple_rule(evidence: dict) -> str:
    """The same evidence and criteria given to the LLM, written as a fixed rule."""
    best, support = None, False
    for cell in evidence["boroughs"]:
        margin = cell["excess_over_baseline"] - cell["two_sd_of_noise_and_baseline"]
        if best is None or margin > best:
            best = margin
        earlier = cell["previous_windows"][-1] if cell["previous_windows"] else cell["baseline_mean"]
        support = support or earlier > cell["baseline_mean"] or any(z >= 1 for z in cell["neighbor_z_scores"].values())
    if best < 0:
        return "dismiss"
    return "confirm" if support or len(evidence["boroughs"]) > 1 else "watch"


rows = []


def review(group: str, t: int, event, released: ReleasedTable, z: np.ndarray) -> None:
    evidence = build_evidence(event, t, released, z, grid, vocabulary, evidence_settings, adjacency)
    result = triage_event(llm, event, t, evidence, llm_cfg["triage_prompt"])
    rows.append({"group": group, "window_start": evidence["window_start"], "boroughs": len(event.boroughs), "llm": result.verdict, "rule": simple_rule(evidence),
                 "grounded": result.grounded, "first_try": result.grounded_first_try, "fallback": result.used_fallback, "briefing": result.briefing})
    print(group, len(rows), result.verdict, simple_rule(evidence), result.grounded, flush=True)


false_count = true_count = runs = 0
want = setup["known_per_group"]
while (false_count < want or true_count < want) and runs < setup["max_runs"]:
    runs += 1
    background = simulate_null(rng, level, dispersion, end)
    if false_count < want:
        values = background + rng.normal(0, sigma, background.shape)
        released, z = released_from(values), z_scores(values, sigma, settings)
        for t, event in alerts(values):
            if false_count < want:
                false_count += 1
                review("known_false", t, event, released, z)
    if true_count < want:
        begin = int(rng.choice(test[:-setup["outbreak_windows"]]))
        place = int(rng.integers(len(boroughs)))
        injected = background.copy()
        injected[begin:begin + setup["outbreak_windows"], place] += np.round(rng.dirichlet(np.ones(setup["outbreak_windows"]) * 3) * setup["excess_cases"])
        values = injected + rng.normal(0, sigma, injected.shape)
        released, z = released_from(values), z_scores(values, sigma, settings)
        for t, event in alerts(values):
            if begin <= t < begin + setup["outbreak_windows"] and boroughs[place] in event.boroughs and true_count < want:
                true_count += 1
                review("known_true", t, event, released, z)

values = truth[:end] + rng.normal(0, sigma, (end, len(boroughs)))
released, z = released_from(values), z_scores(values, sigma, settings)
real = 0
for t, event in alerts(values):
    if real < setup["max_real_alerts"]:
        real += 1
        review("real_unlabeled", t, event, released, z)

table = pd.DataFrame(rows)


def scores_for(column: str) -> dict:
    known_true, known_false = table[table["group"] == "known_true"], table[table["group"] == "known_false"]
    a, b = int((known_true[column] == "confirm").sum()), int((known_false[column] == "confirm").sum())
    return {"confirm_rate_true": round(a / len(known_true), 3), "confirm_rate_false": round(b / len(known_false), 3),
            "dismiss_rate_true": round(float((known_true[column] == "dismiss").mean()), 3), "dismiss_rate_false": round(float((known_false[column] == "dismiss").mean()), 3),
            "precision_of_confirm_at_equal_groups": round(a / (a + b), 3) if a + b else None,
            "fisher_p_confirm": round(float(fisher_exact([[a, len(known_true) - a], [b, len(known_false) - b]])[1]), 4)}


labeled = table[table["group"] != "real_unlabeled"]
report = {
    "note": "Known labels come from a simulated background. Real-data alerts have no labels.",
    "prompt": llm_cfg["triage_prompt"], "model": llm_cfg["model"], "epsilon": setup["epsilon"], "unit": setup["unit"], "threshold": round(tau, 2), "dispersion": round(dispersion, 2),
    "groups": table["group"].value_counts().to_dict(),
    "llm": scores_for("llm"), "simple_rule": scores_for("rule"),
    "confirm_everything": {"confirm_rate_true": 1.0, "confirm_rate_false": 1.0, "precision_of_confirm_at_equal_groups": 0.5},
    "llm_vs_rule_agreement": round(float((labeled["llm"] == labeled["rule"]).mean()), 3),
    "real_verdicts": table[table["group"] == "real_unlabeled"]["llm"].value_counts().to_dict(),
    "grounded_first_try": round(float(table["first_try"].mean()), 3), "grounded_final": round(float(table["grounded"].mean()), 3),
    "fallbacks": int(table["fallback"].sum()), "live_calls": llm.calls, "cache_hits": llm.cache_hits,
    "chars_in": sum(e["chars_in"] for e in llm.log), "chars_out": sum(e["chars_out"] for e in llm.log),
}
print(json.dumps(report, indent=1))
Path("experiments/results/final/e6.json").write_text(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf8")
