"""Go/No-Go 2: one complete pass on validation data, from injection to briefing, with all logs written.

Not a paper result. The first run goes through the institution path, and the others use the fast path.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.aggregator.system import build_system, run_monitoring
from urbanepi.city.injector import make_outbreak_records, sample_spec, write_ground_truth
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config
from urbanepi.gate.events import EscalationGate
from urbanepi.gate.naeg import GateScoreSettings, adjacency_matrix, gate_scores, z_scores
from urbanepi.gate.thresholds import null_threshold
from urbanepi.llm.adapter import CachedLLM, CallLimitReached
from urbanepi.llm.gemini import make_provider_call
from urbanepi.llm.triage import build_evidence, triage_event
from urbanepi.privacy.accounting import StreamBudget

cfg = load_config()
check, gate_cfg, llm_cfg = cfg["go_no_go_2"], cfg["gate"], cfg["llm"]
EPSILON, UNIT = check["epsilon"], check["unit"]
vocabulary = Vocabulary.from_config(cfg)
grid = WindowGrid.from_config(cfg)
boroughs = vocabulary.codes
institutions = list(cfg["institutions"].values())
adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
neighbors = adjacency_matrix(boroughs, adjacency)
first, last = cfg["study"]["splits"]["validation"]
windows = grid.split_windows(first, last)
end = int(windows[-1]) + 1
period_end = grid.first_day(end) - pd.Timedelta(days=1)
events = pd.read_parquet(cfg["data"]["events_path"])
events = events[events["admitted"] <= period_end]
source = events[events["admitted"] >= first]
budget = StreamBudget.from_config(cfg, EPSILON, UNIT)
settings = GateScoreSettings(gate_cfg["baseline_windows"], gate_cfg["guard_windows"], gate_cfg["slack"], gate_cfg["memory_windows"], gate_cfg["spatial_weight"])
evidence_settings = {"confidence_z": cfg["agent"]["confidence_z"], "baseline": settings.baseline, "guard": settings.guard}
seeds = np.random.SeedSequence([cfg["seed"], 6]).spawn(100)
rng = np.random.default_rng(seeds[0])
provider = make_provider_call(llm_cfg["model"], llm_cfg["temperature"], llm_cfg["max_retries"], llm_cfg["retry_wait_seconds"])
llm = CachedLLM(provider, llm_cfg["model"], Path(llm_cfg["cache_dir"]), check["max_llm_calls"])


def fast_release(table_events: pd.DataFrame, generator: np.random.Generator) -> ReleasedTable:
    released = ReleasedTable(boroughs, grid.size, budget.borough_sigma, budget.citywide_sigma)
    true_cells = count_cells(table_events, grid, boroughs, institutions).sum(axis=2).astype(float)
    window = grid.index_of(table_events["admitted"])
    city = {name: np.bincount(window[table_events[flag].to_numpy()], minlength=grid.size).astype(float)
            for name, flag in (("FLU", "flu_confirmed"), ("COVID", "covid_confirmed"))}
    for t in range(end):
        values = true_cells[t] + generator.normal(0, budget.borough_sigma, len(boroughs))
        released.add_window(values, {n: city[n][t] + generator.normal(0, budget.citywide_sigma) for n in city})
    return released


def run_gate(released: ReleasedTable, spec, run_name: str, log: list[dict]) -> dict:
    """Run the gate over validation windows and triage whatever it escalates."""
    table = released.borough_table[:end]
    tau = null_threshold(table, windows, budget.borough_sigma, settings, neighbors, gate_cfg["null_alerts_per_30_days"], grid.window_days,
                         gate_cfg["null_seeds"], rng)
    scores = gate_scores(table, budget.borough_sigma, settings, neighbors)
    z = z_scores(table, budget.borough_sigma, settings)
    gate = EscalationGate(boroughs, adjacency, tau, tau * gate_cfg["tau_lo_ratio"], gate_cfg["cooldown_windows"], gate_cfg["max_events_per_window"])
    begin = stop = None
    if spec:
        begin = int(grid.index_of(pd.Series([pd.Timestamp(spec.start)]))[0])
        stop = int(grid.index_of(pd.Series([pd.Timestamp(spec.start) + pd.Timedelta(days=spec.duration_days - 1)]))[0])
    escalations = detected = untriaged = 0
    for t in windows:
        decision = gate.step(int(t), scores[t])
        for event in decision.escalate:
            escalations += 1
            matched = bool(spec and begin <= t <= stop and event.boroughs & set(spec.boroughs))
            detected += matched
            row = {"run": run_name, "window": int(t), "window_start": str(grid.first_day(int(t)).date()), "boroughs": sorted(event.boroughs),
                   "score": round(float(event.score), 2), "matches_injected_outbreak": matched}
            evidence = build_evidence(event, int(t), released, z, grid, vocabulary, evidence_settings, adjacency)
            started = time.time()
            try:
                result = triage_event(llm, event, int(t), evidence, llm_cfg["triage_prompt"])
                outcome = asdict(result)
                outcome.pop("evidence")
                row.update(outcome, seconds=round(time.time() - started, 1))
            except CallLimitReached:
                untriaged += 1
                row["verdict"] = "not_triaged_call_limit"
            log.append(row)
            print(f"{run_name} {row['window_start']} {row['boroughs']} matched={matched} verdict={row['verdict']} grounded={row.get('grounded')} "
                  f"fallback={row.get('used_fallback')} {row.get('seconds')}s", flush=True)
    summary = {"run": run_name, "threshold": round(tau, 2), "windows": len(windows), "escalations": escalations, "matched_escalations": detected,
               "detected": detected > 0 if spec else None, "untriaged": untriaged,
               "call_reduction_vs_one_per_window": round(1 - escalations / len(windows), 3),
               "watch_or_other_events": len(gate.all_events)}
    print(summary, flush=True)
    return summary


def outbreak(size: int, name: str):
    duration = int(rng.choice(cfg["injection"]["duration_days"]))
    spec = sample_spec(rng, cfg, adjacency, name, size, duration, first, last)
    records = make_outbreak_records(rng, spec, source, vocabulary.populations, cfg["injection"]["lognormal_sigma"])
    return spec, pd.concat([events, records], ignore_index=True)


log: list[dict] = []
summaries = []
specs = []
started = time.time()
spec, combined = outbreak(check["full_path_excess_cases"], "full-path")
specs.append(spec)
system = build_system(cfg, combined, EPSILON, UNIT, 60)
run_monitoring(system, cfg, combined, end - 1, 60)
summaries.append(run_gate(system.released, spec, "full-path", log))
for number, size in enumerate(check["fast_path_excess_cases"]):
    name = f"fast-{number}-A{size}"
    if size:
        spec, combined = outbreak(size, name)
        specs.append(spec)
    else:
        spec, combined = None, events
    summaries.append(run_gate(fast_release(combined, np.random.default_rng(seeds[10 + number])), spec, name, log))

write_ground_truth(specs, cfg["data"]["ground_truth_dir"], "go_no_go_2")
triaged = [r for r in log if "grounded" in r]
report = {
    "note": "Validation data. Pipeline check, not a paper result.",
    "epsilon": EPSILON, "unit": UNIT, "model": llm_cfg["model"], "prompt": llm_cfg["triage_prompt"],
    "runs": summaries,
    "triaged": len(triaged),
    "verdicts": pd.Series([r["verdict"] for r in triaged]).value_counts().to_dict() if triaged else {},
    "verdict_by_match": pd.crosstab([r["matches_injected_outbreak"] for r in triaged], [r["verdict"] for r in triaged]).to_dict() if triaged else {},
    "grounded_first_try": round(float(np.mean([r["grounded_first_try"] for r in triaged])), 3) if triaged else None,
    "grounded_final": round(float(np.mean([r["grounded"] for r in triaged])), 3) if triaged else None,
    "fallbacks": sum(r["used_fallback"] for r in triaged),
    "live_calls": llm.calls, "cache_hits": llm.cache_hits,
    "chars_in": sum(e["chars_in"] for e in llm.log), "chars_out": sum(e["chars_out"] for e in llm.log),
    "minutes": round((time.time() - started) / 60, 1),
    "log": log,
}
Path("experiments/results/go_no_go_2.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf8")
print(json.dumps({k: v for k, v in report.items() if k not in ("log",)}, indent=1, default=str))
