"""Measured overhead, added after review. One machine, simulated institutions, free-tier Gemini over a home connection.

Live parsing calls use a sample of test questions and a temporary cache, and their plans aren't scored.
Fresh-query timing uses validation dates and evaluator-written plans. Monitoring timing covers the validation period.
"""
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from google.genai import types

from urbanepi.agent.agent import QueryAgent
from urbanepi.agent.parser import ParseFailure, parse_question
from urbanepi.agent.qir import QueryPlan
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.system import build_system, run_monitoring
from urbanepi.city.windows import WindowGrid
from urbanepi.config import load_config
from urbanepi.llm.adapter import CachedLLM
from urbanepi.llm.gemini import client_from_env

cfg = load_config()
llm_cfg = cfg["llm"]
vocabulary = Vocabulary.from_config(cfg)
grid = WindowGrid.from_config(cfg)
as_of = pd.Timestamp(cfg["study"]["splits"]["validation"][1])
all_events = pd.read_parquet(cfg["data"]["events_path"])
events = all_events[all_events["admitted"] <= as_of]
settings = {**cfg["agent"], "assumed_records": cfg["privacy"]["assumed_records_per_person"], "delta": cfg["privacy"]["delta"]}


def summary(values: list[float], digits: int = 3) -> dict:
    return {"n": len(values), "median": round(float(np.median(values)), digits), "p90": round(float(np.percentile(values, 90)), digits), "max": round(float(np.max(values)), digits)}


started = time.perf_counter()
system = build_system(cfg, events, 2.0, "record", 41)
build_seconds = time.perf_counter() - started
last_window = int(grid.index_of(pd.Series([as_of]))[0]) - 1
started = time.perf_counter()
run_monitoring(system, cfg, events, last_window, 41)
monitoring_seconds = (time.perf_counter() - started) / (last_window + 1)

adhoc = system.budget.rho_total * cfg["privacy"]["budget_split"]["adhoc"]
agent = QueryAgent(None, vocabulary, grid, system.released, system.federation, adhoc, settings)
fresh_times = []
months = [("2024-05-01", "2024-05-31"), ("2024-06-01", "2024-06-30"), ("2024-07-01", "2024-07-31"), ("2024-08-01", "2024-08-31")]
for syndrome in ("RSV", "ADENOVIRUS", "PARAINFLUENZA"):
    for start, end in months:
        plan = QueryPlan(intent="count", syndromes=[syndrome], time={"start": start, "end": end, "bucket": "none"})
        started = time.perf_counter()
        answer = agent.answer_plan(plan, as_of)
        elapsed = time.perf_counter() - started
        assert answer.status == "answer" and answer.fresh_cells, answer.message
        fresh_times.append(elapsed * 1000)

client = client_from_env()
config = types.GenerateContentConfig(temperature=llm_cfg["temperature"], response_mime_type="application/json")
calls = []


def measured_call(prompt: str) -> str:
    started = time.perf_counter()
    reply = client.models.generate_content(model=llm_cfg["model"], contents=prompt, config=config)
    usage = reply.usage_metadata
    calls.append({"seconds": time.perf_counter() - started, "prompt_tokens": usage.prompt_token_count, "output_tokens": usage.candidates_token_count, "total_tokens": usage.total_token_count})
    time.sleep(5)  # The free tier allows 15 requests a minute. Not counted in latency.
    return reply.text or ""


items = [json.loads(line) for line in Path("data/benchmark/urbanepi_q.jsonl").read_text(encoding="utf8").splitlines()]
items = [i for i in items if i["split"] == "test"][::5]
temporary = Path(tempfile.mkdtemp(prefix="urbanepi_metrics_"))
parse_times = []
try:
    llm = CachedLLM(measured_call, llm_cfg["model"], temporary, 40)
    for item in items:
        started = time.perf_counter()
        try:
            parse_question(llm, item["question"], vocabulary, pd.Timestamp(item["as_of"]))
        except ParseFailure:
            pass
        parse_times.append(time.perf_counter() - started)
        print(item["id"], round(parse_times[-1], 2), flush=True)
finally:
    shutil.rmtree(temporary, ignore_errors=True)

report = {
    "note": __doc__,
    "model": llm_cfg["model"],
    "parse_seconds_per_question": summary(parse_times, 2),
    "llm_call_seconds": summary([c["seconds"] for c in calls], 2),
    "prompt_tokens": summary([c["prompt_tokens"] for c in calls], 0),
    "output_tokens": summary([c["output_tokens"] for c in calls], 0),
    "total_tokens": summary([c["total_tokens"] for c in calls], 0),
    "fresh_query_execution_ms": summary(fresh_times, 1),
    "monitoring_seconds_per_window": round(monitoring_seconds, 4),
    "graph_build_seconds_validation_events": round(build_seconds, 1),
    "live_calls": len(calls),
}
print(json.dumps(report, indent=1))
Path("experiments/results/final/system_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf8")
