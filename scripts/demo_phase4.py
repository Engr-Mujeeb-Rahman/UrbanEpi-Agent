"""End-to-end demo on validation dates only. Usage: demo_phase4.py [--real]

By default, a scripted stand-in returns prepared plans, so that run isn't evidence of parsing quality.
With --real, Gemini parses each question. A few examples still aren't a measure of quality.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from urbanepi.agent.agent import QueryAgent, format_answer
from urbanepi.agent.qir import QueryPlan
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.system import build_system, run_monitoring
from urbanepi.city.windows import WindowGrid
from urbanepi.config import load_config
from urbanepi.llm.adapter import CachedLLM, ScriptedLLM
from urbanepi.llm.gemini import make_provider_call

cfg = load_config()
EPSILON = 2.0
as_of = pd.Timestamp(cfg["study"]["splits"]["validation"][1])
events = pd.read_parquet(cfg["data"]["events_path"])
events = events[events["admitted"] <= as_of]
grid = WindowGrid.from_config(cfg)
vocabulary = Vocabulary.from_config(cfg)
settings = {**cfg["agent"], "assumed_records": cfg["privacy"]["assumed_records_per_person"], "delta": cfg["privacy"]["delta"]}
july = {"start": "2024-07-02", "end": "2024-07-31", "bucket": "none"}
plans = {
    "How many suspected respiratory cases were in Gustavo A. Madero in July 2024?":
        QueryPlan(intent="count", syndromes=["RESP"], districts=["005"], time=july),
    "How many confirmed COVID cases were there citywide in July 2024?":
        QueryPlan(intent="count", syndromes=["COVID"], time=july),
    "Did respiratory cases in Iztapalapa rise in July 2024 compared with the period before?":
        QueryPlan(intent="compare", syndromes=["RESP"], districts=["007"], time=july, comparison={"type": "previous_period"}),
    "How many COVID cases were there in Iztapalapa and Tlalpan, by borough, among people 65 and over in July 2024?":
        QueryPlan(intent="count", syndromes=["COVID"], districts=["007", "012"], group_by=["district"], age_bands=["65+"], time=july),
    "How many RSV cases were there citywide in July 2024?":
        QueryPlan(intent="count", syndromes=["RSV"], time=july),
    "List the names of flu patients in Tlalpan.":
        QueryPlan(intent="out_of_scope", refusal_reason="This system doesn't identify individuals."),
    "How many cases were there?":
        QueryPlan(intent="count", needs_clarification="Which illness, place, and period do you mean?"),
}
system = build_system(cfg, events, EPSILON, "record", 40)
last_window = int(grid.index_of(pd.Series([as_of]))[0]) - 1
run_monitoring(system, cfg, events, last_window, 40)
adhoc = system.budget.rho_total * cfg["privacy"]["budget_split"]["adhoc"]
if "--real" in sys.argv:
    llm_cfg = cfg["llm"]
    provider = make_provider_call(llm_cfg["model"], llm_cfg["temperature"], llm_cfg["max_retries"], llm_cfg["retry_wait_seconds"])
    llm = CachedLLM(provider, llm_cfg["model"], Path(llm_cfg["cache_dir"]), 20)
else:
    llm = ScriptedLLM({q: p.model_dump_json() for q, p in plans.items()})
agent = QueryAgent(llm, vocabulary, grid, system.released, system.federation, adhoc, settings)
print(f"epsilon={EPSILON}, borough sigma={system.budget.borough_sigma:.2f}, citywide sigma={system.budget.citywide_sigma:.2f}, "
      f"ad-hoc rho={adhoc:.5f}, windows released={system.released.released_windows}\n")
for question in plans:
    print("Q:", question)
    print(format_answer(agent.ask(question, as_of)), "\n")
print(f"Ledger spent {system.nodes[0].ledger.spent:.5f} of {system.nodes[0].ledger.total:.5f}; ad-hoc remaining {agent.adhoc_remaining:.5f}")

july_records = events[(events["admitted"] >= "2024-06-29" if "--real" in sys.argv else events["admitted"] >= "2024-07-02") & (events["admitted"] <= "2024-07-31")]
print("\nEvaluator check of true counts (snapped window range):")
print("RESP Gustavo A. Madero:", int((july_records["borough"] == "005").sum()), "| COVID citywide:", int(july_records["covid_confirmed"].sum()),
      "| RSV citywide:", int(july_records["pathogens"].map(lambda p: "RSV" in p).sum()))
