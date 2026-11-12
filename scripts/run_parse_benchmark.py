"""Parse UrbanEpi-Q questions with the configured LLM. Usage: run_parse_benchmark.py <dev|test>

The test split is for final runs only (Docs/rules.md section 5).
"""
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from urbanepi.agent.parser import ParseFailure, parse_question
from urbanepi.agent.qir import QueryPlan
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.config import load_config
from urbanepi.evaluation.plan_match import compare_plans
from urbanepi.llm.adapter import CachedLLM
from urbanepi.llm.gemini import make_provider_call

split = sys.argv[1]
cfg = load_config()
settings = cfg["llm"]
vocabulary = Vocabulary.from_config(cfg)
items = [json.loads(line) for line in Path("data/benchmark/urbanepi_q.jsonl").read_text(encoding="utf8").splitlines()]
items = [item for item in items if item["split"] == split]
provider = make_provider_call(settings["model"], settings["temperature"], settings["max_retries"], settings["retry_wait_seconds"])


def paced(prompt: str) -> str:
    time.sleep(settings["seconds_between_calls"])
    return provider(prompt)


llm = CachedLLM(paced, settings["model"], Path(settings["cache_dir"]), settings["max_calls_per_run"])
rows = []
for number, item in enumerate(items, start=1):
    gold = QueryPlan.model_validate(item["gold"])
    row = {"id": item["id"], "category": item["category"], "question": item["question"]}
    try:
        predicted = parse_question(llm, item["question"], vocabulary, pd.Timestamp(item["as_of"]), settings["prompt_version"])
        row["predicted"] = predicted.model_dump(mode="json")
        row.update(compare_plans(predicted, gold, vocabulary))
    except ParseFailure as failure:
        row.update(exact=False, status_match=False, wrong_fields=["parse_failure"], error=str(failure)[:300])
    rows.append(row)
    print(f"{number}/{len(items)} {item['id']} {item['category']:14s} exact={row['exact']} {row['wrong_fields']}", flush=True)

table = pd.DataFrame(rows)
unsafe = table[table["category"] == "unsafe"]
ambiguous = table[table["category"] == "ambiguous"]
summary = {
    "split": split, "model": settings["model"], "questions": len(table),
    "benchmark_review": "author read 60 rows; an AI tool checked all 200",
    "exact_plan_match": round(float(table["exact"].mean()), 3),
    "by_category": table.groupby("category")["exact"].mean().round(3).to_dict(),
    "refusal_recall": round(float(unsafe["exact"].mean()), 3) if len(unsafe) else None,
    "clarification_recall": round(float(ambiguous["exact"].mean()), 3) if len(ambiguous) else None,
    "wrong_fields": Counter(f for fields in table["wrong_fields"] for f in fields),
    "live_calls": llm.calls, "cache_hits": llm.cache_hits,
    "chars_in": sum(e["chars_in"] for e in llm.log), "chars_out": sum(e["chars_out"] for e in llm.log),
    "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}
print(json.dumps(summary, indent=1))
stamp = settings["prompt_version"] + "_" + settings["model"]
out = Path(f"experiments/results/e1_{split}_{stamp}.json")
out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf8")
