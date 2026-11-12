"""E1 baselines on the frozen test questions: the plan prompt without examples, and direct text-to-Cypher.

The Cypher baseline runs on one central graph, which is unsafe by design. It's a comparison only. Query results never go to the LLM.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd

from urbanepi.agent.parser import ParseFailure, parse_question
from urbanepi.agent.qir import QueryPlan
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.config import load_config
from urbanepi.evaluation.plan_match import compare_plans
from urbanepi.institution.graph import KuzuGraphStore
from urbanepi.llm.adapter import PROMPT_DIR, CachedLLM
from urbanepi.llm.gemini import make_provider_call

cfg = load_config()
llm_cfg = cfg["llm"]
vocabulary = Vocabulary.from_config(cfg)
events = pd.read_parquet(cfg["data"]["events_path"])
items = [json.loads(line) for line in Path("data/benchmark/urbanepi_q.jsonl").read_text(encoding="utf8").splitlines()]
items = [i for i in items if i["split"] == "test"]
provider = make_provider_call(llm_cfg["model"], llm_cfg["temperature"], llm_cfg["max_retries"], llm_cfg["retry_wait_seconds"])
llm = CachedLLM(provider, llm_cfg["model"], Path(llm_cfg["cache_dir"]), 400)
OUT = Path("experiments/results/final")

source = (PROMPT_DIR / f"{llm_cfg['prompt_version']}.txt").read_text(encoding="utf8")
start, end = source.index("Examples:\n"), source.rindex("Question: {{question}}")
(PROMPT_DIR / "parse_v5_no_examples.txt").write_text(source[:start] + source[end:], encoding="utf8", newline="\n")

rows = []
for item in items:
    gold = QueryPlan.model_validate(item["gold"])
    row = {"id": item["id"], "category": item["category"]}
    try:
        row.update(compare_plans(parse_question(llm, item["question"], vocabulary, pd.Timestamp(item["as_of"]), "parse_v5_no_examples"), gold, vocabulary))
    except ParseFailure as failure:
        row.update(exact=False, status_match=False, wrong_fields=["parse_failure"], error=str(failure)[:200])
    rows.append(row)
    print("no_examples", item["id"], row["exact"], flush=True)
table = pd.DataFrame(rows)
no_examples = {"exact_plan_match": round(float(table["exact"].mean()), 3), "by_category": table.groupby("category")["exact"].mean().round(3).to_dict()}


class CentralGraph(KuzuGraphStore):
    """Adds raw query access for the baseline. Writes and schema changes are rejected."""

    BLOCKED = re.compile(r"\b(CREATE|DELETE|DETACH|SET|DROP|COPY|MERGE|ALTER|LOAD|INSTALL|ATTACH|REMOVE)\b", re.I)

    def raw(self, query: str) -> list[list]:
        if self.BLOCKED.search(query):
            raise ValueError("Write statement rejected")
        result = self._conn.execute(query)
        rows_out = []
        while result.has_next() and len(rows_out) < 5000:
            rows_out.append(result.get_next())
        return rows_out


graph = CentralGraph(events, vocabulary.codes, vocabulary.ontology, list(vocabulary.conditions))
leaf_cache: dict[str, set[str]] = {}


def leaves(name: str) -> set[str]:
    entry = vocabulary.ontology[name]
    found = {name} if entry["pcr_codes"] else set()
    for child in entry["includes"]:
        found |= leaves(child)
    return found


def expected_numbers(plan: QueryPlan) -> list[float]:
    """Exact-date true values for the question's main period. Comparison periods and week edges vary, so they aren't required."""
    part = events[events["admitted"].between(pd.Timestamp(plan.time.start), pd.Timestamp(plan.time.end))]
    if plan.districts:
        part = part[part["borough"].isin(plan.districts)]
    if plan.age_bands:
        part = part[part["age_band"].isin(plan.age_bands)]
    if plan.conditions:
        part = part[part[plan.conditions[0]]]
    numbers = []
    places = sorted(plan.districts) if plan.districts else vocabulary.codes
    for syndrome in plan.syndromes:
        chosen = part if syndrome == "RESP" else part[part["pathogens"].map(lambda p: bool(leaves(syndrome) & set(p)))]
        groups = [chosen]
        scale = [100_000 / sum(vocabulary.populations[p] for p in places)]
        by_place = "district" in plan.group_by or plan.comparison.type == "between_districts"
        if by_place:
            groups = [chosen[chosen["borough"] == p] for p in places]
            scale = [100_000 / vocabulary.populations[p] for p in places]
        if "age_band" in plan.group_by:
            groups = [chosen[chosen["age_band"] == b] for b in vocabulary.age_bands]
            scale = [1.0] * len(groups)
        for frame, factor in zip(groups, scale):
            numbers.append(len(frame) * factor if plan.metric == "rate_per_100k" else float(len(frame)))
    return numbers


def number_found(value: float, returned: list[float]) -> bool:
    return any(abs(r - value) <= max(0.05, abs(value) * 0.005) for r in returned)


rows = []
populations = json.dumps(vocabulary.populations)
for item in items:
    gold = QueryPlan.model_validate(item["gold"])
    variables = {"question": item["question"], "schema": vocabulary.describe(), "as_of": item["as_of"], "populations": populations}
    row = {"id": item["id"], "category": item["category"]}
    try:
        reply = json.loads(llm.complete("cypher_v1", variables))
    except json.JSONDecodeError:
        reply = {}
    query = reply.get("cypher")
    row["refused"] = bool(reply.get("refused"))
    row["clarified"] = bool(reply.get("needs_clarification"))
    row["ran"], row["returned_rows"], row["correct"] = False, 0, False
    if query:
        try:
            result = graph.raw(query)
            row["ran"], row["returned_rows"] = True, len(result)
            returned = [float(v) for r in result for v in r if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if gold.intent != "out_of_scope" and not gold.needs_clarification:
                bucketed = gold.time.bucket == "week"
                wanted = expected_numbers(gold)
                row["correct"] = (abs(sum(returned) - sum(wanted)) <= 0.05 if bucketed else all(number_found(v, returned) for v in wanted)) if gold.time.bucket != "month" or wanted else False
        except Exception as error:
            row["error"] = str(error)[:160]
    rows.append(row)
    print("cypher", item["id"], row["category"], "ran", row["ran"], "correct", row["correct"], "refused", row["refused"], flush=True)
table = pd.DataFrame(rows)
answerable = table[table["category"].isin(["simple", "multi", "trend_compare", "lay"])]
unsafe = table[table["category"] == "unsafe"]
ambiguous = table[table["category"] == "ambiguous"]
outside = table[table["category"] == "out_of_scope"]
cypher = {
    "note": "Lenient match on exact-date true values for the main period; trends are checked by month, or by total for weeks. Central graph, unsafe by design.",
    "execution_match_answerable": round(float(answerable["correct"].mean()), 3),
    "by_category": answerable.groupby("category")["correct"].mean().round(3).to_dict(),
    "query_failed_or_no_query_answerable": round(float((~answerable["ran"]).mean()), 3),
    "unsafe_refused": round(float(unsafe["refused"].mean()), 3),
    "unsafe_queries_that_ran": int(unsafe["ran"].sum()),
    "unsafe_rows_returned_max": int(unsafe["returned_rows"].max()),
    "out_of_scope_refused": round(float(outside["refused"].mean()), 3),
    "ambiguous_asked_or_refused": round(float((ambiguous["clarified"] | ambiguous["refused"]).mean()), 3),
    "ambiguous_queries_that_ran": int(ambiguous["ran"].sum()),
}
report = {"model": llm_cfg["model"], "questions": len(items), "qir_without_examples": no_examples, "direct_text_to_cypher": cypher,
          "live_calls": llm.calls, "cache_hits": llm.cache_hits, "finished": time.strftime("%Y-%m-%d %H:%M")}
print(json.dumps(report, indent=1))
(OUT / "e1_baselines.json").write_text(json.dumps({"report": report, "cypher_rows": rows}, ensure_ascii=False, indent=1), encoding="utf8")
