"""Tests for query plans, graph templates, PB-QD, and the end-to-end agent."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from urbanepi.agent.agent import Answer, AnswerLine, QueryAgent, format_answer
from urbanepi.agent.pb_qd import allocate_noise, decompose, expand_plan
from urbanepi.agent.qir import QueryPlan, normalized, policy_refusal, validation_errors
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.system import build_system, run_monitoring
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config
from urbanepi.evaluation.plan_match import compare_plans
from urbanepi.institution.graph import KuzuGraphStore, TemplateRejected
from urbanepi.llm.adapter import CachedLLM, CallLimitReached, ScriptedLLM

CFG = load_config()
VOCABULARY = Vocabulary.from_config(CFG)
GRID = WindowGrid.from_config(CFG)
AS_OF = pd.Timestamp("2024-11-01")
INSTITUTIONS = list(CFG["institutions"].values())
SETTINGS = {**CFG["agent"], "assumed_records": CFG["privacy"]["assumed_records_per_person"], "delta": CFG["privacy"]["delta"]}
REAL_EVENTS = Path(CFG["data"]["events_path"])
CANARY = "CANARY-RECORD-777"
LEAVES = [name for name, entry in VOCABULARY.ontology.items() if entry["pcr_codes"]]
FLU_LEAVES = {"FLU_A_H1N1PDM", "FLU_A_H1", "FLU_A_H3", "FLU_A_H5", "FLU_A_UNSUBTYPED", "FLU_B"}


def plan(**changes) -> QueryPlan:
    base = {"intent": "count", "syndromes": ["RESP"], "districts": ["007"],
            "time": {"start": "2024-07-01", "end": "2024-07-30", "bucket": "none"}}
    base.update(changes)
    return QueryPlan.model_validate(base)


def make_events() -> pd.DataFrame:
    rows, rng = [], np.random.default_rng(5)
    for i in range(1500):
        rows.append({
            "record_id": CANARY if i == 0 else f"r{i}",
            "admitted": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 300))),
            "borough": str(rng.choice(VOCABULARY.codes)),
            "institution": str(rng.choice(INSTITUTIONS)),
            "age_band": str(rng.choice(VOCABULARY.age_bands)),
            "pathogens": sorted(str(p) for p in rng.choice(LEAVES, size=int(rng.integers(0, 3)), replace=False)),
            **{name: bool(rng.random() < 0.2) for name in CFG["conditions"]},
        })
    table = pd.DataFrame(rows)
    table["flu_confirmed"] = table["pathogens"].map(lambda p: bool(set(p) & FLU_LEAVES))
    table["covid_confirmed"] = table["pathogens"].map(lambda p: "COVID" in p)
    return table


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return make_events()


@pytest.fixture()
def agent_and_system(events):
    def build(plans=None, epsilon=2.0):
        system = build_system(CFG, events, epsilon, "record", 1)
        through = int(GRID.index_of(pd.Series([AS_OF]))[0]) - 1
        run_monitoring(system, CFG, events, through, 1)
        budget = system.budget.rho_total * CFG["privacy"]["budget_split"]["adhoc"]
        llm = ScriptedLLM({k: v.model_dump_json() for k, v in (plans or {}).items()})
        return QueryAgent(llm, VOCABULARY, GRID, system.released, system.federation, budget, SETTINGS), system, llm
    return build


def test_plan_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        plan(cypher="MATCH (n) RETURN n")


@pytest.mark.parametrize("changes, message", [
    ({"syndromes": ["DENGUE"]}, "unknown syndrome"), ({"districts": ["999"]}, "unknown district"),
    ({"age_bands": ["90+"]}, "unknown age band"), ({"conditions": ["cancer"]}, "unknown condition"),
    ({"time": {"start": "2020-01-01", "end": "2020-02-01"}}, "outside the study"),
    ({"time": {"start": "2024-08-01", "end": "2024-07-01"}}, "after time.end"),
    ({"time": {"start": "2025-01-01", "end": "2025-01-31"}}, "after the data available"),
    ({"intent": "compare"}, "needs a comparison"), ({"intent": "trend"}, "needs a time bucket"),
    ({"conditions": ["asthma", "copd"]}, "at most one"),
    ({"metric": "rate_per_100k", "age_bands": ["65+"]}, "whole borough populations"),
    ({"metric": "rate_per_100k", "conditions": ["asthma"]}, "whole borough populations"),
    ({"metric": "rate_per_100k", "group_by": ["age_band"]}, "whole borough populations"),
    ({"intent": "compare", "comparison": {"type": "previous_period"}, "time": {"start": "2024-01-01", "end": "2024-01-31"}}, "before the data"),
    ({"intent": "compare", "comparison": {"type": "between_districts"}}, "exactly two"),
])
def test_validator_finds_errors(changes, message):
    assert any(message in e for e in validation_errors(plan(**changes), VOCABULARY, AS_OF))


def test_valid_plan_has_no_errors():
    assert validation_errors(plan(), VOCABULARY, AS_OF) == []


def test_policy_refuses_out_of_scope_and_narrow_groups():
    assert policy_refusal(QueryPlan(intent="out_of_scope", refusal_reason="no"), 14) == "no"
    narrow = plan(conditions=["asthma"], age_bands=["65+"], time={"start": "2024-07-01", "end": "2024-07-05"})
    assert policy_refusal(narrow, 14) is not None
    assert policy_refusal(plan(), 14) is None


def test_gold_matching_ignores_order_and_all_district_spelling():
    assert normalized(plan(districts=VOCABULARY.codes), VOCABULARY)["districts"] is None
    result = compare_plans(plan(syndromes=["FLU", "COVID"]), plan(syndromes=["COVID", "FLU"]), VOCABULARY)
    assert result["exact"]
    assert compare_plans(plan(metric="rate_per_100k"), plan(), VOCABULARY)["wrong_fields"] == ["metric"]


@pytest.fixture(scope="module")
def store(events):
    return KuzuGraphStore(events[events["institution"] == "IMSS"], VOCABULARY.codes, VOCABULARY.ontology, list(VOCABULARY.conditions))


def graph_params(**changes):
    return {"districts": VOCABULARY.codes, "from_date": pd.Timestamp("2024-01-01").date(),
            "to_date": pd.Timestamp("2024-12-31").date(), "age_bands": list(VOCABULARY.age_bands), **changes}


def test_unregistered_template_and_wrong_parameters_are_rejected(store):
    with pytest.raises(TemplateRejected):
        store.run_template("MATCH (n) RETURN n", {})
    with pytest.raises(TemplateRejected):
        store.run_template("all_cases", {"districts": []})


def test_graph_counts_match_the_table(store, events):
    mine = events[events["institution"] == "IMSS"]
    assert store.run_template("all_cases", graph_params()) == len(mine)
    assert store.run_template("by_syndrome", graph_params(syndrome="FLU")) == mine["flu_confirmed"].sum()
    assert store.run_template("by_syndrome_condition", graph_params(syndrome="COVID", condition="asthma")) == (mine["covid_confirmed"] & mine["asthma"]).sum()
    assert store.run_template("all_cases_condition", graph_params(condition="copd", age_bands=["65+"])) == (mine["copd"] & (mine["age_band"] == "65+")).sum()


def test_parent_syndrome_expands_through_the_ontology(store, events):
    mine = events[events["institution"] == "IMSS"]
    has = lambda names: mine["pathogens"].map(lambda p: bool(set(p) & names)).sum()
    assert store.run_template("by_syndrome", graph_params(syndrome="FLU")) == has(FLU_LEAVES)
    assert store.run_template("by_syndrome", graph_params(syndrome="FLU_A")) == has(FLU_LEAVES - {"FLU_B"})
    assert store.run_template("by_syndrome", graph_params(syndrome="FLU_A_H3")) == has({"FLU_A_H3"})
    assert store.run_template("by_syndrome", graph_params(syndrome="RSV")) == has({"RSV"})
    assert store.run_template("by_syndrome", graph_params(syndrome="RESP")) == has(set(LEAVES))


def test_ontology_is_a_consistent_tree():
    ontology = VOCABULARY.ontology
    children = [c for entry in ontology.values() for c in entry["includes"]]
    assert len(children) == len(set(children)) and set(children) <= set(ontology)
    assert set(ontology) - set(children) == {"RESP"}
    for name, entry in ontology.items():
        assert bool(entry["includes"]) != bool(entry["pcr_codes"]), name
    codes = [c for entry in ontology.values() for c in entry["pcr_codes"]]
    assert len(codes) == len(set(codes))
    known = set(pd.read_csv("data/public/pcr_codes.csv", dtype=str)["code"])
    assert set(codes) <= known


@pytest.mark.skipif(not REAL_EVENTS.exists(), reason="events table not built")
def test_pcr_pathogens_agree_with_the_official_flags_on_real_data():
    real = pd.read_parquet(REAL_EVENTS)
    flu = real["pathogens"].map(lambda p: bool(set(p) & FLU_LEAVES))
    covid = real["pathogens"].map(lambda p: "COVID" in p)
    assert (flu == real["flu_confirmed"]).all() and (covid == real["covid_confirmed"]).all()


@pytest.mark.skipif(not REAL_EVENTS.exists(), reason="events table not built")
@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_graph_path_equals_fast_path_on_real_data(institution):
    real = pd.read_parquet(REAL_EVENTS)
    mine = real[real["institution"] == institution]
    graph = KuzuGraphStore(mine, VOCABULARY.codes, VOCABULARY.ontology, list(VOCABULARY.conditions))
    cells = count_cells(real, GRID, VOCABULARY.codes, INSTITUTIONS)[:, :, INSTITUTIONS.index(institution)]
    for window, code in [(0, "007"), (30, "005"), (60, "012"), (100, "015"), (156, "003")]:
        params = {"districts": [code], "from_date": GRID.first_day(window).date(), "to_date": GRID.first_day(window + 1).date(),
                  "age_bands": list(VOCABULARY.age_bands)}
        assert graph.run_template("all_cases", params) == cells[window, VOCABULARY.codes.index(code)]


def test_allocation_meets_budget_and_beats_alternatives():
    deltas, weights, rho = [1.0, 2.0, 1.0], [9.0, 1.0, 0.25], 0.05
    sigmas = np.array(allocate_noise(deltas, weights, rho))
    assert sum(d ** 2 / (2 * s ** 2) for d, s in zip(deltas, sigmas)) == pytest.approx(rho)
    best = (np.array(weights) * sigmas ** 2).sum()
    rng = np.random.default_rng(2)
    for _ in range(500):
        other = sigmas * rng.uniform(0.5, 2, 3)
        cost = sum(d ** 2 / (2 * s ** 2) for d, s in zip(deltas, other))
        assert (np.array(weights) * (other * (cost / rho) ** 0.5) ** 2).sum() >= best - 1e-9


def test_allocation_matches_uniform_for_identical_queries():
    sigmas = allocate_noise([1.0] * 4, [1.0] * 4, 0.1)
    assert sigmas == pytest.approx([(4 / 0.2) ** 0.5] * 4)


def test_expansion_covers_periods_and_previous_period():
    atoms, outputs = expand_plan(plan(intent="compare", comparison={"type": "previous_period"}), VOCABULARY, GRID)
    assert len(atoms) == 2
    assert atoms[1].last_window == atoms[0].first_window - 1
    assert atoms[0].last_window - atoms[0].first_window == atoms[1].last_window - atoms[1].first_window
    assert any("change" in o.label for o in outputs)


def test_single_month_bucket_gives_one_total_and_longer_ranges_split():
    one = plan(time={"start": "2024-07-01", "end": "2024-07-31", "bucket": "month"})
    assert len(expand_plan(one, VOCABULARY, GRID)[0]) == 1
    many = plan(intent="trend", time={"start": "2024-01-01", "end": "2024-06-30", "bucket": "month"})
    atoms = expand_plan(many, VOCABULARY, GRID)[0]
    assert len(atoms) == 6
    assert all(b.first_window == a.last_window + 1 for a, b in zip(atoms, atoms[1:]))


def test_equivalent_plans_compare_as_equal():
    gold = plan(time={"start": "2024-07-01", "end": "2024-07-31", "bucket": "none"})
    month = plan(time={"start": "2024-07-01", "end": "2024-07-31", "bucket": "month"})
    assert compare_plans(month, gold, VOCABULARY)["exact"]
    assert compare_plans(plan(intent="rate", metric="rate_per_100k"), plan(metric="rate_per_100k"), VOCABULARY)["exact"]
    pair = {"intent": "compare", "districts": ["007", "005"], "comparison": {"type": "between_districts"}}
    assert compare_plans(plan(**pair, group_by=["district"]), plan(**pair), VOCABULARY)["exact"]
    trend = {"intent": "trend", "time": {"start": "2024-01-01", "end": "2024-06-30", "bucket": "month"}}
    assert not compare_plans(plan(**trend), plan(time={"start": "2024-01-01", "end": "2024-06-30"}), VOCABULARY)["exact"]


def test_two_syndromes_stay_separate_numbers():
    both = plan(syndromes=["COVID", "RESP"], districts=None)
    atoms, outputs = expand_plan(both, VOCABULARY, GRID)
    assert len(atoms) == 2 and len(outputs) == 2 and all(len(o.terms) == 1 for o in outputs)


def test_every_borough_gets_a_row_when_grouped():
    _, outputs = expand_plan(plan(districts=None, group_by=["district"]), VOCABULARY, GRID)
    assert len(outputs) == 16


def test_answer_text_names_what_was_counted_and_uses_plain_labels():
    _, outputs = expand_plan(plan(syndromes=["RHINO_ENTERO"]), VOCABULARY, GRID)
    assert "rhinovirus/enterovirus" in outputs[0].label
    answer = Answer("answer", lines=[AnswerLine("x", 1.0, 0.0, 2.0)])
    assert "reported" in format_answer(answer) and "sentinel" in format_answer(answer)


def test_refusal_offers_aggregates_and_never_confirms_a_person():
    text = format_answer(Answer("refuse", "No individuals."))
    assert "aggregate" in text and "whether any person" in text


def test_answer_warns_when_an_interval_includes_zero(agent_and_system):
    agent, _, _ = agent_and_system()
    agent.released.citywide_sigma = 1_000.0
    answer = agent.answer_plan(plan(syndromes=["COVID"], districts=None), AS_OF)
    assert any("includes zero" in note for note in answer.notes)


def test_rates_use_public_population():
    _, outputs = expand_plan(plan(intent="rate", metric="rate_per_100k"), VOCABULARY, GRID)
    assert outputs[0].terms[0][1] == pytest.approx(100_000 / VOCABULARY.populations["007"])


def test_reusing_released_cells_costs_nothing(agent_and_system):
    agent, system, _ = agent_and_system()
    spent = [n.ledger.spent for n in system.nodes]
    answer = agent.answer_plan(plan(), AS_OF)
    assert answer.status == "answer" and answer.rho_spent == 0 and answer.fresh_cells == 0 and answer.reused_cells == 1
    assert [n.ledger.spent for n in system.nodes] == spent


def test_citywide_syndrome_is_free_but_borough_syndrome_is_not(agent_and_system):
    agent, _, _ = agent_and_system()
    assert agent.answer_plan(plan(syndromes=["FLU"], districts=None), AS_OF).rho_spent == 0
    assert agent.answer_plan(plan(syndromes=["FLU"]), AS_OF).rho_spent > 0


def test_fresh_query_charges_every_ledger_once_and_repeat_is_free(agent_and_system):
    agent, system, _ = agent_and_system()
    before = [n.ledger.spent for n in system.nodes]
    question = plan(age_bands=["65+"])
    first = agent.answer_plan(question, AS_OF)
    assert first.rho_spent > 0
    for node, earlier in zip(system.nodes, before):
        assert node.ledger.spent - earlier == pytest.approx(first.rho_spent)
    assert agent.answer_plan(question, AS_OF).rho_spent == 0


def test_ad_hoc_budget_runs_out_and_refuses_without_spending(agent_and_system):
    agent, system, _ = agent_and_system()
    agent.adhoc_remaining = 0.0
    before = [n.ledger.spent for n in system.nodes]
    answer = agent.answer_plan(plan(age_bands=["65+"], syndromes=["COVID"]), AS_OF)
    assert answer.status == "refuse"
    assert [n.ledger.spent for n in system.nodes] == before


def test_institution_refusal_blocks_the_release(agent_and_system):
    agent, system, _ = agent_and_system()
    system.nodes[0].ledger.spent = system.nodes[0].ledger.total
    before = [n.ledger.spent for n in system.nodes]
    assert agent.answer_plan(plan(age_bands=["65+"]), AS_OF).status == "refuse"
    assert [n.ledger.spent for n in system.nodes] == before


def test_total_spending_never_exceeds_budget(agent_and_system):
    agent, system, _ = agent_and_system()
    for code in VOCABULARY.codes * 3:
        agent.answer_plan(plan(age_bands=["65+"], districts=[code], syndromes=["FLU"]), AS_OF)
    assert all(n.ledger.spent <= n.ledger.total + 1e-9 for n in system.nodes)


def test_confidence_interval_width_reflects_the_noise_in_reused_cells(agent_and_system):
    agent, system, _ = agent_and_system()
    line = agent.answer_plan(plan(), AS_OF).lines[0]
    atoms, _ = expand_plan(plan(), VOCABULARY, GRID)
    windows = atoms[0].last_window - atoms[0].first_window + 1
    expected = SETTINGS["confidence_z"] * system.budget.borough_sigma * windows ** 0.5
    assert (line.ci_high - line.ci_low) / 2 == pytest.approx(expected)


def test_uniform_and_closed_form_cost_the_same(events, agent_and_system):
    _, system, _ = agent_and_system()
    multi = plan(syndromes=["FLU", "COVID"], age_bands=["65+"], group_by=["district"], districts=["007", "005"])
    a = decompose(multi, VOCABULARY, GRID, system.released, 0.01, SETTINGS, use_coarsening=False)
    b = decompose(multi, VOCABULARY, GRID, system.released, 0.01, SETTINGS, uniform=True, use_coarsening=False)
    assert a.rho == pytest.approx(0.01) and b.rho == pytest.approx(0.01)


def test_coarsening_is_reported(agent_and_system):
    _, system, _ = agent_and_system()
    narrow = plan(intent="trend", syndromes=["FLU"], age_bands=["65+"], time={"start": "2024-06-01", "end": "2024-10-31", "bucket": "window"})
    result = decompose(narrow, VOCABULARY, GRID, system.released, 1e-4, SETTINGS)
    assert any("Coarsened" in n for n in result.notes)


def test_no_private_values_reach_the_llm(agent_and_system):
    question = "How many cases?"
    agent, _, llm = agent_and_system({question: plan(age_bands=["65+"])})
    assert agent.ask(question, AS_OF).status == "answer"
    assert llm.prompts_seen
    joined = "\n".join(llm.prompts_seen)
    assert CANARY not in joined and "r1000" not in joined


def test_parser_refusal_and_clarification_cost_nothing(agent_and_system):
    plans = {"who": QueryPlan(intent="out_of_scope", refusal_reason="no individuals"),
             "what": QueryPlan(intent="count", needs_clarification="Which period?")}
    agent, system, _ = agent_and_system(plans)
    before = [n.ledger.spent for n in system.nodes]
    assert agent.ask("who", AS_OF).status == "refuse"
    assert agent.ask("what", AS_OF).status == "clarify"
    assert [n.ledger.spent for n in system.nodes] == before


def test_invalid_llm_output_is_repaired_once_then_fails():
    class Broken:
        calls = 0

        def complete(self, prompt_id, variables):
            self.calls += 1
            return "not json"
    from urbanepi.agent.parser import ParseFailure, parse_question
    llm = Broken()
    with pytest.raises(ParseFailure):
        parse_question(llm, "q", VOCABULARY, AS_OF)
    assert llm.calls == 2


def test_cache_avoids_repeat_calls_and_call_cap_stops(tmp_path):
    calls = []
    llm = CachedLLM(lambda p: (calls.append(p), "{}")[1], "m", tmp_path, 1)
    variables = {"question": "q", "schema": "s", "as_of": "2024-01-01"}
    assert llm.complete("parse_v1", variables) == llm.complete("parse_v1", variables) == "{}"
    assert len(calls) == 1 and llm.cache_hits == 1
    with pytest.raises(CallLimitReached):
        llm.complete("parse_v1", {**variables, "question": "other"})


def test_system_code_never_reads_ground_truth():
    for folder in ("agent", "aggregator", "gate", "llm", "institution", "privacy"):
        for path in Path(f"src/urbanepi/{folder}").rglob("*.py"):
            text = path.read_text(encoding="utf8")
            assert "ground_truth" not in text and "evaluator_only" not in text and "injector" not in text, path


def test_llm_code_is_only_reached_through_the_adapter():
    pattern = re.compile(r"^\s*(import|from)\s+(openai|google)", re.M)
    for path in Path("src/urbanepi").rglob("*.py"):
        if path.parent.name != "llm":
            assert not pattern.search(path.read_text(encoding="utf8")), path


BENCHMARK = Path("data/benchmark/urbanepi_q.jsonl")


@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark not built")
def test_benchmark_matches_the_design():
    items = [json.loads(line) for line in BENCHMARK.read_text(encoding="utf8").splitlines()]
    assert len(items) == 200 and len({i["question"] for i in items}) == 200
    table = pd.DataFrame(items)
    assert (table["split"] == "dev").sum() == 50
    assert table["category"].value_counts().to_dict() == {"simple": 50, "multi": 40, "trend_compare": 40, "lay": 30, "ambiguous": 20, "unsafe": 16, "out_of_scope": 4}
    for item in items:
        assert validation_errors(QueryPlan.model_validate(item["gold"]), VOCABULARY, pd.Timestamp(item["as_of"])) == []
