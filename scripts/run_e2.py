"""E2: answer quality on the frozen test questions, using gold plans so parsing errors don't mix in.

True counts come from the events table. Tests show the graph and table paths agree. Noise is added the way the system adds it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from urbanepi.agent.pb_qd import Atom, decompose, reuse_atom
from urbanepi.agent.qir import QueryPlan, policy_refusal
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config
from urbanepi.privacy.accounting import StreamBudget

cfg = load_config()
setup = cfg["e2"]
vocabulary = Vocabulary.from_config(cfg)
grid = WindowGrid.from_config(cfg)
settings = {**cfg["agent"], "assumed_records": cfg["privacy"]["assumed_records_per_person"], "delta": cfg["privacy"]["delta"]}
events = pd.read_parquet(cfg["data"]["events_path"])
events = events.assign(window=grid.index_of(events["admitted"]))
leaves_of: dict[str, set[str]] = {}


def leaves(name: str) -> set[str]:
    if name not in leaves_of:
        entry = vocabulary.ontology[name]
        found = {name} if entry["pcr_codes"] else set()
        for child in entry["includes"]:
            found |= leaves(child)
        leaves_of[name] = found
    return leaves_of[name]


truth_cache: dict[tuple, float] = {}


def true_count(atom: Atom, exact_term: bool = False) -> float:
    """Count an atom directly. With exact_term, only the named term counts, as if there were no ontology."""
    key = (atom, exact_term)
    if key in truth_cache:
        return truth_cache[key]
    part = events[events["window"].between(atom.first_window, atom.last_window) & events["borough"].isin(atom.districts)]
    if atom.age_bands:
        part = part[part["age_band"].isin(atom.age_bands)]
    if atom.condition:
        part = part[part[atom.condition]]
    if atom.syndrome != "RESP" or exact_term:
        wanted = {atom.syndrome} if exact_term else leaves(atom.syndrome)
        part = part[part["pathogens"].map(lambda p: bool(wanted & set(p)))]
    truth_cache[key] = float(len(part))
    return truth_cache[key]


items = [json.loads(line) for line in Path("data/benchmark/urbanepi_q.jsonl").read_text(encoding="utf8").splitlines()]
plans = []
for item in items:
    if item["split"] == "test" and item["category"] in ("simple", "multi", "trend_compare", "lay"):
        plan = QueryPlan.model_validate(item["gold"])
        if policy_refusal(plan, settings["minimum_short_query_days"]) is None:
            plans.append((item["id"], item["category"], plan))
through = grid.size - 1
base = count_cells(events, grid, vocabulary.codes, list(cfg["institutions"].values())).sum(axis=2).astype(float)
window_index = events["window"].to_numpy()
city = {name: np.bincount(window_index[events[flag].to_numpy()], minlength=grid.size).astype(float)
        for name, flag in (("FLU", "flu_confirmed"), ("COVID", "covid_confirmed"))}
ARMS = {"full": {}, "no_reuse": {"use_reuse": False}, "uniform_allocation": {"uniform": True}, "no_coarsening": {"use_coarsening": False},
        "exact_term_no_ontology": {"exact_term": True}}


def released_for(budget: StreamBudget, rng: np.random.Generator) -> ReleasedTable:
    released = ReleasedTable(vocabulary.codes, grid.size, budget.borough_sigma, budget.citywide_sigma)
    for t in range(through + 1):
        released.add_window(base[t] + rng.normal(0, budget.borough_sigma, base.shape[1]),
                            {n: city[n][t] + rng.normal(0, budget.citywide_sigma) for n in city})
    return released


def answer_errors(plan: QueryPlan, released: ReleasedTable, rho: float, rng: np.random.Generator, arm: dict) -> dict:
    options = {k: v for k, v in arm.items() if k != "exact_term"}
    exact = arm.get("exact_term", False)
    result = decompose(plan, vocabulary, grid, released, rho, settings, **options)
    sigma_of = {a: q.sigma for q in result.subqueries for a in q.atoms}
    values: dict[Atom, tuple[float, float]] = {}
    for atom in result.atoms:
        if atom in sigma_of:
            values[atom] = (true_count(atom, exact) + rng.normal(0, sigma_of[atom]), sigma_of[atom] ** 2)
        else:
            values[atom] = reuse_atom(atom, released, vocabulary.codes)
    errors, covered, widths = [], [], []
    for output in result.outputs:
        estimate = sum(c * values[a][0] for a, c in output.terms)
        variance = sum(c ** 2 * values[a][1] for a, c in output.terms)
        actual = sum(c * true_count(a) for a, c in output.terms)
        margin = settings["confidence_z"] * variance ** 0.5
        errors.append(abs(estimate - actual) / max(abs(actual), 1.0))
        covered.append(estimate - margin <= actual <= estimate + margin)
        widths.append(2 * margin)
    return {"error": float(np.median(errors)), "covered": float(np.mean(covered)), "width": float(np.median(widths)), "rho": result.rho,
            "free": not result.subqueries, "reused": len(result.reusable), "fresh": sum(len(q.atoms) for q in result.subqueries), "coarsened": bool(result.notes)}


results = {}
seeds = np.random.SeedSequence([cfg["seed"], 70])
for unit, epsilon in setup["settings"]:
    budget = StreamBudget.from_config(cfg, epsilon, unit)
    settings = {**settings, "protected_unit": unit}
    rho = budget.rho_total * cfg["privacy"]["budget_split"]["adhoc"] / settings["planned_adhoc_queries"]
    arms = ARMS if (unit, epsilon) == tuple(setup["primary"]) else {"full": {}}
    per_arm = {arm: {i: [] for i, _, _ in plans} for arm in arms}
    rows = {arm: [] for arm in arms}
    for s in range(setup["noise_seeds"]):
        rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 70, s, int(epsilon * 10), len(unit)]))
        released = released_for(budget, rng)
        for arm_name, arm in arms.items():
            for question, category, plan in plans:
                outcome = answer_errors(plan, released, rho, rng, arm)
                per_arm[arm_name][question].append(outcome["error"])
                rows[arm_name].append({"id": question, "category": category, "multi_syndrome": len(plan.syndromes) > 1, **outcome})
    summary = {}
    medians = {arm: np.array([np.median(per_arm[arm][i]) for i, _, _ in plans]) for arm in arms}
    for arm_name in arms:
        table = pd.DataFrame(rows[arm_name])
        summary[arm_name] = {
            "median_relative_error": round(float(np.median(medians[arm_name])), 3),
            "ci_coverage": round(float(table["covered"].mean()), 3), "median_ci_width": round(float(table["width"].median()), 1),
            "share_answered_free": round(float(table["free"].mean()), 3), "share_coarsened": round(float(table["coarsened"].mean()), 3),
            "mean_rho_per_question": round(float(table["rho"].mean()), 6),
            "median_error_by_category": table.groupby("category")["error"].median().round(3).to_dict(),
            "median_error_multi_syndrome": round(float(table[table["multi_syndrome"]]["error"].median()), 3) if table["multi_syndrome"].any() else None,
        }
        if arm_name != "full":
            differences = medians["full"] - medians[arm_name]
            summary[arm_name]["wilcoxon_p_vs_full"] = round(float(wilcoxon(differences[differences != 0]).pvalue), 5) if (differences != 0).any() else 1.0
            multi = [k for k, (_, _, p) in enumerate(plans) if len(p.syndromes) > 1]
            if multi:
                summary[arm_name]["multi_syndrome_median_error_full_vs_arm"] = [round(float(np.median(medians["full"][multi])), 3), round(float(np.median(medians[arm_name][multi])), 3)]
    fresh_per_full = pd.DataFrame(rows["full"]).groupby("id")["free"].mean()
    free_questions = int((fresh_per_full == 1).sum())
    summary["query_capacity"] = {
        "test_questions_answerable": len(plans), "answered_free_from_released_cells": free_questions,
        "fresh_budget_slots": settings["planned_adhoc_queries"], "answerable_within_budget": min(len(plans), free_questions + settings["planned_adhoc_queries"]),
        "without_reuse": min(len(plans), settings["planned_adhoc_queries"]),
    }
    results[f"{unit}_eps{epsilon}"] = {"borough_sigma": round(budget.borough_sigma, 2), "rho_per_fresh_question": round(rho, 6), "arms": summary}
    print(unit, epsilon, json.dumps(summary, indent=1), flush=True)
out = Path("experiments/results/final/e2.json")
out.write_text(json.dumps({"note": "Frozen test questions and period, gold plans.", "questions": len(plans), "noise_seeds": setup["noise_seeds"], "results": results}, indent=1), encoding="utf8")
