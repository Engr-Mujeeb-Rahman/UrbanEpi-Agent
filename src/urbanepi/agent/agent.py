"""Query agent: from a question to an answer, a confidence interval, and a privacy cost."""
import math
from dataclasses import dataclass, field

import pandas as pd

from urbanepi.agent.parser import ParseFailure, parse_question
from urbanepi.agent.pb_qd import Atom, PlanResult, decompose, reuse_atom
from urbanepi.agent.qir import QueryPlan, policy_refusal, validation_errors
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.aggregator.system import AdhocFederation
from urbanepi.city.windows import WindowGrid
from urbanepi.privacy.accounting import epsilon_from_rho


@dataclass
class AnswerLine:
    label: str
    value: float
    ci_low: float
    ci_high: float


@dataclass
class Answer:
    status: str
    message: str = ""
    lines: list[AnswerLine] = field(default_factory=list)
    rho_spent: float = 0.0
    epsilon_spent: float = 0.0
    reused_cells: int = 0
    fresh_cells: int = 0
    notes: list[str] = field(default_factory=list)
    plan: QueryPlan | None = None


class QueryAgent:
    """Runs Algorithm 1. It sees only public metadata and released values."""

    def __init__(self, llm, vocabulary: Vocabulary, grid: WindowGrid, released: ReleasedTable, federation: AdhocFederation,
                 adhoc_budget: float, cfg: dict):
        self.llm, self.vocabulary, self.grid, self.released, self.federation = llm, vocabulary, grid, released, federation
        self.adhoc_remaining, self.cfg = adhoc_budget, cfg
        self.rho_per_query = adhoc_budget / cfg["planned_adhoc_queries"]

    def ask(self, question: str, as_of: pd.Timestamp) -> Answer:
        try:
            plan = parse_question(self.llm, question, self.vocabulary, as_of)
        except ParseFailure as failure:
            return Answer("clarify", f"I couldn't interpret that question: {failure}")
        return self.answer_plan(plan, as_of)

    def answer_plan(self, plan: QueryPlan, as_of: pd.Timestamp) -> Answer:
        if plan.needs_clarification:
            return Answer("clarify", plan.needs_clarification, plan=plan)
        refusal = policy_refusal(plan, self.cfg["minimum_short_query_days"])
        if refusal:
            return Answer("refuse", refusal, plan=plan)
        errors = validation_errors(plan, self.vocabulary, as_of)
        if errors:
            return Answer("clarify", "; ".join(errors), plan=plan)

        result = decompose(plan, self.vocabulary, self.grid, self.released, min(self.rho_per_query, self.adhoc_remaining) or 1e-12,
                           self.cfg)
        if result.subqueries:
            if result.rho > self.adhoc_remaining + 1e-12:
                return Answer("refuse", "The privacy budget for new questions is used up.", plan=plan)
            if not self.federation.release(result, self.released):
                return Answer("refuse", "An institution refused because its privacy budget is used up.", plan=plan)
            self.adhoc_remaining -= result.rho
        return self._compose(plan, result)

    def _compose(self, plan: QueryPlan, result: PlanResult) -> Answer:
        lines = []
        z = self.cfg["confidence_z"]
        for output in result.outputs:
            value, variance = 0.0, 0.0
            for atom, coefficient in output.terms:
                atom_value, atom_variance = self._cell(atom)
                value += coefficient * atom_value
                variance += coefficient ** 2 * atom_variance
            margin = z * math.sqrt(variance)
            lines.append(AnswerLine(output.label, value, value - margin, value + margin))
        notes = list(result.notes)
        counts = [line for line, output in zip(lines, result.outputs) if len(output.terms) == 1]
        if any(line.ci_low <= 0 for line in counts):
            notes.append("At least one interval includes zero, so that value can't be told apart from no cases at this privacy level.")
        delta = self.cfg["delta"]
        return Answer("answer", lines=lines, rho_spent=result.rho, epsilon_spent=epsilon_from_rho(result.rho, delta) if result.rho else 0.0,
                      reused_cells=len(result.reusable), fresh_cells=sum(len(q.atoms) for q in result.subqueries),
                      notes=notes, plan=plan)

    def _cell(self, atom: Atom) -> tuple[float, float]:
        cell = reuse_atom(atom, self.released, self.vocabulary.codes)
        if cell is None:
            raise RuntimeError(f"No released value for {atom}")
        return cell


def format_answer(answer: Answer) -> str:
    """Plain text answer built only from released values."""
    if answer.status == "refuse":
        return (f"[refuse] {answer.message} This system doesn't say whether any person is in the data. "
                "It can give aggregate counts and rates of respiratory illness by borough, age band, and period.")
    if answer.status != "answer":
        return f"[{answer.status}] {answer.message}"
    rows = [f"{line.label}: {line.value:.1f} (95% CI {line.ci_low:.1f} to {line.ci_high:.1f})" for line in answer.lines]
    rows.append(f"Privacy cost: rho = {answer.rho_spent:.5f} (epsilon about {answer.epsilon_spent:.3f}); "
                f"{answer.reused_cells} cells reused, {answer.fresh_cells} fresh.")
    rows.append("These are reported cases at sentinel units, with privacy noise added. They don't cover everyone who got sick.")
    return "\n".join(rows + answer.notes)
