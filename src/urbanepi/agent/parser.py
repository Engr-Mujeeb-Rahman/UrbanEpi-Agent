"""Turn a question into a validated plan, with one repair attempt."""
import json

import pandas as pd
from pydantic import ValidationError

from urbanepi.agent.qir import QueryPlan, validation_errors
from urbanepi.agent.vocabulary import Vocabulary


class ParseFailure(ValueError):
    """The LLM's output could not be turned into a valid plan."""


def _check(text: str, vocabulary: Vocabulary, as_of: pd.Timestamp) -> tuple[QueryPlan | None, list[str]]:
    try:
        plan = QueryPlan.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as error:
        return None, [str(error)[:800]]
    return plan, validation_errors(plan, vocabulary, as_of)


def parse_question(llm, question: str, vocabulary: Vocabulary, as_of: pd.Timestamp, prompt_id: str = "parse_v5") -> QueryPlan:
    """Call the LLM with public schema text only. Raises ParseFailure if the repair also fails."""
    variables = {"question": question, "schema": vocabulary.describe(), "as_of": str(as_of.date())}
    text = llm.complete(prompt_id, variables)
    plan, errors = _check(text, vocabulary, as_of)
    if plan is not None and not errors:
        return plan
    text = llm.complete("repair_v1", {**variables, "previous": text, "errors": "; ".join(errors)})
    plan, errors = _check(text, vocabulary, as_of)
    if plan is None or errors:
        raise ParseFailure("; ".join(errors))
    return plan
