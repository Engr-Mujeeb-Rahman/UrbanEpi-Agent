"""Compare predicted plans with gold plans."""
from urbanepi.agent.qir import QueryPlan, normalized
from urbanepi.agent.vocabulary import Vocabulary


def compare_plans(predicted: QueryPlan, gold: QueryPlan, vocabulary: Vocabulary) -> dict:
    """Return exact match, status match, and the fields that differ."""
    left, right = normalized(predicted, vocabulary), normalized(gold, vocabulary)
    if left["status"] != right["status"] or left["status"] != "answer":
        return {"exact": left["status"] == right["status"], "status_match": left["status"] == right["status"], "wrong_fields": []}
    wrong = sorted(k for k in right if left.get(k) != right[k])
    return {"exact": not wrong, "status_match": True, "wrong_fields": wrong}
