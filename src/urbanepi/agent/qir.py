"""Typed query plan (QIR), its validator, and the refusal policy."""
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from urbanepi.agent.vocabulary import Vocabulary


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: date
    end: date
    bucket: Literal["none", "window", "week", "month"] = "none"


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["none", "previous_period", "between_districts"] = "none"


class QueryPlan(BaseModel):
    """The LLM's only query output. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")
    intent: Literal["count", "rate", "trend", "compare", "out_of_scope"]
    metric: Literal["cases", "rate_per_100k"] = "cases"
    syndromes: list[str] = []
    districts: list[str] | None = None
    time: TimeRange | None = None
    group_by: list[Literal["district", "age_band"]] = []
    age_bands: list[str] | None = None
    conditions: list[str] | None = None
    comparison: Comparison = Comparison()
    needs_clarification: str | None = None
    refusal_reason: str | None = None


def validation_errors(plan: QueryPlan, vocabulary: Vocabulary, as_of: pd.Timestamp) -> list[str]:
    """Return every reason the plan can't run. Refusals and clarifying questions have no further checks."""
    if plan.intent == "out_of_scope" or plan.needs_clarification:
        return []
    errors = []
    if not plan.syndromes:
        errors.append("syndromes must not be empty")
    errors += [f"unknown syndrome: {s}" for s in plan.syndromes if s not in vocabulary.ontology]
    errors += [f"unknown district: {d}" for d in plan.districts or [] if d not in vocabulary.borough_names]
    errors += [f"unknown age band: {a}" for a in plan.age_bands or [] if a not in vocabulary.age_bands]
    errors += [f"unknown condition: {c}" for c in plan.conditions or [] if c not in vocabulary.conditions]
    if plan.conditions and len(plan.conditions) > 1:
        errors.append("at most one condition is supported")
    if plan.time is None:
        errors.append("time is required")
    else:
        start, end = pd.Timestamp(plan.time.start), pd.Timestamp(plan.time.end)
        if start > end:
            errors.append("time.start is after time.end")
        if start < vocabulary.study_start or end > vocabulary.study_end:
            errors.append("time is outside the study period")
        if end > as_of:
            errors.append("time ends after the data available so far")
        if plan.comparison.type == "previous_period" and start - (end - start) - pd.Timedelta(days=1) < vocabulary.study_start:
            errors.append("the earlier period starts before the data does")
    if plan.metric == "rate_per_100k" and (plan.age_bands or plan.conditions or "age_band" in plan.group_by):
        errors.append("rates are available only for whole borough populations, not age groups or conditions")
    if plan.comparison.type == "between_districts" and len(plan.districts or []) != 2:
        errors.append("between_districts needs exactly two districts")
    if plan.comparison.type != "none" and plan.intent != "compare":
        errors.append("a comparison needs intent compare")
    if plan.intent == "compare" and plan.comparison.type == "none":
        errors.append("intent compare needs a comparison")
    if plan.intent == "trend" and plan.time is not None and plan.time.bucket == "none":
        errors.append("intent trend needs a time bucket")
    return errors


def policy_refusal(plan: QueryPlan, minimum_days: int) -> str | None:
    """Return a reason to refuse a valid plan, or None. Refusals cost no budget."""
    if plan.intent == "out_of_scope":
        return plan.refusal_reason or "The question is outside what this system can answer."
    narrow = plan.conditions and plan.age_bands and plan.districts and len(plan.districts) == 1
    if narrow and plan.time is not None and (plan.time.end - plan.time.start).days + 1 < minimum_days:
        return "The requested group is too narrow to report."
    return None


def _effective_bucket(time: TimeRange) -> str:
    """A bucket as long as the whole range gives one total, the same as no bucket."""
    days = (time.end - time.start).days + 1
    same_month = (time.start.year, time.start.month) == (time.end.year, time.end.month)
    single = {"none": True, "window": days <= 3, "week": days <= 7, "month": same_month}[time.bucket]
    return "none" if single else time.bucket


def normalized(plan: QueryPlan, vocabulary: Vocabulary) -> dict:
    """Canonical form for comparing plans. Choices that give the same answer compare as equal."""
    if plan.intent == "out_of_scope":
        return {"status": "refuse"}
    if plan.needs_clarification:
        return {"status": "clarify"}
    districts = sorted(plan.districts) if plan.districts else None
    if districts == vocabulary.codes:
        districts = None
    group_by = set(plan.group_by)
    if plan.comparison.type == "between_districts" or (districts and len(districts) == 1):
        group_by.discard("district")
    if plan.age_bands and len(plan.age_bands) == 1:
        group_by.discard("age_band")
    return {
        "status": "answer",
        "intent": "count" if plan.intent == "rate" else plan.intent,
        "metric": plan.metric,
        "syndromes": sorted(plan.syndromes),
        "districts": districts,
        "start": str(plan.time.start) if plan.time else None,
        "end": str(plan.time.end) if plan.time else None,
        "bucket": _effective_bucket(plan.time) if plan.time else None,
        "group_by": sorted(group_by),
        "age_bands": sorted(plan.age_bands) if plan.age_bands else None,
        "conditions": sorted(plan.conditions) if plan.conditions else None,
        "comparison": plan.comparison.type,
    }
