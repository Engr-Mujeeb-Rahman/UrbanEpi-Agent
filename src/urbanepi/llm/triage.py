"""LLM triage and briefing for escalated events. Uses released values and public metadata only."""
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.city.windows import WindowGrid
from urbanepi.gate.events import Event
from urbanepi.llm.grounding import misplaced_values, ungrounded_values


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["confirm", "watch", "dismiss"]
    confidence: float
    rationale: str
    briefing: str
    evidence_refs: list[str]


@dataclass
class TriageResult:
    event_id: int
    window: int
    verdict: str
    briefing: str
    rationale: str
    attempts: int
    grounded_first_try: bool
    grounded: bool
    used_fallback: bool
    evidence: dict


def build_evidence(event: Event, window: int, released: ReleasedTable, z: np.ndarray, grid: WindowGrid, vocabulary: Vocabulary,
                   settings: dict, adjacency: dict[str, list[str]]) -> dict:
    """Collect the released values for one event. Nothing here says whether an event was injected."""
    z_value = settings["confidence_z"]
    sigma = released.borough_sigma
    baseline, guard = settings["baseline"], settings["guard"]
    low = max(0, window - guard - baseline)
    high = max(1, window - guard)
    position = {code: i for i, code in enumerate(released.boroughs)}
    cells = []
    for code in sorted(event.boroughs):
        i = position[code]
        now = float(released.borough_table[window, i])
        mean = float(released.borough_table[low:high, i].mean())
        cells.append({
            "ref": f"cell:{code}:RESP:{grid.first_day(window).date()}",
            "district": vocabulary.borough_names[code],
            "released_cases": round(now, 1),
            "ci_low": round(now - z_value * sigma, 1),
            "ci_high": round(now + z_value * sigma, 1),
            "baseline_mean": round(mean, 1),
            "excess_over_baseline": round(now - mean, 1),
            "z_score": round(float(np.nan_to_num(z[window, i])), 1),
            "previous_windows": [round(float(v), 1) for v in released.borough_table[max(0, window - 3):window, i]],
            "two_sd_of_noise_and_baseline": round(2 * float(np.sqrt(max(mean, 0) + sigma ** 2)), 1),
            "population_2020": vocabulary.populations[code],
            "neighbor_z_scores": {vocabulary.borough_names[n]: round(float(np.nan_to_num(z[window, position[n]])), 1)
                                  for n in adjacency[code] if n not in event.boroughs},
        })
    citywide = {}
    for name, series in released.citywide_table.items():
        citywide[name] = {"released_this_window": round(float(series[window]), 1),
                          "released_previous_windows": [round(float(v), 1) for v in series[max(0, window - 3):window]],
                          "noise_sd": round(released.citywide_sigma, 1)}
    return {
        "window_start": str(grid.first_day(window).date()),
        "window_end": str((grid.first_day(window + 1)).date()),
        "first_flagged_window_start": str(grid.first_day(event.first_window).date()),
        "noise_sd_per_borough_cell": round(sigma, 1),
        "boroughs": cells,
        "citywide_confirmed": citywide,
    }


def fallback_briefing(evidence: dict) -> str:
    """Template briefing used when the LLM's text can't be grounded. Every number comes from the evidence."""
    rows = [f"{c['district']}: {c['released_cases']} cases (95% CI {c['ci_low']} to {c['ci_high']}), baseline {c['baseline_mean']}."
            for c in evidence["boroughs"]]
    return f"Automated alert for the window starting {evidence['window_start']}. " + " ".join(rows)


def triage_event(llm, event: Event, window: int, evidence: dict, prompt_id: str = "triage_v1") -> TriageResult:
    """Ask for a verdict and briefing, check every number, retry once, then fall back to a template."""
    feedback = "None."
    attempts = 0
    first_grounded = False
    output: TriageOutput | None = None
    for attempt in range(2):
        attempts += 1
        text = llm.complete(prompt_id, {"evidence": json.dumps(evidence, ensure_ascii=False, indent=1), "feedback": feedback})
        try:
            output = TriageOutput.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as error:
            output = None
            feedback = f"Your last reply was invalid: {str(error)[:300]}"
            continue
        written = f"{output.rationale} {output.briefing}"
        problems = ungrounded_values(written, evidence) + misplaced_values(written, evidence)
        if not problems:
            if attempt == 0:
                first_grounded = True
            return TriageResult(event.event_id, window, output.verdict, output.briefing, output.rationale, attempts,
                                first_grounded, True, False, evidence)
        feedback = f"These values aren't in the evidence, so remove or correct them: {problems[:20]}"
    verdict = output.verdict if output else "watch"
    rationale = output.rationale if output and not ungrounded_values(output.rationale, evidence) else "Unavailable."
    return TriageResult(event.event_id, window, verdict, fallback_briefing(evidence), rationale, attempts, False, False, True, evidence)
