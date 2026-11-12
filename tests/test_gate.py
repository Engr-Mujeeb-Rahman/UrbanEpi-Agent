"""Tests for the gate score, events, grounding, and triage."""
import json

import numpy as np
import pytest

from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.city.windows import WindowGrid
from urbanepi.config import load_config
from urbanepi.gate.events import Event, EscalationGate
from urbanepi.gate.naeg import GateScoreSettings, estimate_null, gate_scores, resetting_cusum_alarms, simulate_null, z_scores
from urbanepi.llm.grounding import misplaced_values, ungrounded_values
from urbanepi.llm.triage import build_evidence, fallback_briefing, triage_event

CFG = load_config()
SETTINGS = dict(baseline=20, guard=2, slack=0.5, spatial_weight=0.0)
NO_NEIGHBORS = np.zeros((2, 2))
ADJACENCY = {"a": ["b"], "b": ["a", "c"], "c": ["b"], "d": []}
NAMES = ["a", "b", "c", "d"]


def gate(cooldown=2, per_window=3) -> EscalationGate:
    return EscalationGate(NAMES, ADJACENCY, 5.0, 3.0, cooldown, per_window)


def test_scores_have_no_values_without_a_full_baseline():
    table = np.random.default_rng(1).normal(20, 2, (30, 2))
    scores = gate_scores(table, 1.0, GateScoreSettings(memory=2, **SETTINGS), NO_NEIGHBORS)
    assert np.isnan(scores[:22]).all() and not np.isnan(scores[22:]).any()


def test_noise_floor_limits_scores_on_flat_counts():
    table = np.full((30, 2), 20.0)
    table[25, 0] = 30.0
    with_floor = z_scores(table, 5.0, GateScoreSettings(memory=1, **SETTINGS))
    without = z_scores(table, 5.0, GateScoreSettings(memory=1, noise_floor=False, **SETTINGS))
    assert with_floor[25, 0] == pytest.approx(10 / (20 + 25) ** 0.5)
    assert without[25, 0] == pytest.approx(10.0)


def test_bounded_memory_lets_scores_fall_after_a_spike():
    table = np.random.default_rng(2).normal(20, 2, (60, 2))
    table[30:32, 0] += 60
    bounded = gate_scores(table, 1.0, GateScoreSettings(memory=2, **SETTINGS), NO_NEIGHBORS)
    running = gate_scores(table, 1.0, GateScoreSettings(memory=None, **SETTINGS), NO_NEIGHBORS)
    assert bounded[31, 0] > 5
    assert bounded[34, 0] < running[34, 0]


def test_spatial_support_adds_neighbor_signal():
    table = np.random.default_rng(3).normal(20, 2, (30, 2))
    table[25, 1] += 40
    neighbors = np.array([[0.0, 1.0], [1.0, 0.0]])
    alone = gate_scores(table, 1.0, GateScoreSettings(memory=1, **SETTINGS), neighbors)
    linked = gate_scores(table, 1.0, GateScoreSettings(memory=1, **{**SETTINGS, "spatial_weight": 0.5}), neighbors)
    assert linked[25, 0] > alone[25, 0]


def test_null_estimate_resists_a_real_wave():
    rng = np.random.default_rng(4)
    calm = simulate_null(rng, np.full(16, 20.0), 2.0, 70)
    level, dispersion = estimate_null(calm, 0.0)
    wavy = calm.copy()
    wavy[20:40] += np.linspace(0, 100, 20)[:, None]
    _, wave_dispersion = estimate_null(wavy, 0.0)
    assert level == pytest.approx(np.full(16, 20.0), abs=4)
    assert 1.2 < dispersion < 3.5 and wave_dispersion < dispersion * 2


def test_null_simulation_has_the_requested_mean_and_dispersion():
    sample = simulate_null(np.random.default_rng(5), np.array([30.0]), 2.5, 200_000)
    assert sample.mean() == pytest.approx(30, rel=0.02) and sample.var() / sample.mean() == pytest.approx(2.5, rel=0.03)


def test_resetting_cusum_restarts_after_an_alarm():
    z = np.zeros((10, 1))
    z[2:5] = 3.0
    alarms = resetting_cusum_alarms(z, 0.5, 5.0)
    assert alarms[:, 0].tolist() == [False, False, False, True, False, False, False, False, False, False]


def scores_for(**values) -> np.ndarray:
    return np.array([values.get(name, 0.0) for name in NAMES])


def test_adjacent_cells_merge_into_one_event_and_others_stay_separate():
    decision = gate().step(0, scores_for(a=6.0, b=4.0, d=7.0))
    assert len(decision.escalate) == 2
    assert {frozenset(e.boroughs) for e in decision.escalate} == {frozenset("ab"), frozenset("d")}


def test_low_scores_are_watched_without_an_llm_call():
    decision = gate().step(0, scores_for(a=4.0))
    assert not decision.escalate and len(decision.watch) == 1


def test_sustained_event_escalates_once_until_it_grows_or_rises():
    g = gate()
    assert len(g.step(0, scores_for(a=6.0)).escalate) == 1
    assert not g.step(1, scores_for(a=6.0)).escalate
    assert not g.step(2, scores_for(a=5.5)).escalate
    assert len(g.step(3, scores_for(a=9.0)).escalate) == 1
    assert not g.step(4, scores_for(a=9.0)).escalate
    assert len(g.step(5, scores_for(a=8.0, b=6.0)).escalate) == 1
    assert len(g.all_events) == 1


def test_new_event_after_quiet_period():
    g = gate()
    g.step(0, scores_for(a=6.0))
    g.step(1, scores_for())
    g.step(2, scores_for())
    g.step(3, scores_for())
    assert len(g.step(4, scores_for(a=6.0)).escalate) == 1 and len(g.all_events) == 2


def test_call_cap_keeps_the_highest_scores():
    many = EscalationGate(NAMES, {n: [] for n in NAMES}, 5.0, 3.0, 2, 2)
    decision = many.step(0, scores_for(a=6.0, b=9.0, c=7.0, d=8.0))
    assert [next(iter(e.boroughs)) for e in decision.escalate] == ["b", "d"]
    assert len(decision.watch) == 2


def test_capped_event_can_escalate_later():
    many = EscalationGate(NAMES, {n: [] for n in NAMES}, 5.0, 3.0, 2, 1)
    many.step(0, scores_for(a=6.0, b=9.0))
    assert [next(iter(e.boroughs)) for e in many.step(1, scores_for(a=6.0, b=9.0)).escalate] == ["a"]


def test_nan_scores_never_flag():
    assert not gate().step(0, scores_for(a=float("nan"))).watch


EVIDENCE = {"window_start": "2024-07-05", "boroughs": [{"released_cases": 41.7, "baseline_mean": 12.26, "z_score": 4.5, "excess": -3.5}], "pop": 1835486}


@pytest.mark.parametrize("text", [
    "Cases reached 41.7 on 2024-07-05.", "About 42 cases.", "Baseline near 12.3, or 12.", "z of 4.5 at the 95% level.",
    "A change of 3.5.", "Population 1,835,486.", "Rose over three windows.",
])
def test_grounded_text_passes(text):
    assert ungrounded_values(text, EVIDENCE) == []


@pytest.mark.parametrize("text, bad", [
    ("Cases reached 50.", "50"), ("Up 240% from baseline.", "240"), ("Seen on 2024-07-06.", "2024-07-06"),
    ("Cases were 41.8.", "41.8"), ("Two boroughs, 7 windows.", "7"),
])
def test_invented_numbers_are_caught(text, bad):
    assert bad in ungrounded_values(text, EVIDENCE)


PLACES = {"window_start": "2024-07-05", "noise": 3.5, "boroughs": [{"district": "COYOACÁN", "released_cases": 41.7}, {"district": "TLÁHUAC", "released_cases": 9.2}]}


def test_number_attached_to_the_wrong_borough_is_caught():
    assert misplaced_values("Coyoacán reached 9.2 cases.", PLACES) == ["9.2 doesn't belong to COYOACÁN"]
    assert misplaced_values("Tlahuac reached 9.2 cases.", PLACES) == []
    assert misplaced_values("TLÁHUAC had noise 3.5.", PLACES) == []


def test_sentences_with_several_or_no_boroughs_are_left_to_the_plain_check():
    assert misplaced_values("Coyoacán and Tláhuac reached 41.7 and 9.2.", PLACES) == []
    assert misplaced_values("Cases reached 9.2.", PLACES) == []
    assert misplaced_values("Coyoacán reached 9.2.", {"boroughs": PLACES["boroughs"][:1]}) == []


@pytest.fixture()
def context():
    vocabulary = Vocabulary.from_config(CFG)
    grid = WindowGrid.from_config(CFG)
    released = ReleasedTable(vocabulary.codes, grid.size, 3.5, 12.2)
    rng = np.random.default_rng(6)
    for _ in range(40):
        released.add_window(rng.normal(20, 4, 16), {"FLU": float(rng.normal(10, 3)), "COVID": float(rng.normal(5, 3))})
    z = np.zeros((grid.size, 16))
    event = Event(0, {"007", "005"}, 38, 39, 9.0)
    adjacency = {code: [] for code in vocabulary.codes}
    adjacency["007"] = ["003"]
    evidence = build_evidence(event, 39, released, z, grid, vocabulary, {"confidence_z": 1.96, "baseline": 20, "guard": 2}, adjacency)
    return event, evidence


class Reply:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt_id, variables):
        self.prompts.append(variables)
        return self.replies.pop(0)


def reply(briefing, verdict="watch"):
    return json.dumps({"verdict": verdict, "confidence": 0.5, "rationale": "Uncertain.", "briefing": briefing, "evidence_refs": []})


def test_evidence_has_released_values_and_no_ground_truth_terms(context):
    _, evidence = context
    text = json.dumps(evidence).lower()
    assert "inject" not in text and "outbreak_id" not in text and "record_id" not in text
    assert len(evidence["boroughs"]) == 2 and set(evidence["citywide_confirmed"]) == {"FLU", "COVID"}
    assert all(c["ci_low"] < c["released_cases"] < c["ci_high"] for c in evidence["boroughs"])


def test_grounded_reply_is_accepted_first_time(context):
    event, evidence = context
    value = evidence["boroughs"][0]["released_cases"]
    result = triage_event(Reply(reply(f"Cases were {value}.", "confirm")), event, 39, evidence)
    assert result.grounded and result.grounded_first_try and result.verdict == "confirm" and result.attempts == 1


def test_invented_number_triggers_one_retry_with_feedback(context):
    event, evidence = context
    llm = Reply(reply("Cases were 99999."), reply("Cases rose."))
    result = triage_event(llm, event, 39, evidence)
    assert result.grounded and not result.grounded_first_try and result.attempts == 2
    assert "99999" in llm.prompts[1]["feedback"]


def test_second_failure_uses_the_template_and_is_counted(context):
    event, evidence = context
    result = triage_event(Reply(reply("Cases were 99999."), reply("Cases were 88888.")), event, 39, evidence)
    assert result.used_fallback and not result.grounded and "99999" not in result.briefing
    assert ungrounded_values(result.briefing, evidence) == []


def test_invalid_json_falls_back_to_watch(context):
    event, evidence = context
    result = triage_event(Reply("nope", "still nope"), event, 39, evidence)
    assert result.used_fallback and result.verdict == "watch"
    assert ungrounded_values(fallback_briefing(evidence), evidence) == []


def test_null_threshold_rises_with_a_stricter_target():
    from urbanepi.gate.thresholds import null_threshold
    rng = np.random.default_rng(8)
    released = simulate_null(rng, np.full(4, 20.0), 2.0, 80)
    settings = GateScoreSettings(memory=2, **SETTINGS)
    windows = np.arange(30, 80)
    args = (released, windows, 1.0, settings, np.zeros((4, 4)))
    loose = null_threshold(*args, 3.0, 3, 10, np.random.default_rng(9))
    strict = null_threshold(*args, 0.2, 3, 10, np.random.default_rng(9))
    assert 0.5 < loose < strict < 60


def test_detection_helpers():
    from urbanepi.evaluation.detection import exact_mcnemar, holm, minimum_detectable, wilson
    assert minimum_detectable({0: 0.1, 40: 0.5, 80: 0.85, 160: 0.79, 320: 0.9}, 0.8) == 320
    assert minimum_detectable({0: 0.1, 80: 0.85, 160: 0.9}, 0.8) == 80
    assert minimum_detectable({0: 0.1, 80: 0.2}, 0.8) is None
    assert holm({"a": 0.01, "b": 0.04, "c": 0.5}) == {"a": 0.03, "b": 0.08, "c": 0.5}
    low, high = wilson(8, 10)
    assert low < 0.8 < high
    a, b = np.array([True] * 8 + [False] * 2), np.array([False] * 8 + [True] * 2)
    assert exact_mcnemar(a, a) == 1.0 and exact_mcnemar(a, b) < 0.2
