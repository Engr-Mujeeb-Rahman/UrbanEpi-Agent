"""Escalation decision: merge flagged cells into events, then limit LLM calls (Docs/design.md section 6.2)."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    event_id: int
    boroughs: set[str]
    first_window: int
    last_window: int
    score: float
    escalated_window: int | None = None
    escalated_score: float = 0.0
    escalated_extent: int = 0


@dataclass
class WindowDecision:
    window: int
    escalate: list[Event] = field(default_factory=list)
    watch: list[Event] = field(default_factory=list)


class EscalationGate:
    """Stateful gate. It sees only scores computed from released values."""

    def __init__(self, boroughs: list[str], adjacency: dict[str, list[str]], tau_hi: float, tau_lo: float, cooldown: int, max_per_window: int):
        self.boroughs, self.adjacency = boroughs, adjacency
        self.tau_hi, self.tau_lo, self.cooldown, self.max_per_window = tau_hi, tau_lo, cooldown, max_per_window
        self.active: list[Event] = []
        self.all_events: list[Event] = []

    def _components(self, flagged: set[str]) -> list[set[str]]:
        # Sorted traversal, so the result does not depend on Python's per-process string hashing.
        components, remaining = [], set(flagged)
        while remaining:
            start = min(remaining)
            remaining.remove(start)
            frontier, component = [start], set()
            while frontier:
                code = frontier.pop()
                component.add(code)
                linked = set(self.adjacency[code]) & remaining
                remaining -= linked
                frontier.extend(sorted(linked))
            components.append(component)
        return components

    def _touches(self, event: Event, component: set[str]) -> bool:
        return bool(event.boroughs & component) or any(set(self.adjacency[b]) & event.boroughs for b in component)

    def step(self, window: int, scores: np.ndarray) -> WindowDecision:
        values = dict(zip(self.boroughs, scores))
        flagged = {code for code, value in values.items() if not np.isnan(value) and value >= self.tau_lo}
        self.active = [e for e in self.active if window - e.last_window <= self.cooldown]
        current: list[tuple[Event, float]] = []
        for component in self._components(flagged):
            score = max(values[code] for code in component)
            event = next((e for e in self.active if self._touches(e, component)), None)
            if event is None:
                event = Event(len(self.all_events), set(component), window, window, score)
                self.active.append(event)
                self.all_events.append(event)
            else:
                event.boroughs |= component
                event.last_window, event.score = window, score
            current.append((event, score))

        decision = WindowDecision(window)
        wanted = []
        for event, score in current:
            never = event.escalated_window is None
            rearm = not never and window - event.escalated_window >= self.cooldown and (
                score > event.escalated_score or len(event.boroughs) > event.escalated_extent)
            if score >= self.tau_hi and (never or rearm):
                wanted.append((score, event))
            else:
                decision.watch.append(event)
        wanted.sort(key=lambda pair: -pair[0])
        for rank, (score, event) in enumerate(wanted):
            if rank < self.max_per_window:
                event.escalated_window, event.escalated_score, event.escalated_extent = window, score, len(event.boroughs)
                decision.escalate.append(event)
            else:
                decision.watch.append(event)
        return decision
