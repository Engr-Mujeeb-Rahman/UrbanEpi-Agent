"""PB-QD: privacy-budgeted query decomposition (Docs/design.md section 5)."""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from urbanepi.agent.qir import QueryPlan
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.city.windows import WindowGrid
from urbanepi.privacy.accounting import rho_for, sigma_for

_COARSER_BUCKET = {"window": "week", "week": "month", "month": "none"}
_BUCKET_DAYS = {"week": 7, "month": 30}


@dataclass(frozen=True)
class Atom:
    """One cell of a question: a syndrome over districts, whole windows, and optional filters."""

    syndrome: str
    districts: tuple[str, ...]
    first_window: int
    last_window: int
    age_bands: tuple[str, ...] | None = None
    condition: str | None = None


@dataclass
class Output:
    """One reported number: a weighted sum of atoms."""

    label: str
    terms: list[tuple[Atom, float]]


@dataclass
class SubQuery:
    """Fresh atoms sharing one noise level. Atoms in a sub-query are disjoint."""

    syndrome: str
    atoms: list[Atom]
    sensitivity: float
    weight: float
    sigma: float = 0.0
    rho: float = 0.0


@dataclass
class PlanResult:
    atoms: list[Atom]
    outputs: list[Output]
    reusable: list[Atom]
    subqueries: list[SubQuery]
    rho: float
    notes: list[str] = field(default_factory=list)


def allocate_noise(sensitivities: list[float], weights: list[float], rho: float) -> list[float]:
    """Minimize the weighted sum of noise variances, subject to the total privacy cost. Returns SDs."""
    share = sum(d * math.sqrt(w) for d, w in zip(sensitivities, weights))
    return [math.sqrt((d / math.sqrt(w)) * share / (2 * rho)) for d, w in zip(sensitivities, weights)]


def _window_ranges(start: pd.Timestamp, end: pd.Timestamp, bucket: str, grid: WindowGrid) -> list[tuple[int, int]]:
    """Snap a time range and its buckets to the window grid. Reports state the actual dates."""
    first = int(grid.index_of(pd.Series([start]))[0])
    last = min(int(grid.index_of(pd.Series([end]))[0]), grid.size - 1)
    if bucket == "none":
        return [(first, last)]
    if bucket == "window":
        return [(w, w) for w in range(first, last + 1)]
    windows = last - first + 1
    pieces = max(1, round(windows / (_BUCKET_DAYS[bucket] / grid.window_days)))
    edges = [first + round(i * windows / pieces) for i in range(pieces + 1)]
    return [(a, b - 1) for a, b in zip(edges, edges[1:])]


def expand_plan(plan: QueryPlan, vocabulary: Vocabulary, grid: WindowGrid) -> tuple[list[Atom], list[Output]]:
    """Turn a valid plan into atoms and the outputs that combine them."""
    districts = tuple(sorted(plan.districts)) if plan.districts else tuple(vocabulary.codes)
    start, end = pd.Timestamp(plan.time.start), pd.Timestamp(plan.time.end)
    ranges = _window_ranges(start, end, plan.time.bucket, grid)
    previous: list[tuple[int, int]] = []
    if plan.comparison.type == "previous_period":
        length = ranges[-1][1] - ranges[0][0] + 1
        if ranges[0][0] - length >= 0:
            previous = [(ranges[0][0] - length, ranges[0][0] - 1)]
    condition = plan.conditions[0] if plan.conditions else None
    bands = tuple(sorted(plan.age_bands)) if plan.age_bands else None
    place_groups = [(d,) for d in districts] if "district" in plan.group_by or plan.comparison.type == "between_districts" else [districts]
    band_groups: list[tuple[str, ...] | None] = [bands]
    if "age_band" in plan.group_by:
        band_groups = [(b,) for b in (bands or vocabulary.age_bands)]
    rated = plan.metric == "rate_per_100k"

    def scale(places: tuple[str, ...]) -> float:
        return 100_000 / sum(vocabulary.populations[p] for p in places) if rated else 1.0

    atoms: dict[Atom, None] = {}
    outputs: list[Output] = []

    def atom_for(syndrome: str, places: tuple[str, ...], band: tuple[str, ...] | None, window_range: tuple[int, int]) -> Atom:
        atom = Atom(syndrome, places, window_range[0], window_range[1], band, condition)
        atoms[atom] = None
        return atom

    def dates(window_range: tuple[int, int]) -> str:
        begin = grid.first_day(window_range[0]).date()
        finish = (grid.first_day(window_range[1] + 1) - pd.Timedelta(days=1)).date()
        return f"{begin} to {finish}"

    for syndrome in plan.syndromes:
        for band in band_groups:
            band_text = f", age {'+'.join(band)}" if band else ""
            shown = vocabulary.ontology[syndrome]["label"]
            for places in place_groups:
                name = "citywide" if len(places) == len(vocabulary.codes) else "+".join(vocabulary.borough_names[p] for p in places)
                for window_range in ranges:
                    atom = atom_for(syndrome, places, band, window_range)
                    outputs.append(Output(f"{shown}, {name}{band_text}, {dates(window_range)}", [(atom, scale(places))]))
                    if previous:
                        earlier = atom_for(syndrome, places, band, previous[0])
                        outputs.append(Output(f"{shown}, {name}{band_text}, {dates(previous[0])}", [(earlier, scale(places))]))
                        outputs.append(Output(f"{shown}, {name}{band_text}, change from previous period",
                                              [(atom, scale(places)), (earlier, -scale(places))]))
            if plan.comparison.type == "between_districts":
                first_place, second_place = place_groups
                for window_range in ranges:
                    a = atom_for(syndrome, first_place, band, window_range)
                    b = atom_for(syndrome, second_place, band, window_range)
                    outputs.append(Output(
                        f"{shown}{band_text}, {vocabulary.borough_names[first_place[0]]} minus {vocabulary.borough_names[second_place[0]]}, {dates(window_range)}",
                        [(a, scale(first_place)), (b, -scale(second_place))]))
    return list(atoms), outputs


def reuse_atom(atom: Atom, released: ReleasedTable, all_codes: list[str]) -> tuple[float, float] | None:
    """Answer an atom from released cells, if possible. Returns the value and its noise variance."""
    if atom in released.adhoc:
        return released.adhoc[atom]
    if atom.age_bands is not None or atom.condition is not None:
        return None
    if atom.syndrome == "RESP":
        return released.resp_total(atom.districts, atom.first_window, atom.last_window)
    if sorted(atom.districts) == all_codes:
        return released.citywide_total(atom.syndrome, atom.first_window, atom.last_window)
    return None


def plan_subqueries(fresh: list[Atom], outputs: list[Output], rho: float, unit: str, assumed_records: int,
                    uniform: bool = False) -> list[SubQuery]:
    """Group fresh atoms by syndrome and set each group's noise.

    Atoms of one syndrome are disjoint, so a record affects only one of them and they share a cost.
    Different syndromes can overlap, so their costs add.
    """
    fresh_set = set(fresh)
    weights: dict[str, float] = {}
    for output in outputs:
        for atom, coefficient in output.terms:
            if atom in fresh_set:
                weights[atom.syndrome] = weights.get(atom.syndrome, 0.0) + coefficient ** 2
    groups: dict[str, list[Atom]] = {}
    for atom in fresh:
        groups.setdefault(atom.syndrome, []).append(atom)
    subqueries = []
    for syndrome, atoms in groups.items():
        periods = len({(a.first_window, a.last_window) for a in atoms})
        sensitivity = 1.0 if unit == "record" else math.sqrt(min(periods, assumed_records))
        subqueries.append(SubQuery(syndrome, atoms, sensitivity, weights.get(syndrome, 1.0)))
    if not subqueries:
        return []
    if uniform:
        share = rho / len(subqueries)
        sigmas = [sigma_for(share, q.sensitivity) for q in subqueries]
    else:
        sigmas = allocate_noise([q.sensitivity for q in subqueries], [q.weight for q in subqueries], rho)
    for query, sigma in zip(subqueries, sigmas):
        query.sigma, query.rho = sigma, rho_for(sigma, query.sensitivity)
    return subqueries


def _expected_size(atom: Atom, released: ReleasedTable, vocabulary: Vocabulary, cfg: dict) -> float:
    """Rough public estimate of an atom's count, from released values only."""
    base = released.resp_total(atom.districts, atom.first_window, atom.last_window)
    if base is None:
        return 1.0
    size = base[0]
    if atom.syndrome != "RESP":
        city = released.resp_total(tuple(vocabulary.codes), atom.first_window, atom.last_window)
        part = released.citywide_total(atom.syndrome, atom.first_window, atom.last_window)
        if city and part and city[0] > 0:
            size *= float(np.clip(part[0] / city[0], 0.01, 1.0))
        else:
            size *= cfg["unmonitored_syndrome_share"]
    if atom.age_bands:
        size *= cfg["age_band_share"] * len(atom.age_bands)
    if atom.condition:
        size *= cfg["condition_share"]
    return max(size, 1.0)


def _coarsen(plan: QueryPlan) -> tuple[QueryPlan, str] | None:
    """Return a coarser plan, one step at a time, or None if nothing is left to merge."""
    bucket = plan.time.bucket
    if bucket in ("window", "week") or (bucket == "month" and plan.intent != "trend"):
        coarser = _COARSER_BUCKET[bucket]
        return plan.model_copy(update={"time": plan.time.model_copy(update={"bucket": coarser})}), f"time grouped by {coarser}"
    if "age_band" in plan.group_by:
        return plan.model_copy(update={"group_by": [g for g in plan.group_by if g != "age_band"]}), "age bands merged"
    if "district" in plan.group_by:
        return plan.model_copy(update={"group_by": [g for g in plan.group_by if g != "district"]}), "districts merged"
    return None


def decompose(plan: QueryPlan, vocabulary: Vocabulary, grid: WindowGrid, released: ReleasedTable, rho: float, cfg: dict,
              use_reuse: bool = True, uniform: bool = False, use_coarsening: bool = True) -> PlanResult:
    """Steps 4 to 11 of Algorithm 1."""
    notes: list[str] = []
    current = plan
    while True:
        atoms, outputs = expand_plan(current, vocabulary, grid)
        reusable = [a for a in atoms if reuse_atom(a, released, vocabulary.codes) is not None] if use_reuse else []
        fresh = [a for a in atoms if a not in set(reusable)]
        subqueries = plan_subqueries(fresh, outputs, rho, cfg["protected_unit"], cfg["assumed_records"], uniform) if fresh else []
        if not subqueries or not use_coarsening:
            break
        errors = [q.sigma / _expected_size(a, released, vocabulary, cfg) for q in subqueries for a in q.atoms]
        if float(np.median(errors)) <= cfg["rse_max"]:
            break
        coarser = _coarsen(current)
        if coarser is None:
            notes.append("Predicted error is high, and no coarser grouping is available.")
            break
        current, note = coarser
        notes.append(f"Coarsened to reduce error: {note}.")
    return PlanResult(atoms, outputs, reusable, subqueries, sum(q.rho for q in subqueries), notes)
