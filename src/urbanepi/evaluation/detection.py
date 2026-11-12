"""Shared machinery for the detection experiments. Thresholds use validation-period released values only."""
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

from urbanepi.city.injector import make_outbreak_records, sample_spec
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.gate.c2 import alert_onsets
from urbanepi.gate.events import EscalationGate
from urbanepi.gate.naeg import estimate_null, simulate_null

Scorer = Callable[[np.ndarray, float], np.ndarray]


@dataclass
class Outbreak:
    size: int
    boroughs: tuple[str, ...]
    day_start: int
    duration: int
    days: np.ndarray
    places: np.ndarray
    holders: np.ndarray


class DetectionContext:
    """Loads the events once, and builds count tables for any window length."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.events = pd.read_parquet(cfg["data"]["events_path"])
        table = pd.read_csv(cfg["data"]["boroughs_path"], dtype={"code": str})
        self.boroughs = table["code"].tolist()
        self.populations = dict(zip(self.boroughs, table["population_2020"]))
        self.adjacency = json.loads(Path(cfg["data"]["adjacency_path"]).read_text(encoding="utf8"))
        self.institutions = list(cfg["institutions"].values())
        self.position = {code: i for i, code in enumerate(self.boroughs)}
        self.start = pd.Timestamp(cfg["study"]["start"])

    def counts(self, grid: WindowGrid) -> np.ndarray:
        return count_cells(self.events, grid, self.boroughs, self.institutions).astype(float)

    def windows_of(self, grid: WindowGrid, split: str) -> np.ndarray:
        first, last = self.cfg["study"]["splits"][split]
        return grid.split_windows(first, last)

    def draw_outbreaks(self, rng: np.random.Generator, sizes: list[int], trials: int, split: str = "test") -> dict[int, list[Outbreak]]:
        """Draw outbreaks inside the split. The same draws are reused across settings, so comparisons are paired."""
        grid = WindowGrid.from_config(self.cfg)
        windows = self.windows_of(grid, split)
        first = grid.first_day(int(windows[0]))
        last = grid.first_day(int(windows[-1]) + 1) - pd.Timedelta(days=1)
        source = self.events[(self.events["admitted"] >= first) & (self.events["admitted"] <= last)]
        holder = {name: i for i, name in enumerate(self.institutions)}
        plans: dict[int, list[Outbreak]] = {}
        for size in sizes:
            plans[size] = []
            for trial in range(trials):
                duration = int(rng.choice(self.cfg["injection"]["duration_days"]))
                spec = sample_spec(rng, self.cfg, self.adjacency, f"{split}-{size}-{trial}", size, duration, str(first.date()), str(last.date()))
                days = places = holders = np.array([], dtype=int)
                if size:
                    records = make_outbreak_records(rng, spec, source, self.populations, self.cfg["injection"]["lognormal_sigma"])
                    days = (records["admitted"] - self.start).dt.days.to_numpy()
                    places = np.array([self.position[b] for b in records["borough"]])
                    holders = np.array([holder[h] for h in records["institution"]])
                plans[size].append(Outbreak(size, spec.boroughs, (pd.Timestamp(spec.start) - self.start).days, duration, days, places, holders))
        return plans

    def single_borough_outbreaks(self, rng: np.random.Generator, code: str, size: int, trials: int) -> list[Outbreak]:
        """Draw outbreaks in one named borough, for per-borough detection limits."""
        grid = WindowGrid.from_config(self.cfg)
        windows = self.windows_of(grid, "test")
        first = grid.first_day(int(windows[0]))
        last = grid.first_day(int(windows[-1]) + 1) - pd.Timedelta(days=1)
        source = self.events[(self.events["admitted"] >= first) & (self.events["admitted"] <= last)]
        holder = {name: i for i, name in enumerate(self.institutions)}
        solo = dict(self.cfg)
        solo["injection"] = {**self.cfg["injection"], "max_extent": 1}
        out = []
        for trial in range(trials):
            duration = int(rng.choice(self.cfg["injection"]["duration_days"]))
            last_start = (last - first).days - duration + 1
            start = first + pd.Timedelta(days=int(rng.integers(0, last_start + 1)))
            from urbanepi.city.injector import OutbreakSpec
            spec = OutbreakSpec(f"{code}-{size}-{trial}", (code,), str(start.date()), duration, size)
            records = make_outbreak_records(rng, spec, source, self.populations, self.cfg["injection"]["lognormal_sigma"])
            out.append(Outbreak(size, (code,), (start - self.start).days, duration, (records["admitted"] - self.start).dt.days.to_numpy(),
                                np.array([self.position[b] for b in records["borough"]]), np.array([holder[h] for h in records["institution"]])))
        return out


def add_outbreak(base: np.ndarray, outbreak: Outbreak, grid: WindowGrid) -> np.ndarray:
    """Add an outbreak to a (windows, boroughs) table."""
    values = base.copy()
    if outbreak.size:
        window = outbreak.days // grid.window_days
        inside = window < grid.size
        np.add.at(values, (window[inside], outbreak.places[inside]), 1)
    return values


def outbreak_windows(outbreak: Outbreak, grid: WindowGrid) -> tuple[int, int]:
    return outbreak.day_start // grid.window_days, (outbreak.day_start + outbreak.duration - 1) // grid.window_days


def calibrate(released: np.ndarray, windows: np.ndarray, sigma: float, scorer: Scorer, target: float, window_days: int, seeds: int,
              rng: np.random.Generator) -> float:
    """Same rule as gate/thresholds.py, for any scorer: the lowest threshold that meets the simulated alert-rate target."""
    level, dispersion = estimate_null(released[windows], sigma)
    runs = []
    for _ in range(seeds):
        simulated = simulate_null(rng, level, dispersion, released.shape[0])
        noisy = simulated + rng.normal(0.0, sigma, simulated.shape) if sigma else simulated
        runs.append(scorer(noisy, sigma))
    days = len(windows) * window_days
    for candidate in np.arange(0.5, 60.01, 0.05):
        rate = np.mean([alert_onsets(r, candidate)[windows].sum() for r in runs]) / (days / 30)
        if rate <= target:
            return float(candidate)
    return float("inf")


def run_gate(scores: np.ndarray, windows: np.ndarray, boroughs: list[str], adjacency: dict, tau: float, gate_cfg: dict) -> list[tuple[int, set[str]]]:
    """Return every escalation as (window, boroughs)."""
    gate = EscalationGate(boroughs, adjacency, tau, tau * gate_cfg["tau_lo_ratio"], gate_cfg["cooldown_windows"], gate_cfg["max_events_per_window"])
    alerts = []
    for t in windows:
        for event in gate.step(int(t), scores[t]).escalate:
            alerts.append((int(t), set(event.boroughs)))
    return alerts


def first_hit(alerts: list[tuple[int, set[str]]], outbreak: Outbreak, grid: WindowGrid) -> int | None:
    """Return the first alert window overlapping the outbreak's place and time."""
    begin, stop = outbreak_windows(outbreak, grid)
    places = set(outbreak.boroughs)
    hits = [t for t, where in alerts if begin <= t <= stop and where & places]
    return min(hits) if hits else None


def wilson(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    denominator = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominator
    return round(max(0.0, center - margin), 3), round(min(1.0, center + margin), 3)


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact McNemar test for paired detection results."""
    only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
    if only_a + only_b == 0:
        return 1.0
    return float(binomtest(only_a, only_a + only_b, 0.5).pvalue)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    ordered = sorted(p_values.items(), key=lambda pair: pair[1])
    adjusted, running = {}, 0.0
    for rank, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (len(ordered) - rank)))
        adjusted[name] = round(running, 5)
    return adjusted


def minimum_detectable(pod: dict[int, float], target: float) -> int | None:
    """Return the smallest size meeting the target, requiring all larger sizes to meet it too."""
    sizes = sorted(s for s in pod if s)
    for size in sizes:
        if all(pod[s] >= target for s in sizes if s >= size):
            return size
    return None
