"""Noise-aware score of the escalation gate (Docs/design.md section 6). Reads released values only."""
from dataclasses import dataclass

import numpy as np


def adjacency_matrix(boroughs: list[str], adjacency: dict[str, list[str]]) -> np.ndarray:
    """Row-normalized matrix that averages a value over each borough's neighbors."""
    position = {code: i for i, code in enumerate(boroughs)}
    matrix = np.zeros((len(boroughs), len(boroughs)))
    for code, neighbors in adjacency.items():
        for neighbor in neighbors:
            matrix[position[code], position[neighbor]] = 1 / len(neighbors)
    return matrix


@dataclass(frozen=True)
class GateScoreSettings:
    baseline: int
    guard: int
    slack: float
    memory: int | None  # Windows summed in the score. None uses a running CUSUM.
    spatial_weight: float
    noise_floor: bool = True


def z_scores(table: np.ndarray, sigma: float, settings: GateScoreSettings) -> np.ndarray:
    """Return z-scores of shape (windows, boroughs). Windows without a full baseline are NaN."""
    z = np.full(table.shape, np.nan)
    for t in range(settings.baseline + settings.guard, table.shape[0]):
        past = table[t - settings.guard - settings.baseline:t - settings.guard]
        mean = past.mean(axis=0)
        variance = past.var(axis=0, ddof=1)
        if settings.noise_floor:
            variance = np.maximum(variance, np.maximum(mean, 0) + sigma ** 2)
        z[t] = (table[t] - mean) / np.sqrt(np.maximum(variance, 1.0))
    return z


def gate_scores(table: np.ndarray, sigma: float, settings: GateScoreSettings, neighbors: np.ndarray) -> np.ndarray:
    """Return the gate score G of shape (windows, boroughs)."""
    z = z_scores(table, sigma, settings)
    excess = np.maximum(np.nan_to_num(z, nan=0.0) - settings.slack, 0.0)
    if settings.memory is None:
        held = np.zeros_like(excess)
        running = np.zeros(table.shape[1])
        for t in range(table.shape[0]):
            running = np.maximum(0.0, running + np.nan_to_num(z[t], nan=0.0) - settings.slack)
            held[t] = running
    else:
        cumulative = np.cumsum(excess, axis=0)
        held = cumulative.copy()
        held[settings.memory:] -= cumulative[:-settings.memory]
    support = np.maximum(np.nan_to_num(z, nan=0.0), 0.0) @ neighbors.T
    scores = held + settings.spatial_weight * support
    scores[np.isnan(z)] = np.nan
    return scores


def resetting_cusum_alarms(z: np.ndarray, slack: float, threshold: float) -> np.ndarray:
    """Standard CUSUM comparison detector: alarm at the threshold, then restart that borough from zero."""
    alarms = np.zeros(z.shape, dtype=bool)
    running = np.zeros(z.shape[1])
    for t in range(z.shape[0]):
        running = np.maximum(0.0, running + np.nan_to_num(z[t], nan=0.0) - slack)
        alarms[t] = running >= threshold
        running[alarms[t]] = 0.0
    return alarms


def estimate_null(table: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
    """Estimate each borough's level and overall overdispersion from released values only.

    Differences between neighboring windows respond little to slow waves, so real waves inflate this less.
    """
    level = np.maximum(np.median(table, axis=0), 0.5)
    differences = np.diff(table, axis=0)
    spread = (1.4826 * np.median(np.abs(differences - np.median(differences, axis=0)), axis=0)) ** 2 / 2
    dispersion = np.median(np.maximum(spread - sigma ** 2, 0.0) / level)
    return level, float(max(dispersion, 1.05))


def simulate_null(rng: np.random.Generator, level: np.ndarray, dispersion: float, windows: int) -> np.ndarray:
    """Draw outbreak-free counts from a negative binomial with the estimated level and overdispersion."""
    success = 1 / dispersion
    return rng.negative_binomial(level * success / (1 - success), success, size=(windows, len(level))).astype(float)


def naeg_scores(table: np.ndarray, sigma: float, baseline: int, guard: int, slack: float, spatial_weight: float,
                neighbors: np.ndarray) -> np.ndarray:
    """Original running-CUSUM score, kept so scripts/go_no_go_1b.py can still run."""
    settings = GateScoreSettings(baseline, guard, slack, None, spatial_weight)
    return gate_scores(table, sigma, settings, neighbors)
