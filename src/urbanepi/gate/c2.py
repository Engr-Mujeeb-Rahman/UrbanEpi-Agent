"""EARS C2-style detector on released values, used as the simple baseline."""
import numpy as np


def c2_scores(table: np.ndarray, baseline: int, guard: int, min_variance: float = 1.0) -> np.ndarray:
    """Return z-scores with shape (windows, boroughs). Windows without a full baseline are NaN."""
    windows = table.shape[0]
    scores = np.full(table.shape, np.nan)
    for t in range(baseline + guard, windows):
        past = table[t - guard - baseline:t - guard]
        spread = np.maximum(past.var(axis=0, ddof=1), min_variance)
        scores[t] = (table[t] - past.mean(axis=0)) / np.sqrt(spread)
    return scores


def alert_onsets(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Mark the first window of each run of alerts in a borough."""
    above = np.nan_to_num(scores, nan=-np.inf) >= threshold
    onsets = above.copy()
    onsets[1:] &= ~above[:-1]
    return onsets
