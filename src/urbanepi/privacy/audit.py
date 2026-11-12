"""Empirical lower bound on epsilon. Passing supports correctness but does not prove privacy."""
import math

import numpy as np
from scipy.stats import beta


def _clopper_pearson(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return low, high


def empirical_epsilon(without: np.ndarray, with_record: np.ndarray, delta: float, thresholds: np.ndarray,
                      alpha: float = 0.05) -> float:
    """Return the largest epsilon the threshold tests can prove, at overall confidence 1 - alpha."""
    per_test = alpha / len(thresholds)
    best = 0.0
    for threshold in thresholds:
        true_rate, _ = _clopper_pearson(int((with_record > threshold).sum()), len(with_record), per_test)
        _, false_rate = _clopper_pearson(int((without > threshold).sum()), len(without), per_test)
        if true_rate > delta and false_rate > 0:
            best = max(best, math.log((true_rate - delta) / false_rate))
    return best
