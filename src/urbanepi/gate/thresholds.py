"""Set gate thresholds from a null model. Sigma is public, and released values cost no privacy budget."""
import numpy as np

from urbanepi.gate.c2 import alert_onsets
from urbanepi.gate.naeg import GateScoreSettings, estimate_null, gate_scores, simulate_null


def null_threshold(released: np.ndarray, calibration_windows: np.ndarray, sigma: float, settings: GateScoreSettings,
                   neighbors: np.ndarray, target_per_30_days: float, window_days: int, seeds: int,
                   rng: np.random.Generator) -> float:
    """Return the lowest threshold whose simulated alert rate meets the target."""
    level, dispersion = estimate_null(released[calibration_windows], sigma)
    windows = released.shape[0]
    runs = []
    for _ in range(seeds):
        simulated = simulate_null(rng, level, dispersion, windows)
        noisy = simulated + rng.normal(0.0, sigma, simulated.shape) if sigma else simulated
        runs.append(gate_scores(noisy, sigma, settings, neighbors))
    days = len(calibration_windows) * window_days
    for candidate in np.arange(0.5, 60.01, 0.05):
        rate = np.mean([alert_onsets(r, candidate)[calibration_windows].sum() for r in runs]) / (days / 30)
        if rate <= target_per_30_days:
            return float(candidate)
    return float("inf")
