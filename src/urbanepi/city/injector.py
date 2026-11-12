"""Add outbreaks with known place and time. Ground truth is for the evaluator only."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OutbreakSpec:
    """One injected outbreak."""

    outbreak_id: str
    boroughs: tuple[str, ...]
    start: str
    duration_days: int
    excess_cases: int


def sample_spec(rng: np.random.Generator, cfg: dict, adjacency: dict[str, list[str]], outbreak_id: str,
                excess_cases: int, duration_days: int, test_start: str, test_end: str) -> OutbreakSpec:
    """Draw a place and start date. The full outbreak stays inside the test period, on adjacent boroughs."""
    extent = int(rng.integers(1, cfg["injection"]["max_extent"] + 1))
    chosen = [str(rng.choice(sorted(adjacency)))]
    while len(chosen) < extent:
        options = sorted({n for b in chosen for n in adjacency[b]} - set(chosen))
        chosen.append(str(rng.choice(options)))
    last_start = pd.Timestamp(test_end) - pd.Timedelta(days=duration_days - 1)
    span = (last_start - pd.Timestamp(test_start)).days
    start = pd.Timestamp(test_start) + pd.Timedelta(days=int(rng.integers(0, span + 1)))
    return OutbreakSpec(outbreak_id, tuple(sorted(chosen)), start.strftime("%Y-%m-%d"), duration_days, excess_cases)


def _onset_offsets(rng: np.random.Generator, count: int, duration_days: int, sigma: float) -> np.ndarray:
    """Draw day offsets from a log-normal curve, truncated to the outbreak's duration."""
    scale = (duration_days - 1) / np.exp(sigma * 2.326)
    offsets: list[np.ndarray] = []
    remaining = count
    while remaining > 0:
        draw = rng.lognormal(0.0, sigma, remaining * 2) * scale
        draw = draw[draw < duration_days][:remaining]
        offsets.append(draw)
        remaining -= len(draw)
    return np.floor(np.concatenate(offsets)).astype(int) if offsets else np.array([], dtype=int)


def make_outbreak_records(rng: np.random.Generator, spec: OutbreakSpec, source: pd.DataFrame,
                          populations: dict[str, int], sigma: float) -> pd.DataFrame:
    """Build added records. Attributes, including institution, come from real records in the same borough."""
    weights = np.array([populations[b] for b in spec.boroughs], dtype=float)
    per_borough = rng.multinomial(spec.excess_cases, weights / weights.sum())
    parts = []
    for borough, count in zip(spec.boroughs, per_borough):
        if count == 0:
            continue
        pool = source[(source["borough"] == borough) & ~source["injected"]]
        picks = pool.iloc[rng.integers(0, len(pool), int(count))].copy()
        offsets = _onset_offsets(rng, int(count), spec.duration_days, sigma)
        picks["admitted"] = pd.Timestamp(spec.start) + pd.to_timedelta(offsets, unit="D")
        parts.append(picks)
    added = pd.concat(parts, ignore_index=True)
    added["injected"] = True
    added["record_id"] = [f"INJ-{spec.outbreak_id}-{i}" for i in range(len(added))]
    return added


def write_ground_truth(specs: list[OutbreakSpec], directory: str | Path, run_id: str) -> Path:
    """Save outbreak details where only the evaluator reads them."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{run_id}.json"
    path.write_text(json.dumps([asdict(s) for s in specs], indent=1), encoding="utf8")
    return path
