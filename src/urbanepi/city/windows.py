"""Time windows and true-count tables. True counts must stay inside the institution layer."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowGrid:
    """Fixed windows from the study start. A window belongs to the split containing its first day."""

    start: pd.Timestamp
    end: pd.Timestamp
    window_days: int

    @classmethod
    def from_config(cls, cfg: dict, window_days: int | None = None) -> "WindowGrid":
        study = cfg["study"]
        return cls(pd.Timestamp(study["start"]), pd.Timestamp(study["end"]), window_days or study["window_days"])

    @property
    def size(self) -> int:
        return ((self.end - self.start).days + 1) // self.window_days

    def index_of(self, dates: pd.Series) -> np.ndarray:
        return ((dates - self.start).dt.days // self.window_days).to_numpy()

    def first_day(self, window: int) -> pd.Timestamp:
        return self.start + pd.Timedelta(days=window * self.window_days)

    def split_windows(self, first: str, last: str) -> np.ndarray:
        """Return the windows whose first day falls within the given dates."""
        starts = [self.first_day(w) for w in range(self.size)]
        return np.array([w for w, s in enumerate(starts) if pd.Timestamp(first) <= s <= pd.Timestamp(last)])


def count_cells(events: pd.DataFrame, grid: WindowGrid, boroughs: list[str], institutions: list[str]) -> np.ndarray:
    """Count records into an array of shape (windows, boroughs, institutions). Windows cut short are dropped."""
    window = grid.index_of(events["admitted"])
    if not events["borough"].isin(boroughs).all() or not events["institution"].isin(institutions).all():
        raise ValueError("Events contain an unknown borough or institution")
    borough = pd.Categorical(events["borough"], categories=boroughs).codes
    institution = pd.Categorical(events["institution"], categories=institutions).codes
    inside = (window >= 0) & (window < grid.size)
    cells = np.zeros((grid.size, len(boroughs), len(institutions)), dtype=np.int64)
    np.add.at(cells, (window[inside], borough[inside], institution[inside]), 1)
    return cells
