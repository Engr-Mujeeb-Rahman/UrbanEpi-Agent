"""Everything already released. Reusing it is post-processing, so it costs no privacy budget."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReleasedTable:
    """Monitoring tables plus ad-hoc cells, each with its noise variance."""

    boroughs: list[str]
    windows: int
    borough_sigma: float
    citywide_sigma: float
    borough_table: np.ndarray = None
    citywide_table: dict[str, np.ndarray] = field(default_factory=dict)
    released_windows: int = 0
    adhoc: dict[tuple, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        self.borough_table = np.full((self.windows, len(self.boroughs)), np.nan)

    def add_window(self, borough_values: np.ndarray, citywide_values: dict[str, float]) -> None:
        window = self.released_windows
        self.borough_table[window] = borough_values
        for name, value in citywide_values.items():
            self.citywide_table.setdefault(name, np.full(self.windows, np.nan))[window] = value
        self.released_windows += 1

    def resp_total(self, districts: tuple[str, ...], first: int, last: int) -> tuple[float, float] | None:
        """Sum released borough cells for windows first to last, inclusive."""
        if last >= self.released_windows:
            return None
        columns = [self.boroughs.index(d) for d in districts]
        cells = (last - first + 1) * len(columns)
        return float(self.borough_table[first:last + 1][:, columns].sum()), cells * self.borough_sigma ** 2

    def citywide_total(self, name: str, first: int, last: int) -> tuple[float, float] | None:
        """Sum released citywide cells for windows first to last, inclusive."""
        if last >= self.released_windows or name not in self.citywide_table:
            return None
        return float(self.citywide_table[name][first:last + 1].sum()), (last - first + 1) * self.citywide_sigma ** 2
