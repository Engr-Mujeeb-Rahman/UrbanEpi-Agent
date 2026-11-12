"""One institution's noise and masking step."""
from dataclasses import dataclass

import numpy as np

from urbanepi.institution.ledger import BudgetLedger
from urbanepi.privacy.secure_agg import add_mask, encode


@dataclass
class InstitutionNode:
    """Holds an institution's ledger and its own noise generator."""

    name: str
    ledger: BudgetLedger
    rng: np.random.Generator
    scale_bits: int

    def masked_share(self, counts: np.ndarray, sigma: float, parties: int, mask: np.ndarray) -> np.ndarray:
        """Add this institution's noise share to every cell, then hide the result."""
        noisy = counts + self.rng.normal(0.0, sigma / np.sqrt(parties), size=counts.shape)
        return add_mask(encode(noisy, self.scale_bits), mask)
