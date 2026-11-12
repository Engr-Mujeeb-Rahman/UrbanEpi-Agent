"""Monitoring release: one budget reservation covers every window of a table."""
import numpy as np

from urbanepi.institution.ledger import reserve_all
from urbanepi.institution.node import InstitutionNode
from urbanepi.privacy.secure_agg import aggregate, decode, pairwise_masks


class BudgetRefused(RuntimeError):
    """An institution's ledger refused the query."""


class MonitoringStream:
    """Releases one table per window under a fixed, already charged budget."""

    def __init__(self, stream_id: str, nodes: list[InstitutionNode], sigma: float, rho: float,
                 mask_rng: np.random.Generator, scale_bits: int):
        ledgers = [node.ledger for node in nodes]
        if not reserve_all(ledgers, stream_id, rho):
            raise BudgetRefused(stream_id)
        for ledger in ledgers:
            ledger.commit(stream_id)
        self.nodes, self.sigma, self.mask_rng, self.scale_bits = nodes, sigma, mask_rng, scale_bits

    def release_window(self, counts_by_node: list[np.ndarray]) -> np.ndarray:
        """Return the noised total for one window. Each entry of the input belongs to one institution."""
        parties = len(self.nodes)
        masks = pairwise_masks(self.mask_rng, parties, counts_by_node[0].shape)
        shares = [node.masked_share(counts, self.sigma, parties, mask)
                  for node, counts, mask in zip(self.nodes, counts_by_node, masks)]
        return decode(aggregate(shares), self.scale_bits)


def fast_noisy_table(true_totals: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Experiment shortcut with the same output distribution as the institution path."""
    return true_totals + rng.normal(0.0, sigma, size=true_totals.shape)
