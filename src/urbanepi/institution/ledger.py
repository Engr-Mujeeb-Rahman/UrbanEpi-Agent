"""Per-institution privacy budget."""
from dataclasses import dataclass, field

_TOLERANCE = 1e-12


@dataclass
class BudgetLedger:
    """Tracks spending at one institution. Reserved budget counts as spent until released."""

    total: float
    spent: float = 0.0
    reserved: dict[str, float] = field(default_factory=dict)

    def remaining(self) -> float:
        return self.total - self.spent - sum(self.reserved.values())

    def reserve(self, query_id: str, rho: float) -> bool:
        if rho < 0:
            raise ValueError("Privacy cost cannot be negative")
        if query_id in self.reserved or rho > self.remaining() + _TOLERANCE:
            return False
        self.reserved[query_id] = rho
        return True

    def commit(self, query_id: str) -> None:
        self.spent += self.reserved.pop(query_id)

    def release(self, query_id: str) -> None:
        self.reserved.pop(query_id)


def reserve_all(ledgers: list[BudgetLedger], query_id: str, rho: float) -> bool:
    """Reserve at every institution, or at none."""
    done: list[BudgetLedger] = []
    for ledger in ledgers:
        if not ledger.reserve(query_id, rho):
            for earlier in done:
                earlier.release(query_id)
            return False
        done.append(ledger)
    return True
