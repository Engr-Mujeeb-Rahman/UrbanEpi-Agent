"""zCDP accounting for the Gaussian mechanism."""
import math
from dataclasses import dataclass


def rho_from_epsilon(epsilon: float, delta: float) -> float:
    """Return the largest rho whose (epsilon, delta) conversion stays within epsilon."""
    log_term = math.log(1 / delta)
    return (math.sqrt(log_term + epsilon) - math.sqrt(log_term)) ** 2


def epsilon_from_rho(rho: float, delta: float) -> float:
    """Convert rho-zCDP to (epsilon, delta)-DP."""
    return rho + 2 * math.sqrt(rho * math.log(1 / delta))


def sigma_for(rho: float, sensitivity: float) -> float:
    """Noise SD that gives a Gaussian release the stated rho."""
    return sensitivity / math.sqrt(2 * rho)


def rho_for(sigma: float, sensitivity: float) -> float:
    """Privacy cost of a Gaussian release."""
    return sensitivity ** 2 / (2 * sigma ** 2)


def sensitivity_for(unit: str, cells_per_record: int, records_per_person: int) -> float:
    """L2 sensitivity of a whole-horizon table, for one record or one person."""
    if unit == "record":
        return math.sqrt(cells_per_record)
    if unit == "person":
        return math.sqrt(cells_per_record * records_per_person)
    raise ValueError(f"Unknown protected unit: {unit}")


@dataclass(frozen=True)
class StreamBudget:
    """Noise and cost of the two monitoring tables for one epsilon and protected unit."""

    rho_total: float
    borough_rho: float
    citywide_rho: float
    borough_sigma: float
    citywide_sigma: float

    @classmethod
    def from_config(cls, cfg: dict, epsilon: float, unit: str) -> "StreamBudget":
        privacy = cfg["privacy"]
        split = privacy["budget_split"]
        if not math.isclose(sum(split.values()), 1.0):
            raise ValueError("Budget split must add up to 1")
        total = rho_from_epsilon(epsilon, privacy["delta"])
        people = privacy["assumed_records_per_person"]
        borough_rho = total * split["borough_table"]
        citywide_rho = total * split["citywide_table"]
        return cls(
            rho_total=total,
            borough_rho=borough_rho,
            citywide_rho=citywide_rho,
            borough_sigma=sigma_for(borough_rho, sensitivity_for(unit, 1, people)),
            citywide_sigma=sigma_for(citywide_rho, sensitivity_for(unit, privacy["citywide_cells_per_record"], people)),
        )
