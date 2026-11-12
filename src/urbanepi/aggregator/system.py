"""Assembles institutions, monitoring, and ad-hoc release."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from urbanepi.agent.pb_qd import Atom, PlanResult
from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.aggregator.released import ReleasedTable
from urbanepi.aggregator.stream import MonitoringStream
from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.institution.graph import KuzuGraphStore
from urbanepi.institution.ledger import BudgetLedger, reserve_all
from urbanepi.institution.node import InstitutionNode
from urbanepi.privacy.accounting import StreamBudget
from urbanepi.privacy.secure_agg import aggregate, decode, pairwise_masks


def _template_for(atom: Atom, grid: WindowGrid, age_bands: list[str]) -> tuple[str, dict]:
    """Fill a registered template from an atom's public fields."""
    params = {
        "districts": list(atom.districts),
        "from_date": grid.first_day(atom.first_window).date(),
        "to_date": grid.first_day(atom.last_window + 1).date(),
        "age_bands": list(atom.age_bands or age_bands),
    }
    template = "all_cases"
    if atom.syndrome != "RESP":
        template, params["syndrome"] = "by_syndrome", atom.syndrome
    if atom.condition:
        template, params["condition"] = f"{template}_condition", atom.condition
    return template, params


@dataclass
class AdhocFederation:
    """Sends fresh sub-queries to every institution and stores the released totals."""

    nodes: list[InstitutionNode]
    graphs: list[KuzuGraphStore]
    grid: WindowGrid
    age_bands: list[str]
    mask_rng: np.random.Generator
    scale_bits: int
    query_counter: int = 0

    def release(self, result: PlanResult, released: ReleasedTable) -> bool:
        """Reserve at all institutions or none, then run, mask, and add up each sub-query."""
        self.query_counter += 1
        query_id = f"adhoc-{self.query_counter}"
        ledgers = [node.ledger for node in self.nodes]
        if not reserve_all(ledgers, query_id, result.rho):
            return False
        for ledger in ledgers:
            ledger.commit(query_id)
        parties = len(self.nodes)
        for query in result.subqueries:
            per_node = []
            for graph in self.graphs:
                counts = [graph.run_template(*_template_for(a, self.grid, self.age_bands)) for a in query.atoms]
                per_node.append(np.array(counts, dtype=float))
            masks = pairwise_masks(self.mask_rng, parties, per_node[0].shape)
            shares = [node.masked_share(counts, query.sigma, parties, mask)
                      for node, counts, mask in zip(self.nodes, per_node, masks)]
            totals = decode(aggregate(shares), self.scale_bits)
            for atom, value in zip(query.atoms, totals):
                released.adhoc[atom] = (float(value), query.sigma ** 2)
        return True


@dataclass
class System:
    released: ReleasedTable
    federation: AdhocFederation
    nodes: list[InstitutionNode]
    budget: StreamBudget


def build_system(cfg: dict, events: pd.DataFrame, epsilon: float, unit: str, seed_key: int) -> System:
    """Build one graph per institution and charge the monitoring budget."""
    vocabulary = Vocabulary.from_config(cfg)
    institutions = list(cfg["institutions"].values())
    grid = WindowGrid.from_config(cfg)
    budget = StreamBudget.from_config(cfg, epsilon, unit)
    scale_bits = cfg["privacy"]["noise_scale_bits"]
    seeds = np.random.SeedSequence([cfg["seed"], seed_key]).spawn(len(institutions) + 3)
    nodes = [InstitutionNode(name, BudgetLedger(budget.rho_total), np.random.default_rng(seeds[i]), scale_bits)
             for i, name in enumerate(institutions)]
    graphs = [KuzuGraphStore(events[events["institution"] == name], vocabulary.codes, vocabulary.ontology, list(vocabulary.conditions))
              for name in institutions]
    released = ReleasedTable(vocabulary.codes, grid.size, budget.borough_sigma, budget.citywide_sigma)
    federation = AdhocFederation(nodes, graphs, grid, list(vocabulary.age_bands), np.random.default_rng(seeds[-1]), scale_bits)
    return System(released, federation, nodes, budget)


def run_monitoring(system: System, cfg: dict, events: pd.DataFrame, through_window: int, seed_key: int) -> None:
    """Release both monitoring tables, window by window, through the institution path."""
    grid = system.federation.grid
    institutions = [node.name for node in system.nodes]
    boroughs = system.released.boroughs
    scale_bits = cfg["privacy"]["noise_scale_bits"]
    budget = system.budget
    seeds = np.random.SeedSequence([cfg["seed"], seed_key, 7]).spawn(2)
    borough_stream = MonitoringStream("borough_table", system.nodes, budget.borough_sigma, budget.borough_rho, np.random.default_rng(seeds[0]), scale_bits)
    city_stream = MonitoringStream("citywide_table", system.nodes, budget.citywide_sigma, budget.citywide_rho, np.random.default_rng(seeds[1]), scale_bits)
    cells = count_cells(events, grid, boroughs, institutions)
    window = grid.index_of(events["admitted"])
    position = pd.Categorical(events["institution"], categories=institutions).codes
    city = {flag: np.zeros((grid.size, len(institutions))) for flag in ("flu_confirmed", "covid_confirmed")}
    for flag, table in city.items():
        chosen = events[flag].to_numpy()
        np.add.at(table, (window[chosen], position[chosen]), 1)
    for t in range(through_window + 1):
        values = borough_stream.release_window([cells[t, :, i].astype(float) for i in range(len(institutions))])
        by_node = [np.array([city["flu_confirmed"][t, i], city["covid_confirmed"][t, i]]) for i in range(len(institutions))]
        totals = city_stream.release_window(by_node)
        system.released.add_window(values, {"FLU": float(totals[0]), "COVID": float(totals[1])})

