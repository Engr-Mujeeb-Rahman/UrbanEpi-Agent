"""Empirical privacy audit for the paper: the check from the test suite, saved with its value. Passing supports correctness but does not prove privacy.

Two noise levels are audited for each claimed epsilon:
- "full_budget": sigma set from the whole rho of that epsilon (the level used in the first version of this audit);
- "borough_table": sigma set from 60% of rho, the noise the borough table actually releases.
For the borough table, the epsilon it can claim on its own is rho_b + 2 sqrt(rho_b ln(1/delta)), which is also saved.
The lower bound is log((TPR_low - delta) / FPR_high) over 25 thresholds, with Clopper-Pearson bounds Bonferroni-corrected to 95% overall.
"""
import json
import math
from pathlib import Path

import numpy as np

from urbanepi.aggregator.stream import MonitoringStream
from urbanepi.config import load_config
from urbanepi.institution.ledger import BudgetLedger
from urbanepi.institution.node import InstitutionNode
from urbanepi.privacy.accounting import rho_from_epsilon, sigma_for
from urbanepi.privacy.audit import empirical_epsilon

cfg = load_config()
BITS, DELTA, TRIALS = cfg["privacy"]["noise_scale_bits"], cfg["privacy"]["delta"], 400_000
SHARE = cfg["privacy"]["budget_split"]["borough_table"]
results = {}
for k, epsilon in enumerate((1.0, 2.0, 4.0)):
    rho = rho_from_epsilon(epsilon, DELTA)
    rows = {}
    for level, share in (("full_budget", 1.0), ("borough_table", SHARE)):
        level_rho = rho * share
        sigma = sigma_for(level_rho, 1.0)
        seeds = np.random.SeedSequence([cfg["seed"], 90, k]).spawn(5)
        nodes = [InstitutionNode(str(i), BudgetLedger(1.0), np.random.default_rng(s), BITS) for i, s in enumerate(seeds)]
        stream = MonitoringStream("audit", nodes, sigma, level_rho, np.random.default_rng(np.random.SeedSequence([cfg["seed"], 91, k])), BITS)
        base = [np.zeros((TRIALS, 1)) for _ in nodes]
        added = [b.copy() for b in base]
        added[0][:, 0] = 1
        without, with_record = stream.release_window(base).ravel(), stream.release_window(added).ravel()
        rows[level] = {"rho": round(level_rho, 5), "sigma": round(sigma, 3),
                       "epsilon_of_this_level": round(level_rho + 2 * math.sqrt(level_rho * math.log(1 / DELTA)), 3),
                       "audit_lower_bound": round(empirical_epsilon(without, with_record, DELTA, np.linspace(0, 4 * sigma, 25)), 3)}
    results[str(epsilon)] = {"claimed": epsilon, **rows}
    print(epsilon, results[str(epsilon)], flush=True)
Path("experiments/results/final/audit.json").write_text(json.dumps({"trials": TRIALS, "institutions": 5, "thresholds": 25, "delta": DELTA, "note": __doc__, "results": results}, indent=1), encoding="utf8")
