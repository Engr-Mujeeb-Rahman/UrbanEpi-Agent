"""Empirical privacy audit for the paper: the check from the test suite, saved with its value. Passing supports correctness but does not prove privacy."""
import json
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
results = {}
for k, epsilon in enumerate((1.0, 2.0, 4.0)):
    rho = rho_from_epsilon(epsilon, DELTA)
    sigma = sigma_for(rho, 1.0)
    seeds = np.random.SeedSequence([cfg["seed"], 90, k]).spawn(5)
    nodes = [InstitutionNode(str(i), BudgetLedger(1.0), np.random.default_rng(s), BITS) for i, s in enumerate(seeds)]
    stream = MonitoringStream("audit", nodes, sigma, rho, np.random.default_rng(np.random.SeedSequence([cfg["seed"], 91, k])), BITS)
    base = [np.zeros((TRIALS, 1)) for _ in nodes]
    added = [b.copy() for b in base]
    added[0][:, 0] = 1
    without, with_record = stream.release_window(base).ravel(), stream.release_window(added).ravel()
    results[str(epsilon)] = {"claimed": epsilon, "audit_lower_bound": round(empirical_epsilon(without, with_record, DELTA, np.linspace(0, 4 * sigma, 25)), 3), "sigma": round(sigma, 3)}
    print(epsilon, results[str(epsilon)], flush=True)
Path("experiments/results/final/audit.json").write_text(json.dumps({"trials": TRIALS, "institutions": 5, "note": __doc__, "results": results}, indent=1), encoding="utf8")
