"""Measurements added after review: allocation audit, institution dominance in reference alerts, and computed payload. No LLM calls."""
import json
from pathlib import Path

import numpy as np

source = Path("scripts/run_e2.py").read_text(encoding="utf8")
exec(source[:source.index("results = {}\nseeds =")])

budget = StreamBudget.from_config(cfg, 2, "record")
settings = {**settings, "protected_unit": "record"}
rho = budget.rho_total * cfg["privacy"]["budget_split"]["adhoc"] / settings["planned_adhoc_queries"]
released = released_for(budget, np.random.default_rng(5))
same, differ, free, one_group, details = 0, 0, 0, 0, []
for question, category, plan in plans:
    full = decompose(plan, vocabulary, grid, released, rho, settings)
    flat = decompose(plan, vocabulary, grid, released, rho, settings, uniform=True)
    if not full.subqueries:
        free += 1
        continue
    if len(full.subqueries) == 1:
        one_group += 1
    a, b = [q.sigma for q in full.subqueries], [q.sigma for q in flat.subqueries]
    if len(a) == len(b) and np.allclose(a, b, rtol=1e-9):
        same += 1
    else:
        differ += 1
        details.append({"id": question, "category": category, "groups": len(a), "weights": [round(q.weight, 2) for q in full.subqueries], "sigma_full": [round(x, 2) for x in a], "sigma_uniform": [round(x, 2) for x in b]})
allocation = {"questions": len(plans), "reused_with_no_fresh_noise": free, "fresh_with_identical_noise": same, "fresh_with_one_group": one_group, "fresh_with_different_noise": differ, "different": details}
print(json.dumps(allocation)[:900], flush=True)

e8 = json.loads(Path("experiments/results/final/e8.json").read_text(encoding="utf8"))["results"]["reference_alerts"]
names = list(cfg["institutions"].values())
cells = count_cells(events, grid, vocabulary.codes, list(cfg["institutions"].values())).astype(float)
B, g = cfg["gate"]["baseline_windows"], cfg["gate"]["guard_windows"]
rows = []
for alert in e8:
    t = int(grid.index_of(__import__("pandas").Series([__import__("pandas").Timestamp(alert["window_start"])]))[0])
    for code in alert["boroughs"]:
        d = vocabulary.codes.index(code)
        excess = cells[t, d] - cells[t - g - B:t - g, d].mean(axis=0)
        positive = np.clip(excess, 0, None)
        top = int(positive.argmax())
        rows.append({"window_start": alert["window_start"], "borough": code, "excess": round(float(excess.sum()), 1), "top_institution": names[top], "top_share_of_positive_excess": round(float(positive[top] / positive.sum()), 3) if positive.sum() else None})
dominated = sum(r["top_share_of_positive_excess"] is not None and r["top_share_of_positive_excess"] > 0.8 for r in rows)
payload_cells = len(vocabulary.codes) + 2
report = {"note": __doc__, "allocation": allocation, "reference_alert_cells": rows, "cells_with_one_institution_over_80pct": dominated,
          "payload": {"cells_per_window": payload_cells, "bytes_per_cell": 8, "bytes_per_institution_per_window": payload_cells * 8, "note": "Computed, excluding protocol overhead and key exchange, which are not simulated."}}
Path("experiments/results/final/review_checks.json").write_text(json.dumps(report, indent=1), encoding="utf8")
print(json.dumps({k: report[k] for k in ("reference_alert_cells", "cells_with_one_institution_over_80pct", "payload")}))
