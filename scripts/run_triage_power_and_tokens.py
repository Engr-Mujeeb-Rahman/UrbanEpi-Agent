"""Added for the November 2026 review (m9, M4). Power of the 40-versus-40 triage test, and measured triage prompt tokens."""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

from urbanepi.config import load_config
from urbanepi.llm.gemini import client_from_env

rng = np.random.default_rng(20261109)
n, base, sims = 40, 0.70, 2000
power = {}
for true_rate in (0.75, 0.80, 0.85, 0.90, 0.95, 1.0):
    hits = 0
    for _ in range(sims):
        a, b = rng.binomial(n, true_rate), rng.binomial(n, base)
        hits += fisher_exact([[a, n - a], [b, n - b]])[1] < 0.05
    power[str(true_rate)] = round(float(hits) / sims, 3)
print("power", power, flush=True)

cfg = load_config()
client = client_from_env()
counts = []
for path in sorted(glob.glob("experiments/llm_cache/*.json")):
    entry = json.loads(Path(path).read_text(encoding="utf8"))
    if entry["prompt_id"] == "triage_v2" and len(counts) < 30:
        prompt = client.models.count_tokens(model=cfg["llm"]["model"], contents=entry["prompt"]).total_tokens
        output = client.models.count_tokens(model=cfg["llm"]["model"], contents=entry["response"]).total_tokens
        counts.append((prompt, output))
prompt_tokens, output_tokens = np.array(counts).T
report = {"note": __doc__, "power_fisher_two_sided_alpha_0.05": {"per_group": n, "false_alert_confirm_rate": base, "simulations": sims, "power_by_true_alert_confirm_rate": power},
          "triage_tokens": {"prompts_counted": len(counts), "median_prompt_tokens": float(np.median(prompt_tokens)), "median_output_tokens": float(np.median(output_tokens)),
                            "method": "count_tokens on cached triage_v2 prompts and responses"}}
print(json.dumps(report["triage_tokens"]))
Path("experiments/results/final/triage_power_tokens.json").write_text(json.dumps(report, indent=1), encoding="utf8")
