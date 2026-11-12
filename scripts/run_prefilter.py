"""Added for the supervisor's comments (34). Measures the rule-based pre-filter that runs before any LLM call.

The filter was written from the dev questions only; this script records its checksum and runs it once on the test set.
"""
import hashlib
import json
from pathlib import Path

from urbanepi.agent.prefilter import blocked
from urbanepi.evaluation.detection import wilson

rows = [json.loads(line) for line in Path("data/benchmark/urbanepi_q.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
checksum = hashlib.sha256(Path("src/urbanepi/agent/prefilter.py").read_bytes()).hexdigest()
results = {}
for split in ("dev", "test"):
    chosen = [r for r in rows if r["split"] == split]
    unsafe = [r for r in chosen if r["category"] == "unsafe"]
    other = [r for r in chosen if r["category"] != "unsafe"]
    caught = [r["id"] for r in unsafe if blocked(r["question"])]
    false_blocks = [{"id": r["id"], "category": r["category"], "pattern": blocked(r["question"])} for r in other if blocked(r["question"])]
    results[split] = {"unsafe": len(unsafe), "unsafe_blocked": len(caught), "unsafe_blocked_ci": wilson(len(caught), len(unsafe)),
                      "other": len(other), "other_blocked": len(false_blocks), "other_blocked_ci": wilson(len(false_blocks), len(other)),
                      "missed_unsafe": [r["id"] for r in unsafe if r["id"] not in caught], "false_blocks": false_blocks}
    print(split, json.dumps(results[split]), flush=True)
Path("experiments/results/final/prefilter.json").write_text(json.dumps({"note": __doc__, "prefilter_sha256": checksum, "results": results}, indent=1), encoding="utf8")
