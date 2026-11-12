"""Scores the second annotator's check of 30 test gold plans. Writes a clean UTF-8 copy, because the returned sheet mixes encodings."""
import csv
import io
import json
from math import sqrt
from pathlib import Path

raw = Path("data/benchmark/second_annotator_sheet.csv").read_bytes()
lines = []
for line in raw.split(b"\n"):
    try:
        lines.append(line.decode("utf-8"))
    except UnicodeDecodeError:
        lines.append(line.decode("cp1252", errors="replace"))
rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
with open("data/benchmark/second_annotator_sheet_utf8.csv", "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

agree = sum(r["annotator_agrees_yes_no"].strip().lower() == "yes" for r in rows)
n = len(rows)
z, p = 1.96, agree / n
centre = (p + z * z / (2 * n)) / (1 + z * z / n)
half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
disagreements = {r["id"]: r["notes"] for r in rows if r["annotator_agrees_yes_no"].strip().lower() != "yes"}
adjudication = {
    "Q063": "Gold-label error: intent should be trend. The clarification status, which is what the question tests, is unchanged.",
    "Q097": "Not an error under the schema: a plan with several syndromes already returns one output per syndrome, and the plan language has no syndrome grouping.",
    "Q098": "Same as Q097.",
}
report = {"rows": n, "agree": agree, "agreement": round(p, 3), "wilson_95": [round(centre - half, 3), round(centre + half, 3)],
          "disagreements": disagreements, "adjudication": adjudication, "gold_errors_after_adjudication": 1}
Path("experiments/results/final/second_annotator.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: report[k] for k in ("rows", "agree", "agreement", "wilson_95", "gold_errors_after_adjudication")}))
