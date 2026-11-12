"""Write UrbanEpi-Q as a spreadsheet for human review."""
import json
from pathlib import Path

import pandas as pd

from urbanepi.agent.vocabulary import Vocabulary
from urbanepi.config import load_config

vocabulary = Vocabulary.from_config(load_config())


def describe(gold: dict) -> str:
    if gold["intent"] == "out_of_scope":
        return "REFUSE"
    if gold.get("needs_clarification"):
        return "ASK FOR CLARIFICATION"
    places = ", ".join(vocabulary.borough_names[d].title() for d in gold["districts"]) if gold["districts"] else "whole city"
    time = gold["time"]
    parts = [f"{gold['intent']} of {' + '.join(gold['syndromes'])}", gold["metric"], places, f"{time['start']} to {time['end']}"]
    if time["bucket"] != "none":
        parts.append(f"per {time['bucket']}")
    if gold.get("group_by"):
        parts.append("by " + " and ".join(gold["group_by"]))
    if gold.get("age_bands"):
        parts.append("ages " + ", ".join(gold["age_bands"]))
    if gold.get("conditions"):
        parts.append("with " + ", ".join(gold["conditions"]))
    if gold["comparison"]["type"] != "none":
        parts.append("compare: " + gold["comparison"]["type"])
    return "; ".join(parts)


items = [json.loads(line) for line in Path("data/benchmark/urbanepi_q.jsonl").read_text(encoding="utf8").splitlines()]
sheet = pd.DataFrame([{"id": i["id"], "split": i["split"], "category": i["category"], "question": i["question"],
                       "correct_plan": describe(i["gold"]), "approved_yes_no": "", "correction_or_comment": ""} for i in items])
sheet.to_csv("data/benchmark/urbanepi_q_review_sheet.csv", index=False, encoding="utf-8-sig")

# A short sheet for a person: every judgment call, plus a sample of template rows across categories.
judgment = sheet[sheet["category"].isin(["ambiguous", "unsafe"])]
templated = sheet[~sheet["category"].isin(["ambiguous", "unsafe"])]
sample = templated.groupby("category", group_keys=False).sample(n=5, random_state=20261106)
short = pd.concat([judgment, sample]).sort_values(["category", "id"])
short.to_csv("data/benchmark/human_review_sample.csv", index=False, encoding="utf-8-sig")
print(len(sheet), len(short), short["category"].value_counts().to_dict())
