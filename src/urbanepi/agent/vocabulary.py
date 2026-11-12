"""Public vocabulary: the only values a query plan may use."""
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Vocabulary:
    borough_names: dict[str, str]
    populations: dict[str, int]
    ontology: dict[str, dict]
    age_bands: tuple[str, ...]
    conditions: tuple[str, ...]
    study_start: pd.Timestamp
    study_end: pd.Timestamp

    @classmethod
    def from_config(cls, cfg: dict) -> "Vocabulary":
        table = pd.read_csv(cfg["data"]["boroughs_path"], dtype={"code": str})
        ontology = json.loads(Path(cfg["data"]["ontology_path"]).read_text(encoding="utf8"))["syndromes"]
        return cls(
            borough_names=dict(zip(table["code"], table["name"])),
            populations=dict(zip(table["code"], table["population_2020"].astype(int))),
            ontology=ontology,
            age_bands=tuple(cfg["age_bands"]["labels"]),
            conditions=tuple(cfg["conditions"]),
            study_start=pd.Timestamp(cfg["study"]["start"]),
            study_end=pd.Timestamp(cfg["study"]["end"]),
        )

    @property
    def codes(self) -> list[str]:
        return sorted(self.borough_names)

    @property
    def syndromes(self) -> list[str]:
        return sorted(self.ontology)

    def describe(self) -> str:
        """Public schema text for the parsing prompt."""
        return json.dumps({
            "syndromes": {name: entry["aliases"] for name, entry in self.ontology.items()},
            "districts": self.borough_names,
            "age_bands": list(self.age_bands),
            "conditions": list(self.conditions),
            "study_period": [str(self.study_start.date()), str(self.study_end.date())],
        }, ensure_ascii=False, indent=1)
