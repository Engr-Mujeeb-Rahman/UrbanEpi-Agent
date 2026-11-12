"""An institution's local knowledge graph. Only registered templates can run, and results stay in this layer."""
import json
import tempfile
from pathlib import Path

import kuzu
import pandas as pd

_SCHEMA = [
    "CREATE NODE TABLE CaseRecord(case_id STRING, admitted DATE, age_band STRING, PRIMARY KEY(case_id))",
    "CREATE NODE TABLE District(code STRING, PRIMARY KEY(code))",
    "CREATE NODE TABLE Pathogen(name STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE Syndrome(name STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE ConditionType(name STRING, PRIMARY KEY(name))",
    "CREATE REL TABLE RESIDES_IN(FROM CaseRecord TO District)",
    "CREATE REL TABLE CONFIRMED_AS(FROM CaseRecord TO Pathogen)",
    "CREATE REL TABLE HAS_CONDITION(FROM CaseRecord TO ConditionType)",
    "CREATE REL TABLE COVERS(FROM Syndrome TO Pathogen)",
    "CREATE REL TABLE INCLUDES(FROM Syndrome TO Syndrome)",
]
_FILTERS = "WHERE d.code IN $districts AND c.admitted >= $from_date AND c.admitted < $to_date AND c.age_band IN $age_bands"
_ALL_CASES = "MATCH (c:CaseRecord)-[:RESIDES_IN]->(d:District)"
_BY_SYNDROME = (_ALL_CASES + ", (c)-[:CONFIRMED_AS]->(:Pathogen)<-[:COVERS]-(:Syndrome)"
                "<-[:INCLUDES*0..3]-(s:Syndrome {name: $syndrome})")
_CONDITION = ", (c)-[:HAS_CONDITION]->(:ConditionType {name: $condition})"
_COUNT = "RETURN count(DISTINCT c) AS cases"
TEMPLATES = {
    "all_cases": {"query": f"{_ALL_CASES} {_FILTERS} {_COUNT}", "params": {"districts", "from_date", "to_date", "age_bands"}},
    "all_cases_condition": {"query": f"{_ALL_CASES}{_CONDITION} {_FILTERS} {_COUNT}", "params": {"districts", "from_date", "to_date", "age_bands", "condition"}},
    "by_syndrome": {"query": f"{_BY_SYNDROME} {_FILTERS} {_COUNT}", "params": {"districts", "from_date", "to_date", "age_bands", "syndrome"}},
    "by_syndrome_condition": {"query": f"{_BY_SYNDROME}{_CONDITION} {_FILTERS} {_COUNT}", "params": {"districts", "from_date", "to_date", "age_bands", "syndrome", "condition"}},
}


class TemplateRejected(ValueError):
    """The template isn't registered, or its parameters don't match."""


class KuzuGraphStore:
    """Embedded graph with one public layer and this institution's private records."""

    def __init__(self, events: pd.DataFrame, codes: list[str], ontology: dict[str, dict], conditions: list[str]):
        self._folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        folder = Path(self._folder.name)
        self._db = kuzu.Database(str(folder / "graph"))
        self._conn = kuzu.Connection(self._db)
        for statement in _SCHEMA:
            self._conn.execute(statement)
        ids = events["record_id"]
        confirmed = events[["record_id", "pathogens"]].explode("pathogens").dropna()
        leaves = [name for name, entry in ontology.items() if entry["pcr_codes"]]
        tables = {
            "CaseRecord": pd.DataFrame({"case_id": ids, "admitted": events["admitted"].dt.date, "age_band": events["age_band"]}),
            "District": pd.DataFrame({"code": codes}),
            "Pathogen": pd.DataFrame({"name": leaves}),
            "Syndrome": pd.DataFrame({"name": list(ontology)}),
            "ConditionType": pd.DataFrame({"name": conditions}),
            "RESIDES_IN": pd.DataFrame({"from": ids, "to": events["borough"]}),
            "CONFIRMED_AS": pd.DataFrame({"from": confirmed["record_id"], "to": confirmed["pathogens"]}),
            "HAS_CONDITION": pd.concat([pd.DataFrame({"from": ids[events[name]], "to": name}) for name in conditions]),
            "COVERS": pd.DataFrame({"from": leaves, "to": leaves}),
            "INCLUDES": pd.DataFrame([{"from": s, "to": c} for s, entry in ontology.items() for c in entry["includes"]]),
        }
        for name, table in tables.items():
            if len(table):
                path = folder / f"{name}.parquet"
                table.astype({c: str for c in table.columns if c != "admitted"}).to_parquet(path, index=False)
                self._conn.execute(f"COPY {name} FROM '{path.as_posix()}'")

    def run_template(self, template_id: str, params: dict) -> int:
        """Run a registered template and return its count."""
        template = TEMPLATES.get(template_id)
        if template is None:
            raise TemplateRejected(f"Unregistered template: {template_id}")
        if set(params) != template["params"]:
            raise TemplateRejected(f"Wrong parameters for {template_id}: {sorted(params)}")
        return int(self._conn.execute(template["query"], params).get_next()[0])
