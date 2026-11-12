"""Phase 0 inspection of a Synthea CSV sample. Usage: inspect_sample.py <csv_dir>"""
import sys
from pathlib import Path

import pandas as pd

csv_dir = Path(sys.argv[1])
pat = pd.read_csv(csv_dir / "patients.csv", dtype={"ZIP": str})
enc = pd.read_csv(csv_dir / "encounters.csv", usecols=["Id", "START", "PATIENT", "ORGANIZATION", "ENCOUNTERCLASS"])
con = pd.read_csv(csv_dir / "conditions.csv", dtype={"CODE": str})
org = pd.read_csv(csv_dir / "organizations.csv", dtype={"ZIP": str})
enc["START"] = pd.to_datetime(enc["START"], utc=True).dt.tz_localize(None)
con["START"] = pd.to_datetime(con["START"])

print("patients", len(pat), "alive", pat["DEATHDATE"].isna().sum())
print("cities", pat["CITY"].value_counts().head(5).to_dict())
print("county", pat["COUNTY"].value_counts().head(3).to_dict())
print("encounters", len(enc), enc["START"].min(), enc["START"].max())
print("conditions", len(con), con["START"].min(), con["START"].max())
print("organizations", len(org))

z = pat["ZIP"].fillna("MISSING").value_counts()
print("\nZIP: distinct", z.size, "missing", int(pat["ZIP"].isna().sum()), "min/median/max", z.min(), z.median(), z.max())
print(z.head(40).to_dict())
e = len(pat) / z.size
print("uniformity chi2/df", round(((z - e) ** 2 / e).sum() / max(z.size - 1, 1), 2))

print("\nEncounters per year\n", enc["START"].dt.year.value_counts().sort_index().tail(12).to_dict())
print("Enc classes", enc["ENCOUNTERCLASS"].value_counts().to_dict())
print("Orgs used per patient (all years)", enc.groupby("PATIENT")["ORGANIZATION"].nunique().describe().round(2).to_dict())

win = con[(con["START"] >= "2015-01-01") & (con["START"] < "2020-01-01")]
print("\nConditions 2015-2019 top 45")
print(win.groupby(["CODE", "DESCRIPTION"]).size().sort_values(ascending=False).head(45).to_string())

pattern = r"(?i)sinusitis|bronchitis|pharyngitis|sore throat|otitis|cough|fever|influenza|pneumonia|covid|respiratory|gastro|diarrh|vomit|nausea|norovirus|food"
acute = win[win["DESCRIPTION"].str.contains(pattern, regex=True)]
print("\nSyndrome-candidate conditions")
print(acute.groupby(["CODE", "DESCRIPTION"]).size().sort_values(ascending=False).to_string())

py = len(pat) * 5
print("\nCandidate events per person-year", round(len(acute) / py, 3))
print("By month", acute["START"].dt.month.value_counts().sort_index().to_dict())
print("By weekday (Mon=0)", acute["START"].dt.day_of_week.value_counts().sort_index().to_dict())
print("By year", acute["START"].dt.year.value_counts().sort_index().to_dict())
per = acute.groupby("PATIENT").agg(n=("CODE", "size"))
per_year = acute.groupby(["PATIENT", acute["START"].dt.year]).size()
print("Max per patient-year quantiles", per_year.quantile([.5, .9, .99, 1]).to_dict())
print("Share of patients with any", round(len(per) / len(pat), 3))

m = acute.merge(pat[["Id", "ZIP"]], left_on="PATIENT", right_on="Id")
wk = m.groupby(["ZIP", pd.Grouper(key="START", freq="3D")]).size()
print("Events per ZIP-3day (non-empty mean, this sample)", round(wk.mean(), 2))
print("Scaled to 200k: mean events per ZIP per 3 days", round(len(acute) * 200 / z.size / (5 * 365 / 3), 2))

from scipy.stats import chisquare

resp = acute[~acute["DESCRIPTION"].str.contains("Chronic|diarrhea", regex=True)]
print("\nAcute respiratory events", len(resp), "per person-year", round(len(resp) / py, 3))
print("Month chi2 p", round(chisquare(resp["START"].dt.month.value_counts().sort_index()).pvalue, 3))
print("Weekday chi2 p", round(chisquare(resp["START"].dt.day_of_week.value_counts().sort_index()).pvalue, 3))
print("Patient-year counts quantiles", resp.groupby(["PATIENT", resp["START"].dt.year]).size().quantile([.5, .9, .99, 1]).to_dict())
print("Horizon cap check: share of patient-years above L=3 / L=4",
      round((resp.groupby(["PATIENT", resp["START"].dt.year]).size() > 3).mean(), 4),
      round((resp.groupby(["PATIENT", resp["START"].dt.year]).size() > 4).mean(), 4))
