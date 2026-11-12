"""Profile the January 2024 to April 2025 Mexico City study period."""
import math
import zipfile

import pandas as pd

ZIP = "data/raw/mexico/datos_abiertos_influenza_covid19_20250430.zip"
archive = zipfile.ZipFile(ZIP)
with archive.open("COVID19MEXICO.csv") as handle:
    df = pd.read_csv(handle, dtype=str, encoding="latin-1")
df = df[df["ENTIDAD_RES"] == "09"].copy()
df["adm"] = pd.to_datetime(df["FECHA_INGRESO"])
df["sym"] = pd.to_datetime(df["FECHA_SINTOMAS"])
print("records", len(df), "admission range", df["adm"].min().date(), df["adm"].max().date())
print("onset-to-admission days", (df["adm"] - df["sym"]).dt.days.describe().round(2).to_dict())
print("weekly records, last 8 weeks", df.set_index("adm").resample("W").size().tail(8).to_dict())

flu, cov = df["CLASIFICACION_FINAL_FLU"], df["CLASIFICACION_FINAL_COVID"]
print("\nCOVID by FLU classification\n", pd.crosstab(cov, flu).to_string())
print("\nPCR codes", df["RESULTADO_PCR"].value_counts().to_dict())
print("Pneumonia", df["NEUMONIA"].value_counts().to_dict())
print("Sector", df["SECTOR"].value_counts().to_dict())
print("Monthly flu confirmed", df[flu == "3"].groupby(df["adm"].dt.to_period("M")).size().to_dict())
print("Monthly covid confirmed", df[cov == "3"].groupby(df["adm"].dt.to_period("M")).size().to_dict())

splits = {"burn-in": ("2024-01-01", "2024-03-31"), "validation": ("2024-04-01", "2024-10-31"), "test": ("2024-11-01", "2025-04-15")}
leaves = {"ALL": pd.Series(True, index=df.index), "FLU": flu == "3", "COVID": cov == "3", "OTHER": (flu != "3") & (cov != "3")}
print()
for name, (a, b) in splits.items():
    days = (pd.Timestamp(b) - pd.Timestamp(a)).days + 1
    part = df[(df["adm"] >= a) & (df["adm"] <= b)]
    print(name, a, b, "days", days, "windows", math.ceil(days / 3), "records", len(part))
    for leaf, mask in leaves.items():
        p = part[mask.loc[part.index]]
        idx = pd.date_range(a, b, freq="3D")
        cells = p.groupby(["MUNICIPIO_RES", pd.Grouper(key="adm", freq="3D", origin=pd.Timestamp(a))]).size().unstack(fill_value=0).reindex(columns=idx, fill_value=0)
        cells = cells.reindex(sorted(df["MUNICIPIO_RES"].unique()), fill_value=0)
        v = cells.to_numpy().ravel()
        print(f"  {leaf}: mean {v.mean():.1f}, median {pd.Series(v).median():.0f}, p90 {pd.Series(v).quantile(.9):.0f}, max {v.max()}, zero-share {(v == 0).mean():.2f}")
    print("  institutions", part["SECTOR"].value_counts().head(6).to_dict())
per_sector = df.groupby("MUNICIPIO_RES")["SECTOR"].nunique()
print("\nInstitutions serving each borough, min/max", per_sector.min(), per_sector.max())

d = math.log(1e6)
T, K = math.ceil(471 / 3), 5
print("\neps rho_m record_dist person_L4_dist person_L4_silo perwindow_L1_dist perwindow_silo T=", T, "K=", K)
for e in (0.5, 1, 2, 4, 8):
    x = (-2 * math.sqrt(d) + math.sqrt(4 * d + 4 * e)) / 2
    r = 0.7 * x * x
    s = lambda delta2: math.sqrt(delta2 / (2 * r))
    print(e, round(r, 4), round(s(1), 1), round(s(4), 1), round(s(4) * math.sqrt(K), 1), round(s(T), 1), round(s(T) * math.sqrt(K), 1))
x = (-2 * math.sqrt(d) + math.sqrt(4 * d + 8)) / 2
print("adhoc eps2 sigma N_q=20", round(math.sqrt(20 / (2 * .2 * x * x)), 1))
