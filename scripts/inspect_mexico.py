"""Check a Mexico open-data ZIP for UrbanEpi-Agent. Usage: inspect_mexico.py <zip> [state_code]"""
import sys
import zipfile

import pandas as pd

path, state = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "09")
archive = zipfile.ZipFile(path)
member = next(i for i in archive.infolist() if i.filename.lower().endswith(".csv"))
with archive.open(member) as handle:
    columns = [c.strip('"') for c in handle.readline().decode("latin-1").strip().split(",")]
print(path, member.filename, member.file_size)
print("Columns", len(columns), columns)

extra = [c for c in columns if any(k in c for k in ("RESULTADO", "CLASIFICACION", "TOMA_MUESTRA", "FLU", "INFLU"))]
keep = ["ID_REGISTRO", "ORIGEN", "SECTOR", "ENTIDAD_UM", "ENTIDAD_RES", "MUNICIPIO_RES", "TIPO_PACIENTE",
        "FECHA_INGRESO", "FECHA_SINTOMAS", "NEUMONIA", "EDAD", "FECHA_ACTUALIZACION"] + extra
parts, total, national_unique = [], 0, set()
with archive.open(member) as handle:
    for chunk in pd.read_csv(handle, usecols=keep, dtype=str, chunksize=1_000_000, encoding="latin-1"):
        total += len(chunk)
        parts.append(chunk[chunk["ENTIDAD_RES"] == state])
df = pd.concat(parts, ignore_index=True)
df["FECHA_SINTOMAS"] = pd.to_datetime(df["FECHA_SINTOMAS"], errors="coerce")
df["FECHA_INGRESO"] = pd.to_datetime(df["FECHA_INGRESO"], errors="coerce")

print(f"\nNational rows {total:,}; state {state} rows {len(df):,}")
print("Updated", df["FECHA_ACTUALIZACION"].unique()[:3])
print("Symptom dates", df["FECHA_SINTOMAS"].min(), df["FECHA_SINTOMAS"].max(), "missing", df["FECHA_SINTOMAS"].isna().sum())
print("Rows per ID: unique IDs", df["ID_REGISTRO"].nunique(), "max repeats", df["ID_REGISTRO"].value_counts().max())
print("Treated in same state", round((df["ENTIDAD_UM"] == state).mean(), 3))

print("\nBy year", df["FECHA_SINTOMAS"].dt.year.value_counts().sort_index().to_dict())
print("\nBy municipality", df["MUNICIPIO_RES"].value_counts().to_dict())
print("\nBy sector", df["SECTOR"].value_counts().to_dict())
print("Origin", df["ORIGEN"].value_counts().to_dict())
print("Patient type", df["TIPO_PACIENTE"].value_counts().to_dict())
for col in extra:
    print(col, df[col].value_counts().head(10).to_dict())

for label, part in df.groupby(df["FECHA_SINTOMAS"].dt.year):
    if len(part) < 1000:
        continue
    span = (part["FECHA_SINTOMAS"].max() - part["FECHA_SINTOMAS"].min()).days + 1
    cells = part.groupby(["MUNICIPIO_RES", pd.Grouper(key="FECHA_SINTOMAS", freq="3D")]).size()
    print(f"{label}: per borough per 3 days mean {len(part) / df['MUNICIPIO_RES'].nunique() / (span / 3):.1f}, "
          f"median non-empty {cells.median():.0f}, p99 {cells.quantile(.99):.0f}; span {span} days")
monthly = df.groupby(df["FECHA_SINTOMAS"].dt.to_period("M")).size()
print("\nMonthly\n", monthly.to_string())
print("\nSector by borough share (top sector), median", round(
    df.groupby("MUNICIPIO_RES")["SECTOR"].agg(lambda s: s.value_counts(normalize=True).iloc[0]).median(), 3))
