"""Build borough names, census populations, and adjacency from public sources.

Usage: build_public_metadata.py <boroughs_geojson> <catalog_zip>
Boundaries: PhantomInsights/mexico-geojson (MIT), from CONABIO shapefiles.
Populations: INEGI Censo 2020. Checked against the official city total.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd
from shapely.geometry import shape

CITY_TOTAL_2020 = 9_209_944
POPULATION_2020 = {
    "002": 432_205, "003": 614_447, "004": 217_686, "005": 1_173_351, "006": 404_695, "007": 1_835_486,
    "008": 247_622, "009": 152_685, "010": 759_137, "011": 392_313, "012": 699_928, "013": 442_178,
    "014": 434_153, "015": 545_884, "016": 414_470, "017": 443_704,
}
TOUCH_DEGREES = 1e-4
MIN_SHARED_AREA = 1e-7

geojson_path, catalog_zip = sys.argv[1:3]
if sum(POPULATION_2020.values()) != CITY_TOTAL_2020:
    raise ValueError("Borough populations do not add up to the city total")

with zipfile.ZipFile(catalog_zip) as archive:
    workbook = openpyxl.load_workbook(io.BytesIO(archive.read("201128 Catalogos.xlsx")), read_only=True)
sheet = next(workbook[n] for n in workbook.sheetnames if "MUNICIPIOS" in n.upper())
names = {str(r[0]): str(r[1]).strip() for r in sheet.iter_rows(values_only=True) if str(r[2]) == "09" and str(r[0]) != "999"}

features = json.loads(Path(geojson_path).read_bytes().decode("utf8", errors="replace"))["features"]
shapes = {f["properties"]["CVE_MUN"]: shape(f["geometry"]).buffer(0) for f in features}
if set(shapes) != set(POPULATION_2020) or set(names) != set(POPULATION_2020):
    raise ValueError("Borough codes differ across sources")

adjacency = {}
for code, area in shapes.items():
    grown = area.buffer(TOUCH_DEGREES)
    adjacency[code] = sorted(
        other for other, target in shapes.items()
        if other != code and grown.intersection(target.buffer(TOUCH_DEGREES)).area > MIN_SHARED_AREA
    )

out = Path("data/public")
pd.DataFrame(
    [{"code": c, "name": names[c], "population_2020": POPULATION_2020[c]} for c in sorted(names)]
).to_csv(out / "boroughs.csv", index=False, encoding="utf8")
(out / "adjacency.json").write_text(json.dumps(dict(sorted(adjacency.items())), indent=1), encoding="utf8")
print(pd.read_csv(out / "boroughs.csv", dtype=str).to_string())
print(json.dumps(dict(sorted(adjacency.items()))))
