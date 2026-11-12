"""Build the events table and its report from the configured source file."""
import json
from pathlib import Path

import pandas as pd

from urbanepi.city.etl import clean_records, read_state_records
from urbanepi.city.windows import WindowGrid
from urbanepi.config import load_config

cfg = load_config()
events, report = clean_records(read_state_records(cfg), cfg)

grid = WindowGrid.from_config(cfg)
report["windows"] = grid.size
window = grid.index_of(events["admitted"])
report["splits"] = {}
for name, (first, last) in cfg["study"]["splits"].items():
    windows = grid.split_windows(first, last)
    report["splits"][name] = {
        "windows": int(len(windows)),
        "first_day": str(grid.first_day(int(windows[0])).date()),
        "last_day": str((grid.first_day(int(windows[-1] + 1)) - pd.Timedelta(days=1)).date()),
        "records": int(((window >= windows[0]) & (window <= windows[-1])).sum()),
    }
report["institutions"] = events["institution"].value_counts().to_dict()
report["flu_confirmed"] = int(events["flu_confirmed"].sum())
report["covid_confirmed"] = int(events["covid_confirmed"].sum())
report["both_confirmed"] = int((events["flu_confirmed"] & events["covid_confirmed"]).sum())

Path(cfg["data"]["events_path"]).parent.mkdir(parents=True, exist_ok=True)
events.to_parquet(cfg["data"]["events_path"], index=False)
Path(cfg["data"]["report_path"]).write_text(json.dumps(report, indent=1), encoding="utf8")
print(json.dumps(report, indent=1))
