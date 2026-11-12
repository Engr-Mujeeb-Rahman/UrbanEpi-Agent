"""Check whether changes in which institutions report could look like outbreaks."""
import pandas as pd

from urbanepi.city.windows import WindowGrid, count_cells
from urbanepi.config import load_config

cfg = load_config()
events = pd.read_parquet(cfg["data"]["events_path"])
boroughs_table = pd.read_csv(cfg["data"]["boroughs_path"], dtype=str)
boroughs = boroughs_table["code"].tolist()
names = dict(zip(boroughs_table["code"], boroughs_table["name"]))
institutions = list(cfg["institutions"].values())
grid = WindowGrid.from_config(cfg)
cells = count_cells(events, grid, boroughs, institutions)

print("Monthly records by institution")
print(pd.crosstab(events["admitted"].dt.to_period("M"), events["institution"]).to_string())

print("\nInstitution share of each borough, by split")
split_windows = {n: grid.split_windows(a, b) for n, (a, b) in cfg["study"]["splits"].items()}
for split, windows in split_windows.items():
    totals = cells[windows].sum(axis=0)
    share = pd.DataFrame(totals / totals.sum(axis=1, keepdims=True), index=[names[b] for b in boroughs], columns=institutions)
    print(f"\n{split}\n", share.round(2).to_string())

print("\nFirst full month with at least 10 records, per institution and borough, after the first month")
monthly = events.groupby(["institution", "borough", events["admitted"].dt.to_period("M")]).size().unstack(fill_value=0)
for (institution, borough), row in monthly.iterrows():
    active = row[row >= 10]
    if len(active) and active.index[0] > row.index[0]:
        print(f"{institution:10s} {names[borough]:28s} starts {active.index[0]}  peak {row.max()}")

print("\nEffect on the mean records per borough window (test)")
test = split_windows["test"]
pemex = institutions.index("PEMEX")
everyone = cells[test].sum(axis=2)
without = everyone - cells[test, :, pemex]
for i, code in enumerate(boroughs):
    print(f"{names[code]:28s} all {everyone[:, i].mean():6.1f}  without PEMEX {without[:, i].mean():6.1f}  "
          f"largest single-window PEMEX {cells[test, i, pemex].max():4d}")

print("\nJumps: windows where one institution alone adds at least 3 SDs of the borough's prior 20 windows")
total = cells.sum(axis=2)
rows = []
for t in range(22, grid.size):
    base = total[t - 22:t - 2]
    mean, sd = base.mean(axis=0), base.std(axis=0, ddof=1).clip(min=1)
    for b in range(len(boroughs)):
        if total[t, b] > mean[b] + 3 * sd[b]:
            prior = cells[t - 22:t - 2, b].mean(axis=0)
            gain = cells[t, b] - prior
            top = int(gain.argmax())
            rows.append((t >= test[0], institutions[top], gain[top] / max(total[t, b] - mean[b], 1), names[boroughs[b]], str(grid.first_day(t).date()), int(total[t, b])))
jumps = pd.DataFrame(rows, columns=["in_test", "main_institution", "share_of_excess", "borough", "window_start", "records"])
print("Windows above 3 SD:", len(jumps), "in test:", int(jumps["in_test"].sum()))
print(jumps.groupby(["in_test", "main_institution"])["share_of_excess"].agg(["count", "median"]).round(2).to_string())
print("Windows where one institution explains over 80% of the excess:", int((jumps["share_of_excess"] > 0.8).sum()))
print("Test-period jumps where one institution explains over 80%")
print(jumps[jumps["in_test"] & (jumps["share_of_excess"] > 0.8)].drop(columns="in_test").round(2).to_string(index=False))
