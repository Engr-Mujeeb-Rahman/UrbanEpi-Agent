"""Contribution bounding. The public data has no person IDs, so this path runs only on test IDs."""
import pandas as pd


def apply_person_cap(records: pd.DataFrame, person_column: str, cell_columns: list[str], cap: int) -> pd.DataFrame:
    """Keep one record per person per cell, then each person's first `cap` cells in time order."""
    ordered = records.sort_values("admitted", kind="stable")
    single = ordered.drop_duplicates([person_column, *cell_columns], keep="first")
    return single[single.groupby(person_column).cumcount() < cap]
