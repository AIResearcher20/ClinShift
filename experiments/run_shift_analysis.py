"""Covariate shift analysis across sites.

For each pair of source and target sites, compute the total variation
distance on categorical features and the standardized mean difference
on numeric features. Results are written to results/shift.csv.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_all
from src.shift import covariate_shift_table, ks_table

SOURCE_SITE = "cleveland"
RESULTS_DIR = Path("results")


def main() -> None:
    data = load_all()
    targets = [s for s in data["site"].unique() if s != SOURCE_SITE]

    all_rows = []
    for target in targets:
        shift = covariate_shift_table(data, SOURCE_SITE, target)
        ks = ks_table(data, SOURCE_SITE, target)
        merged = shift.join(ks[["ks", "p"]])
        merged["source"] = SOURCE_SITE
        merged["target"] = target
        all_rows.append(merged.reset_index())

    combined = pd.concat(all_rows, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "shift.csv"
    combined.to_csv(out, index=False)
    print(combined.to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
