"""Loading and harmonization of the UCI Heart Disease dataset.

The dataset consists of patient records from four sites. Feature
encoding and missingness differ across sites. This module loads a
single combined CSV file and applies a documented set of harmonization
rules so that downstream analysis operates on a single consistent
representation.

The cholesterol feature is excluded from the shared feature set. The
standardized mean difference for cholesterol between Cleveland and
Switzerland is 6.74 with a Kolmogorov-Smirnov statistic of 1.0,
indicating no overlap between the two distributions. This is most
likely a unit or recording discrepancy rather than a genuine
physiological difference. See docs/decisions.md for details.
"""

from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/heart_disease_uci.csv")

SITE_COLUMN = "dataset"
TARGET_COLUMN = "num"

SHARED_FEATURES = [
    "age", "sex", "cp", "trestbps",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

COLUMN_RENAMES = {
    "thalch": "thalach",
}


def load_all() -> pd.DataFrame:
    """Load the combined CSV and return a harmonized frame."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {RAW_PATH}. Download the UCI Heart "
            f"Disease dataset from Kaggle and place it there."
        )

    frame = pd.read_csv(RAW_PATH)
    frame = frame.rename(columns=COLUMN_RENAMES)
    frame["site"] = frame[SITE_COLUMN].astype(str).str.lower()
    frame["target"] = (frame[TARGET_COLUMN] > 0).astype(int)

    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].astype("Int64")

    return frame


def site_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report per-site sample size and missingness on shared features."""
    rows = []
    for site, group in frame.groupby("site"):
        row = {"site": site, "n": len(group)}
        for column in SHARED_FEATURES:
            if column in group.columns:
                row[f"{column}_missing_pct"] = round(
                    group[column].isna().mean() * 100, 1
                )
        row["target_rate"] = round(group["target"].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("site")


if __name__ == "__main__":
    data = load_all()
    print(site_summary(data).to_string())
