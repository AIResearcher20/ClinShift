"""Loading and harmonization of the UCI Heart Disease dataset.

The dataset consists of patient records from four sites. Feature
encoding and missingness differ across sites. This module loads each
site from the original UCI source and applies a documented set of
harmonization rules so that downstream analysis operates on a single
consistent representation.
"""

from io import StringIO
from urllib.request import urlopen

import numpy as np
import pandas as pd


COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target",
]

SITES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

BASE_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease/"

# Features retained for cross-site analysis. Columns that are almost
# entirely missing at one or more sites (ca, thal) are excluded to
# avoid imputation dominating the signal.
SHARED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]


def _download(filename: str) -> str:
    url = BASE_URL + filename
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


def load_site(name: str, filename: str) -> pd.DataFrame:
    """Load one site and attach a site label."""
    raw = _download(filename)
    frame = pd.read_csv(
        StringIO(raw),
        header=None,
        names=COLUMNS,
        na_values="?",
    )
    frame["site"] = name
    return frame


def binarize_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the five-level target to a binary indicator.

    The original target ranges from 0 to 4, where 0 indicates absence
    of disease and values above 0 indicate presence. This convention
    follows the original dataset description.
    """
    frame = frame.copy()
    frame["target"] = (frame["target"] > 0).astype(int)
    return frame


def harmonize(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply shared feature selection and basic type enforcement."""
    frame = frame.copy()
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("Int64")
    return frame


def load_all() -> pd.DataFrame:
    """Load all four sites and return a single harmonized frame."""
    frames = [load_site(name, fn) for name, fn in SITES.items()]
    combined = pd.concat(frames, ignore_index=True)
    combined = binarize_target(combined)
    combined = harmonize(combined)
    return combined


def site_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Report per-site sample size and missingness on shared features."""
    rows = []
    for site, group in frame.groupby("site"):
        row = {"site": site, "n": len(group)}
        for column in SHARED_FEATURES:
            row[f"{column}_missing_pct"] = round(
                group[column].isna().mean() * 100, 1
            )
        row["target_rate"] = round(group["target"].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("site")


if __name__ == "__main__":
    data = load_all()
    print(site_summary(data).to_string())
