"""Load and harmonize the original UCI Heart Disease datasets.

The four sites are downloaded directly from the UCI machine learning
repository. This avoids any preprocessing or unit conversion that may
have been applied in redistributions of the dataset. Site-level
differences in feature encoding and missingness are preserved so that
they can be analyzed rather than silently removed.
"""

from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


ARCHIVE_URL = (
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
)

FILES = {
    "cleveland": "processed.cleveland.data",
    "hungary": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

SHARED_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
]


def load_all() -> pd.DataFrame:
    """Download the UCI archive and return a harmonized frame."""

    request = Request(
        ARCHIVE_URL,
        headers={"User-Agent": "ClinShift research project"},
    )

    with urlopen(request, timeout=60) as response:
        archive_data = response.read()

    frames = []

    with ZipFile(BytesIO(archive_data)) as archive:
        archive_names = archive.namelist()

        for site, filename in FILES.items():
            matches = [
                name
                for name in archive_names
                if name.endswith(filename)
            ]

            if not matches:
                raise FileNotFoundError(
                    f"Could not find {filename} in the UCI archive."
                )

            with archive.open(matches[0]) as file:
                frame = pd.read_csv(
                    file,
                    header=None,
                    names=COLUMNS,
                    na_values=["?", ""],
                    skipinitialspace=True,
                )

            frame["site"] = site
            frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    # Coerce every original column to numeric. Non-numeric entries
    # become NaN and are handled downstream by the imputation pipeline.
    for column in COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Records without a usable outcome cannot be used for supervised
    # evaluation and are dropped.
    data = data.dropna(subset=["num"]).copy()

    # The original outcome ranges from 0 to 4, where 0 indicates absence
    # of disease and values above 0 indicate presence.
    data["target"] = (data["num"] > 0).astype(int)

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("Int64")

    return data.reset_index(drop=True)


def site_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize sample sizes, missingness, and outcome rates by site."""

    rows = []

    for site, group in frame.groupby("site"):
        row = {
            "site": site,
            "n": len(group),
        }

        for column in SHARED_FEATURES:
            row[f"{column}_missing_pct"] = round(
                group[column].isna().mean() * 100,
                1,
            )

        row["target_rate"] = round(group["target"].mean(), 3)
        rows.append(row)

    return pd.DataFrame(rows).set_index("site")


if __name__ == "__main__":
    data = load_all()

    print("Dataset loaded successfully.")
    print(f"Total records: {len(data)}")
    print(f"Total features: {len(SHARED_FEATURES)}")

    print("\nSite summary:")
    print(site_summary(data).to_string())

