"""Inspect cholesterol distributions across the four UCI Heart Disease sites."""

from io import BytesIO
from urllib.request import urlopen
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
"age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
"thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

def main() -> None:
print("Downloading the original UCI Heart Disease archive...")

with urlopen(ARCHIVE_URL, timeout=60) as response:
    archive_data = response.read()

frames = []

with ZipFile(BytesIO(archive_data)) as archive:
    archive_names = archive.namelist()

    for site, filename in FILES.items():
        matches = [
            name for name in archive_names
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
            )

        frame["chol"] = pd.to_numeric(
            frame["chol"], errors="coerce"
        )
        frame["site"] = site
        frames.append(frame)

data = pd.concat(frames, ignore_index=True)

print("\nCholesterol summary by site:")
print(
    data.groupby("site")["chol"]
    .agg(["count", "mean", "std", "min", "median", "max"])
    .round(2)
    .to_string()
)

print("\nMissing cholesterol values by site:")
print(
    data.groupby("site")["chol"]
    .apply(lambda values: values.isna().sum())
    .to_string()
)

print("\nFive smallest cholesterol values by site:")
for site, group in data.groupby("site"):
    print(f"{site:<12} {group['chol'].dropna().nsmallest(5).tolist()}")

print("\nFive largest cholesterol values by site:")
for site, group in data.groupby("site"):
    print(f"{site:<12} {group['chol'].dropna().nlargest(5).tolist()}")

if name == "main":
main()
