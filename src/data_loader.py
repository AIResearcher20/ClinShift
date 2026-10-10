"""Loading and harmonization of the UCI Heart Disease dataset.

The dataset is downloaded from the UCI Machine Learning Repository
and cached under data/raw/. If all required local files are present,
no network access is attempted.

This module does not impute missing values or fit preprocessing steps.
Those operations belong in the training pipeline to prevent data leakage.
"""

from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import pandas as pd


RAW_DIR = Path("data/raw")

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
    "target",
]

SHARED_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
]

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

TARGET_COLUMN = "target"


def _ensure_raw_files() -> None:
    """Download and extract the archive if required files are missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        filename
        for filename in FILES.values()
        if not (RAW_DIR / filename).is_file()
    ]

    if not missing:
        return

    request = Request(
        ARCHIVE_URL,
        headers={"User-Agent": "ClinShift research project"},
    )

    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "Unable to download the UCI Heart Disease archive. "
            "Check your internet connection or place all four "
            f"required files in {RAW_DIR.resolve()}."
        ) from exc

    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            available = {
                Path(member).name: member
                for member in archive.namelist()
                if not member.endswith("/")
            }

            missing_from_archive = [
                filename
                for filename in FILES.values()
                if filename not in available
            ]

            if missing_from_archive:
                raise RuntimeError(
                    "The downloaded archive does not contain all "
                    f"required files: {missing_from_archive}"
                )

            # Read all required files before writing anything, so a
            # missing archive member cannot leave a partial extraction.
            extracted = {
                filename: archive.read(available[filename])
                for filename in FILES.values()
            }

    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            "The downloaded UCI archive is not a valid ZIP file."
        ) from exc

    for filename, content in extracted.items():
        (RAW_DIR / filename).write_bytes(content)


def _load_site(name: str, filename: str) -> pd.DataFrame:
    """Load one center and validate its structure."""
    path = RAW_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset file: {path}. "
            "Run _ensure_raw_files() first."
        )

    frame = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?", ""],
        skipinitialspace=True,
    )

    if frame.empty:
        raise ValueError(f"The dataset file is empty: {path}")

    if frame.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Unexpected number of columns in {path}: "
            f"{frame.shape[1]}"
        )

    # Convert expected numeric fields explicitly.
    for column in COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    # A malformed feature value becomes missing; report it through
    # the regular missing-value summary rather than imputing it here.
    frame["site"] = name

    return frame


def load_all() -> pd.DataFrame:
    """Load all centers and harmonize the binary outcome.

    The original UCI target uses 0 for absence of heart disease and
    positive integers for disease severity. This function maps valid
    target values to binary labels: 0 = absence, 1 = presence.

    Rows with missing target labels are excluded. Missing feature
    values are preserved for downstream, leakage-safe preprocessing.
    """
    _ensure_raw_files()

    frames = [
        _load_site(name, filename)
        for name, filename in FILES.items()
    ]

    combined = pd.concat(frames, ignore_index=True)

    # Check target values before converting to binary labels.
    observed_targets = combined[TARGET_COLUMN].dropna()
    invalid_targets = observed_targets[
        ~observed_targets.isin([0, 1, 2, 3, 4])
    ]

    if not invalid_targets.empty:
        raise ValueError(
            "Unexpected target values found: "
            f"{sorted(invalid_targets.unique().tolist())}"
        )

    missing_target_count = combined[TARGET_COLUMN].isna().sum()

    if missing_target_count:
        print(
            "Warning: excluding "
            f"{missing_target_count} rows with missing target labels."
        )

    combined = combined.loc[
        combined[TARGET_COLUMN].notna()
    ].copy()

    combined[TARGET_COLUMN] = (
        combined[TARGET_COLUMN] > 0
    ).astype("int8")

    # Preserve categorical codes as nullable integers.
    for column in CATEGORICAL_FEATURES:
        combined[column] = combined[column].astype("Int64")

    combined[TARGET_COLUMN] = combined[TARGET_COLUMN].astype("int8")

    combined = combined.reset_index(drop=True)

    return combined


def site_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize sample counts, missingness, and outcome rates by center."""
    required = set(SHARED_FEATURES + [TARGET_COLUMN, "site"])
    missing_columns = required.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            "Input frame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    rows = []

    for site, group in frame.groupby("site", sort=True):
        row = {
            "site": site,
            "n": len(group),
            "target_rate": round(
                group[TARGET_COLUMN].mean(), 3
            ),
        }

        for column in SHARED_FEATURES:
            row[f"{column}_missing_pct"] = round(
                group[column].isna().mean() * 100,
                1,
            )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("site")


if __name__ == "__main__":
    data = load_all()

    print("Dataset shape:", data.shape)
    print("\nSamples per center:")
    print(data["site"].value_counts().sort_index().to_string())

    print("\nTarget distribution by center:")
    print(
        pd.crosstab(
            data["site"],
            data[TARGET_COLUMN],
            margins=True,
        ).to_string()
    )

    print("\nMissingness and target-rate summary:")
    print(site_summary(data).to_string())
