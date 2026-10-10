"""Cross-site transfer experiment.

A model is trained on the Cleveland site and evaluated on each of the
three remaining sites. For every target site, we report discrimination
and calibration before and after post-hoc calibration. Calibrators are
fitted on a held-out portion of the source site, never on the target.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.calibration import apply_calibrators
from src.data_loader import SHARED_FEATURES, load_all
from src.metrics import summarize
from src.models import (
    build_pipeline,
    gradient_boosting,
    logistic_regression,
)

SEED = 42
SOURCE_SITE = "cleveland"
RESULTS_DIR = Path("results")


def split_source(frame, test_size: float, seed: int):
    features = frame[SHARED_FEATURES].to_numpy()
    target = frame["target"].to_numpy()
    return train_test_split(
        features, target,
        test_size=test_size,
        stratify=target,
        random_state=seed,
    )


def evaluate(spec, source_frame, target_frame) -> dict:
    x_train, x_cal, y_train, y_cal = split_source(
        source_frame, test_size=0.3, seed=SEED
    )
    pipeline = build_pipeline(spec)
    pipeline.fit(x_train, y_train)

    prob_cal = pipeline.predict_proba(x_cal)[:, 1]
    prob_target = pipeline.predict_proba(
        target_frame[SHARED_FEATURES].to_numpy()
    )[:, 1]
    y_target = target_frame["target"].to_numpy()

    raw = summarize(y_target, prob_target)
    calibrated = apply_calibrators(prob_cal, y_cal, prob_target)

    output = {"raw": raw}
    for method, prob in calibrated.items():
        output[method] = summarize(y_target, prob)
    return output


def main() -> None:
    data = load_all()
    source_frame = data[data["site"] == SOURCE_SITE]

    specs = [logistic_regression(SEED), gradient_boosting(SEED)]
    results = {}

    for spec in specs:
        results[spec.name] = {}
        for site, group in data.groupby("site"):
            if site == SOURCE_SITE:
                continue
            metrics = evaluate(spec, source_frame, group)
            results[spec.name][site] = metrics
            print(f"\n{spec.name} -> {site}")
            for method, values in metrics.items():
                print(f"  {method}")
                for key, value in values.items():
                    print(f"    {key:<8} {value:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "transfer.json"
    with open(out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
