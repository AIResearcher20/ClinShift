"""Baseline experiment: within-site cross-validation.

Each site is evaluated independently using stratified k-fold
cross-validation. The purpose is to establish an in-distribution
reference for each model before studying cross-site transfer.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data_loader import SHARED_FEATURES, load_all, site_summary
from src.metrics import summarize
from src.models import (
    build_pipeline,
    gradient_boosting,
    logistic_regression,
)

SEED = 42
N_SPLITS = 5
RESULTS_DIR = Path("results")


def cross_validate(frame, spec, n_splits: int, seed: int) -> dict:
    features = frame[SHARED_FEATURES]
    target = frame["target"].to_numpy()

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    fold_metrics = []
    for train_idx, test_idx in splitter.split(features, target):
        pipeline = build_pipeline(spec)
        pipeline.fit(
            features.iloc[train_idx],
            target[train_idx],
        )
        prob = pipeline.predict_proba(
            features.iloc[test_idx]
        )[:, 1]
        fold_metrics.append(summarize(target[test_idx], prob))

    return {
        key: float(np.mean([m[key] for m in fold_metrics]))
        for key in fold_metrics[0]
    }


def main() -> None:
    data = load_all()
    print("Site summary")
    print(site_summary(data).to_string())

    specs = [logistic_regression(SEED), gradient_boosting(SEED)]
    results = {}

    for site, group in data.groupby("site"):
        results[site] = {}
        for spec in specs:
            metrics = cross_validate(group, spec, N_SPLITS, SEED)
            results[site][spec.name] = metrics
            print(f"\n{site} / {spec.name}")
            for key, value in metrics.items():
                print(f"  {key:<8} {value:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "baseline.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
