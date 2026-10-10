"""Stability of baseline and transfer results across random seeds.

The main pipeline uses a single seed. This script repeats the
within-site cross-validation and the cross-site transfer with several
seeds and reports the mean and standard deviation of the metrics. The
purpose is to show that the findings in the main results are not
specific to one random initialization.

The output is written to results/robustness/ and does not overwrite
the primary result files in results/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.calibration import apply_calibrators
from src.data_loader import SHARED_FEATURES, load_all
from src.metrics import summarize
from src.models import (
    build_pipeline,
    gradient_boosting,
    logistic_regression,
)

SEEDS = [0, 1, 2, 3, 4]
SOURCE_SITE = "cleveland"
N_SPLITS = 5
RESULTS_DIR = Path("results/robustness")


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


def transfer_once(
    spec,
    source_frame,
    target_frame,
    seed: int,
) -> dict:
    features = source_frame[SHARED_FEATURES]
    target = source_frame["target"].to_numpy()
    x_train, x_cal, y_train, y_cal = train_test_split(
        features,
        target,
        test_size=0.3,
        stratify=target,
        random_state=seed,
    )
    pipeline = build_pipeline(spec)
    pipeline.fit(x_train, y_train)

    prob_cal = pipeline.predict_proba(x_cal)[:, 1]
    prob_target = pipeline.predict_proba(
        target_frame[SHARED_FEATURES]
    )[:, 1]
    y_target = target_frame["target"].to_numpy()

    raw = summarize(y_target, prob_target)
    calibrated = apply_calibrators(prob_cal, y_cal, prob_target)

    output = {"raw": raw}
    for method, prob in calibrated.items():
        output[method] = summarize(y_target, prob)
    return output


def aggregate(records: list[dict]) -> dict:
    """Compute mean and std for each metric across seeds."""
    keys = records[0].keys()
    return {
        key: {
            "mean": float(np.mean([r[key] for r in records])),
            "std": float(np.std([r[key] for r in records], ddof=0)),
        }
        for key in keys
    }


def main() -> None:
    data = load_all()
    source = data[data["site"] == SOURCE_SITE]
    targets = [
        site for site in data["site"].unique()
        if site != SOURCE_SITE
    ]

    specs = [
        ("logreg", logistic_regression),
        ("gbm", gradient_boosting),
    ]

    results = {"seeds": SEEDS, "baseline": {}, "transfer": {}}

    for spec_name, spec_factory in specs:
        results["baseline"][spec_name] = {}
        for site, group in data.groupby("site"):
            per_seed = [
                cross_validate(group, spec_factory(seed), N_SPLITS, seed)
                for seed in SEEDS
            ]
            results["baseline"][spec_name][site] = aggregate(per_seed)

        results["transfer"][spec_name] = {}
        for target_site in targets:
            target_frame = data[data["site"] == target_site]
            per_seed = []
            for seed in SEEDS:
                transfer_result = transfer_once(
                    spec_factory(seed),
                    source,
                    target_frame,
                    seed,
                )
                per_seed.append(transfer_result)

            methods = per_seed[0].keys()
            results["transfer"][spec_name][target_site] = {
                method: aggregate(
                    [r[method] for r in per_seed]
                )
                for method in methods
            }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "seed_sensitivity.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Ran {len(SEEDS)} seeds across {len(targets)} targets")
    print(f"Saved to {out}")

    print("\nSummary of AUROC standard deviation across seeds")
    for spec_name in results["baseline"]:
        for site, metrics in results["baseline"][spec_name].items():
            auroc = metrics["auroc"]
            ece = metrics["ece"]
            print(
                f"  {spec_name:<8} {site:<12} "
                f"auroc={auroc['mean']:.4f}±{auroc['std']:.4f}  "
                f"ece={ece['mean']:.4f}±{ece['std']:.4f}"
            )


if __name__ == "__main__":
    main()
