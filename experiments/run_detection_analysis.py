"""Unlabeled shift indicators and their relation to model degradation.

For each target site, compute feature-based and prediction-based shift
indicators that do not use target labels, and compare them to the
change in discrimination and calibration observed when a
source-trained model is evaluated on that target. The comparison is
descriptive; with three target sites, no inferential claim is made
about the general relationship between indicators and degradation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold

from src.data_loader import SHARED_FEATURES, load_all
from src.models import build_pipeline, logistic_regression
from src.prediction_indicators import compute_prediction_indicators
from src.shift_detection import compute_all_indicators

SEED = 42
SOURCE_SITE = "cleveland"
N_FOLDS = 5
RESULTS_DIR = Path("results")


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the upstream experiments first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def source_out_of_fold_probabilities(
    source: pd.DataFrame,
    seed: int,
    n_splits: int,
) -> np.ndarray:
    """Return out-of-fold predictions on the source site.

    Each sample is predicted by a model that did not see it during
    training. This avoids the optimistic bias that would arise if the
    model were evaluated on its own training data.
    """
    features = source[SHARED_FEATURES]
    target = source["target"].to_numpy()

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    predictions = np.full(len(source), np.nan)
    spec = logistic_regression(seed)

    for train_idx, test_idx in splitter.split(features, target):
        pipeline = build_pipeline(spec)
        pipeline.fit(
            features.iloc[train_idx],
            target[train_idx],
        )
        predictions[test_idx] = pipeline.predict_proba(
            features.iloc[test_idx]
        )[:, 1]

    return predictions


def target_probabilities(
    source: pd.DataFrame,
    target: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    """Return predictions on the target site from a source-trained
    model.

    The model is fitted on the full source site. Labels from the
    target site are not used at any point.
    """
    spec = logistic_regression(seed)
    pipeline = build_pipeline(spec)
    pipeline.fit(
        source[SHARED_FEATURES],
        source["target"].to_numpy(),
    )
    return pipeline.predict_proba(
        target[SHARED_FEATURES]
    )[:, 1]


def degradation_table(
    baseline: dict,
    transfer: dict,
    model: str,
) -> pd.DataFrame:
    """Compute the change in AUROC and ECE from baseline to transfer."""
    rows = []
    for site in transfer[model]:
        base = baseline[site][model]
        trans = transfer[model][site]["raw"]
        rows.append({
            "site": site,
            "baseline_auroc": base["auroc"],
            "transfer_auroc": trans["auroc"],
            "delta_auroc": trans["auroc"] - base["auroc"],
            "baseline_ece": base["ece"],
            "transfer_ece": trans["ece"],
            "delta_ece": trans["ece"] - base["ece"],
        })
    return pd.DataFrame(rows).set_index("site")


def main() -> None:
    data = load_all()
    source = data[data["site"] == SOURCE_SITE].copy()

    targets = [
        site for site in data["site"].unique()
        if site != SOURCE_SITE
    ]

    source_oof = source_out_of_fold_probabilities(
        source, SEED, N_FOLDS
    )

    indicator_rows = []
    for target_site in targets:
        target = data[data["site"] == target_site].copy()

        feature_indicators = compute_all_indicators(
            source, target, seed=SEED
        )

        target_prob = target_probabilities(source, target, SEED)
        prediction_indicators = compute_prediction_indicators(
            source_oof, target_prob
        )

        row = {
            "site": target_site,
            **feature_indicators,
            **prediction_indicators,
        }
        indicator_rows.append(row)

        print(f"\n{target_site}")
        for key, value in row.items():
            if key == "site":
                continue
            print(f"  {key:<24} {value:.4f}")

    indicators = pd.DataFrame(indicator_rows).set_index("site")

    baseline = load_json("baseline.json")
    transfer = load_json("transfer.json")

    for model in ["logreg", "gbm"]:
        degradation = degradation_table(baseline, transfer, model)
        combined = indicators.join(degradation)

        print(f"\n{'=' * 60}")
        print(f"Combined view for {model}")
        print(f"{'=' * 60}")
        print(combined.round(4).to_string())

        print(f"\nCorrelation with delta_ece ({model})")
        for indicator in indicators.columns:
            x = combined[indicator].to_numpy()
            y = combined["delta_ece"].to_numpy()
            if len(x) < 3:
                continue
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            print(
                f"  {indicator:<24} "
                f"pearson={pearson.statistic:+.3f} "
                f"spearman={spearman.statistic:+.3f}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "shift_indicators.csv"
    indicators.to_csv(out)
    print(f"\nSaved indicators to {out}")


if __name__ == "__main__":
    main()
