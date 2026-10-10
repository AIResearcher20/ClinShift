"""Unlabeled shift indicators and their relation to model degradation.

For each target site, compute a set of shift indicators that do not
depend on target labels, and compare them to the change in
discrimination and calibration observed when a source-trained model is
evaluated on that target. The purpose is to test whether the unlabeled
indicators are informative about degradation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.data_loader import SHARED_FEATURES, load_all
from src.shift_detection import compute_all_indicators

SEED = 42
SOURCE_SITE = "cleveland"
RESULTS_DIR = Path("results")


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the upstream experiments first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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

    indicator_rows = []
    for target_site in targets:
        target = data[data["site"] == target_site].copy()
        indicators = compute_all_indicators(source, target, seed=SEED)
        indicators["site"] = target_site
        indicator_rows.append(indicators)
        print(f"\n{target_site}")
        for key, value in indicators.items():
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
