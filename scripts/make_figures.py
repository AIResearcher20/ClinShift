"""Generate figures from saved experiment results.

Reads the JSON and CSV files produced by the experiment scripts and
writes PNG figures into results/figures/. The functions in this module
perform no modeling and do not depend on the source dataset.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SITE_LABELS = {
    "hungary": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}


def load_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the experiment scripts first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_degradation_table(model: str) -> pd.DataFrame:
    """Combine baseline and transfer metrics into one table."""
    baseline = load_json("baseline.json")
    transfer = load_json("transfer.json")

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


def figure_baseline_vs_transfer() -> None:
    """Bar chart of baseline and transfer ECE per site and model."""
    for model in ["logreg", "gbm"]:
        table = build_degradation_table(model)

        x = np.arange(len(table))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            x - width / 2,
            table["baseline_ece"],
            width,
            label="Baseline (within site)",
        )
        ax.bar(
            x + width / 2,
            table["transfer_ece"],
            width,
            label="Transfer from Cleveland",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [SITE_LABELS[s] for s in table.index],
            rotation=15,
        )
        ax.set_ylabel("Expected Calibration Error")
        ax.set_title(f"Calibration before and after transfer ({model})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"ece_baseline_transfer_{model}.png", dpi=200)
        plt.close(fig)


def figure_indicator_vs_degradation() -> None:
    """Scatter of entropy difference against change in ECE."""
    indicators = pd.read_csv(
        RESULTS_DIR / "shift_indicators.csv", index_col="site"
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    markers = {"logreg": "o", "gbm": "s"}
    for model, marker in markers.items():
        table = build_degradation_table(model)
        x = indicators["entropy_difference"].reindex(table.index).to_numpy()
        y = table["delta_ece"].to_numpy()
        ax.scatter(x, y, marker=marker, s=70, label=model)

        for site, xv, yv in zip(table.index, x, y):
            ax.annotate(
                SITE_LABELS[site],
                xy=(xv, yv),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Entropy difference (target minus source)")
    ax.set_ylabel("Change in Expected Calibration Error")
    ax.set_title("Prediction entropy shift vs. calibration degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "entropy_vs_delta_ece.png", dpi=200)
    plt.close(fig)


def figure_feature_indicators() -> None:
    """Bar chart of feature-based shift indicators per target site."""
    shift = pd.read_csv(RESULTS_DIR / "shift.csv")

    numeric = shift[shift["statistic"] == "smd"]
    pivot = numeric.pivot_table(
        index="feature", columns="target", values="value", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    pivot.plot(kind="barh", ax=ax)
    ax.set_xlabel("Standardized mean difference from Cleveland")
    ax.set_title("Feature-level covariate shift by target site")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_shift.png", dpi=200)
    plt.close(fig)


def main() -> None:
    figure_baseline_vs_transfer()
    figure_indicator_vs_degradation()
    figure_feature_indicators()
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
