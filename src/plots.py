"""Figure generation for the ClinShift report.

All figures are saved as PNG files in results/figures at 200 dpi. Each
function takes already-computed values so that plotting is decoupled
from modeling.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import reliability_curve

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str,
    filename: str,
    n_bins: int = 10,
) -> None:
    centers, mean_prob, obs_rate = reliability_curve(
        y_true, y_prob, n_bins=n_bins
    )
    mask = ~np.isnan(mean_prob)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.plot(mean_prob[mask], obs_rate[mask], marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)


def shift_vs_degradation(
    shift_table: pd.DataFrame,
    transfer_table: pd.DataFrame,
    filename: str,
) -> None:
    """Scatter plot of shift magnitude against AUROC drop per site."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for model_name, per_site in transfer_table.items():
        sites = list(per_site.keys())
        drops = [
            per_site[s]["raw"]["auroc"] for s in sites
        ]
        shifts = [
            shift_table.loc[shift_table["target"] == s, "value"].mean()
            for s in sites
        ]
        ax.scatter(shifts, drops, label=model_name)

    ax.set_xlabel("Mean covariate shift magnitude")
    ax.set_ylabel("AUROC on target site")
    ax.set_title("Shift vs. transfer performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)
