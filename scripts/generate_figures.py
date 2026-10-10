"""Generate all figures from saved experiment results.

Reads baseline.json, transfer.json, and shift.csv from results/ and
writes PNG figures into results/figures/. This script is intended to
run after the experiment scripts and does not perform any modeling.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import reliability_curve

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    with open(RESULTS_DIR / name) as handle:
        return json.load(handle)


def plot_transfer_summary(transfer: dict) -> None:
    """Bar chart of AUROC and ECE per target site, per model."""
    rows = []
    for model, per_site in transfer.items():
        for site, methods in per_site.items():
            rows.append({
                "model": model,
                "site": site,
                "auroc": methods["raw"]["auroc"],
                "ece": methods["raw"]["ece"],
            })
    frame = pd.DataFrame(rows)

    for metric in ["auroc", "ece"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = frame.pivot(index="site", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Transfer performance by site ({metric.upper()})")
        ax.axhline(
            0.5 if metric == "auroc" else 0.0,
            color="gray", linestyle="--", linewidth=0.8,
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"transfer_{metric}.png", dpi=200)
        plt.close(fig)


def plot_shift_vs_auroc(shift: pd.DataFrame, transfer: dict) -> None:
    """Scatter of mean covariate shift magnitude against AUROC."""
    per_target = shift.groupby("target")["value"].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    for model, per_site in transfer.items():
        sites = list(per_site.keys())
        x = [per_target[s] for s in sites]
        y = [per_site[s]["raw"]["auroc"] for s in sites]
        ax.scatter(x, y, label=model, s=60)
    ax.set_xlabel("Mean covariate shift magnitude")
    ax.set_ylabel("AUROC on target site")
    ax.set_title("Covariate shift vs. transfer AUROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shift_vs_auroc.png", dpi=200)
    plt.close(fig)


def plot_shift_per_feature(shift: pd.DataFrame) -> None:
    """Heatmap of per-feature shift magnitude across target sites."""
    pivot = shift.pivot(index="feature", columns="target", values="value")
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="Shift magnitude")
    ax.set_title("Per-feature shift relative to Cleveland")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shift_per_feature.png", dpi=200)
    plt.close(fig)


def main() -> None:
    baseline = load_json("baseline.json")
    transfer = load_json("transfer.json")
    shift = pd.read_csv(RESULTS_DIR / "shift.csv")

    plot_transfer_summary(transfer)
    plot_shift_vs_auroc(shift, transfer)
    plot_shift_per_feature(shift)

    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
