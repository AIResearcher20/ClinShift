"""Evaluation metrics for discrimination and calibration.

Discrimination is measured with AUROC. Calibration is measured with the
Brier score and the Expected Calibration Error (ECE), computed with
equal-width binning of predicted probabilities.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width binning.

    The probability range [0, 1] is partitioned into n_bins intervals
    of equal width. Within each bin, the absolute difference between
    the mean predicted probability and the observed frequency is
    weighted by the fraction of samples in that bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError("Predicted probabilities must lie in [0, 1].")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)

    total = len(y_prob)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = mask.sum()
        if count == 0:
            continue
        avg_prob = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += (count / total) * abs(avg_prob - avg_true)
    return float(ece)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, mean predicted probability, and observed rate.

    Bins with no samples are reported as NaN so that callers can filter
    them before plotting.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mean_prob = np.full(n_bins, np.nan)
    obs_rate = np.full(n_bins, np.nan)
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.sum() == 0:
            continue
        mean_prob[bin_id] = y_prob[mask].mean()
        obs_rate[bin_id] = y_true[mask].mean()
    return centers, mean_prob, obs_rate


def summarize(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all reported metrics for one evaluation set."""
    return {
        "auroc": auroc(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }
