"""Unlabeled indicators of distribution shift between sites.

The functions in this module quantify the distance between a source
and a target feature distribution without using outcome labels at the
target. They are intended to be computed at deployment time, before
outcomes become available, so that the reliability of a transferred
model can be estimated in advance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import CATEGORICAL_FEATURES, SHARED_FEATURES


def _encode(frame: pd.DataFrame) -> np.ndarray:
    """Return a numeric matrix for distance computations.

    Categorical features are one-hot encoded. Missing values are
    replaced by the column median for numeric features and by zero for
    categorical features, which matches the preprocessing applied in
    the modeling pipeline.
    """
    numeric = [
        column for column in SHARED_FEATURES
        if column not in CATEGORICAL_FEATURES
    ]
    parts = []

    numeric_block = frame[numeric].to_numpy(dtype=float)
    if numeric_block.size:
        median = np.nanmedian(numeric_block, axis=0)
        median = np.where(np.isnan(median), 0.0, median)
        mask = np.isnan(numeric_block)
        numeric_block = np.where(mask, median, numeric_block)
        parts.append(numeric_block)

    for column in CATEGORICAL_FEATURES:
        values = frame[column].astype("float").fillna(-1).to_numpy()
        categories = np.unique(values)
        one_hot = np.zeros((len(values), len(categories)))
        for idx, category in enumerate(categories):
            one_hot[:, idx] = (values == category).astype(float)
        parts.append(one_hot)

    matrix = np.hstack(parts)
    return StandardScaler().fit_transform(matrix)


def density_ratio_auroc(
    source: pd.DataFrame,
    target: pd.DataFrame,
    seed: int = 0,
) -> float:
    """AUROC of a classifier trained to distinguish source from target.

    A value near 0.5 indicates the two distributions are
    indistinguishable. A value near 1.0 indicates they are separable,
    which implies substantial covariate shift.
    """
    x_source = _encode(source)
    x_target = _encode(target)

    x = np.vstack([x_source, x_target])
    y = np.concatenate([
        np.zeros(len(x_source)),
        np.ones(len(x_target)),
    ])

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in splitter.split(x, y):
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(x[train_idx], y[train_idx])
        prob = clf.predict_proba(x[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], prob))

    return float(np.mean(scores))


def mmd_rbf(
    source: pd.DataFrame,
    target: pd.DataFrame,
    gamma: float | None = None,
    max_samples: int = 500,
    seed: int = 0,
) -> float:
    """Maximum Mean Discrepancy with an RBF kernel.

    The estimator is the biased V-statistic. Subsampling keeps the
    computation tractable and reduces memory pressure on Colab.
    """
    rng = np.random.default_rng(seed)

    x = _encode(source)
    y = _encode(target)

    if len(x) > max_samples:
        idx = rng.choice(len(x), max_samples, replace=False)
        x = x[idx]
    if len(y) > max_samples:
        idx = rng.choice(len(y), max_samples, replace=False)
        y = y[idx]

    if gamma is None:
        combined = np.vstack([x, y])
        pair_dist = np.sum(
            (combined[:, None, :] - combined[None, :, :]) ** 2,
            axis=-1,
        )
        median = np.median(pair_dist[pair_dist > 0])
        gamma = 1.0 / median if median > 0 else 1.0

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * dist)

    k_xx = kernel(x, x).mean()
    k_yy = kernel(y, y).mean()
    k_xy = kernel(x, y).mean()

    return float(k_xx + k_yy - 2 * k_xy)


def energy_distance(
    source: pd.DataFrame,
    target: pd.DataFrame,
    max_samples: int = 500,
    seed: int = 0,
) -> float:
    """Energy distance between two empirical distributions."""
    rng = np.random.default_rng(seed)

    x = _encode(source)
    y = _encode(target)

    if len(x) > max_samples:
        idx = rng.choice(len(x), max_samples, replace=False)
        x = x[idx]
    if len(y) > max_samples:
        idx = rng.choice(len(y), max_samples, replace=False)
        y = y[idx]

    def pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        )

    d_xy = pairwise(x, y).mean()
    d_xx = pairwise(x, x).mean()
    d_yy = pairwise(y, y).mean()

    value = 2 * d_xy - d_xx - d_yy
    return float(max(value, 0.0))


def compute_all_indicators(
    source: pd.DataFrame,
    target: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Compute every unlabeled shift indicator for one pair of sites."""
    return {
        "density_ratio_auroc": density_ratio_auroc(source, target, seed),
        "mmd_rbf": mmd_rbf(source, target, seed=seed),
        "energy_distance": energy_distance(source, target, seed=seed),
    }
