"""Measurement of covariate and concept shift between sites.

Covariate shift refers to differences in the marginal distribution of
features between a source and a target site. It is quantified with the
total variation distance on categorical features and the standardized
mean difference on numeric features.

Concept shift refers to differences in the conditional relationship
between features and outcome. It is quantified by comparing the
predictive behavior of a model trained on the source site when
evaluated on subsets of the target site defined by predicted risk.
This is a descriptive measure, not a causal one.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data_loader import CATEGORICAL_FEATURES, SHARED_FEATURES


def total_variation(
    source: pd.Series,
    target: pd.Series,
) -> float:
    """Total variation distance between two categorical distributions."""
    source_counts = source.value_counts(normalize=True)
    target_counts = target.value_counts(normalize=True)
    categories = source_counts.index.union(target_counts.index)
    source_aligned = source_counts.reindex(categories, fill_value=0.0)
    target_aligned = target_counts.reindex(categories, fill_value=0.0)
    return float(0.5 * np.abs(source_aligned - target_aligned).sum())


def standardized_mean_difference(
    source: pd.Series,
    target: pd.Series,
) -> float:
    """Absolute standardized difference in means for a numeric feature."""
    source = source.dropna()
    target = target.dropna()
    if len(source) < 2 or len(target) < 2:
        return np.nan
    pooled = np.sqrt((source.var(ddof=1) + target.var(ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float(abs(source.mean() - target.mean()) / pooled)


def covariate_shift_table(
    frame: pd.DataFrame,
    source_site: str,
    target_site: str,
) -> pd.DataFrame:
    """Compute per-feature shift statistics between two sites."""
    source = frame[frame["site"] == source_site]
    target = frame[frame["site"] == target_site]

    rows = []
    for column in SHARED_FEATURES:
        if column in CATEGORICAL_FEATURES:
            value = total_variation(source[column], target[column])
            statistic = "total_variation"
        else:
            value = standardized_mean_difference(
                source[column], target[column]
            )
            statistic = "smd"
        rows.append({
            "feature": column,
            "statistic": statistic,
            "value": value,
        })
    return pd.DataFrame(rows).set_index("feature")


def ks_table(
    frame: pd.DataFrame,
    source_site: str,
    target_site: str,
) -> pd.DataFrame:
    """Kolmogorov-Smirnov statistic for numeric features between sites."""
    source = frame[frame["site"] == source_site]
    target = frame[frame["site"] == target_site]
    rows = []
    for column in SHARED_FEATURES:
        if column in CATEGORICAL_FEATURES:
            continue
        a = source[column].dropna()
        b = target[column].dropna()
        if len(a) < 2 or len(b) < 2:
            rows.append({"feature": column, "ks": np.nan, "p": np.nan})
            continue
        result = ks_2samp(a, b)
        rows.append({
            "feature": column,
            "ks": float(result.statistic),
            "p": float(result.pvalue),
        })
    return pd.DataFrame(rows).set_index("feature")
