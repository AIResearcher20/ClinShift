"""Descriptive measures of concept divergence between sites.

Concept divergence refers to differences in the relationship between
features and outcome across sites. This module reports two descriptive
quantities that are consistent with such a difference and that can
motivate further analysis. It does not identify concept shift, and it
does not separate it from other sources of variation such as sample
size, class balance, or measurement noise.

The first quantity is the Euclidean distance between coefficient
vectors of logistic regression models fitted independently at each
site. All models operate on a feature space defined by a single
preprocessor fitted on the source site, so that differences in scaling
or encoding do not contribute to the distance.

The second quantity is the mean absolute difference between the
predictions of these models on the source evaluation set. If two
models have learned different decision boundaries, their predictions
on the same samples will differ.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import SHARED_FEATURES
from src.models import preprocessor


@dataclass
class ConceptFit:
    """Container for a shared preprocessor and site-specific models."""

    preprocessor: Pipeline
    models: dict
    feature_names: np.ndarray


def fit_site_models(
    data: pd.DataFrame,
    source_site: str,
    seed: int,
) -> ConceptFit:
    """Fit one preprocessor and one logistic model per site.

    The preprocessor is fitted only on the source site. Every site's
    features are then transformed with this single fitted object, and a
    logistic regression is fitted on the transformed features using
    that site's own labels.
    """
    source = data[data["site"] == source_site]

    transformer = preprocessor()
    transformer.fit(source[SHARED_FEATURES])
    feature_names = transformer.get_feature_names_out()

    models = {}
    for site, group in data.groupby("site"):
        features = transformer.transform(group[SHARED_FEATURES])
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
        )
        model.fit(features, group["target"].to_numpy())
        models[site] = model

    return ConceptFit(
        preprocessor=transformer,
        models=models,
        feature_names=feature_names,
    )


def coefficient_table(fit: ConceptFit) -> pd.DataFrame:
    """Return the coefficients of every site-specific model."""
    rows = []
    for site, model in fit.models.items():
        rows.append({
            "site": site,
            **dict(zip(fit.feature_names, model.coef_[0])),
        })
    return pd.DataFrame(rows).set_index("site")


def pairwise_coefficient_distance(
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """L2 distance between coefficient vectors of each pair of sites."""
    sites = coefficients.index.tolist()
    rows = []
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            distance = float(
                np.linalg.norm(
                    coefficients.loc[a].to_numpy()
                    - coefficients.loc[b].to_numpy()
                )
            )
            rows.append({
                "site_a": a,
                "site_b": b,
                "coefficient_distance": distance,
            })
    return pd.DataFrame(rows)


def prediction_divergence(
    fit: ConceptFit,
    data: pd.DataFrame,
    source_site: str,
) -> pd.DataFrame:
    """Mean absolute prediction difference between each pair of models
    on the source evaluation set.
    """
    source = data[data["site"] == source_site]
    features = fit.preprocessor.transform(source[SHARED_FEATURES])

    predictions = {
        site: model.predict_proba(features)[:, 1]
        for site, model in fit.models.items()
    }

    sites = list(predictions.keys())
    rows = []
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            diff = np.abs(predictions[a] - predictions[b]).mean()
            rows.append({
                "site_a": a,
                "site_b": b,
                "mean_absolute_prediction_difference": float(diff),
            })
    return pd.DataFrame(rows)
