"""Model definitions used across experiments.

Two model families are considered: a linear model with L2 penalty and
a gradient-boosted tree ensemble. The linear model serves as a
transparent baseline; the ensemble captures nonlinearities at the cost
of interpretability. Both are wrapped in a small pipeline that handles
categorical and numerical features consistently.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import CATEGORICAL_FEATURES, SHARED_FEATURES


NUMERIC_FEATURES = [
    column for column in SHARED_FEATURES
    if column not in CATEGORICAL_FEATURES
]


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


@dataclass
class ModelSpec:
    name: str
    estimator: object
    params: dict = field(default_factory=dict)


def logistic_regression(seed: int = 0) -> ModelSpec:
    return ModelSpec(
        name="logreg",
        estimator=LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=seed,
        ),
    )


def gradient_boosting(seed: int = 0) -> ModelSpec:
    return ModelSpec(
        name="gbm",
        estimator=GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        ),
    )


def build_pipeline(spec: ModelSpec) -> Pipeline:
    return Pipeline([
        ("preprocess", _preprocessor()),
        ("model", spec.estimator),
    ])
