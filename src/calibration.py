"""Post-hoc calibration methods.

Two standard approaches are provided: Platt scaling, which fits a
logistic function to the scores of a trained model, and isotonic
regression, which fits a monotone step function. Both operate on the
output of an already-fitted classifier and are fitted on a held-out
set.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattScaler:
    """Logistic calibration on the logit of predicted probabilities."""

    def __init__(self):
        self._model = LogisticRegression(C=1e6, solver="lbfgs")

    @staticmethod
    def _logit(prob: np.ndarray) -> np.ndarray:
        eps = 1e-9
        prob = np.clip(prob, eps, 1.0 - eps)
        return np.log(prob / (1.0 - prob)).reshape(-1, 1)

    def fit(self, prob: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        self._model.fit(self._logit(prob), y_true)
        return self

    def transform(self, prob: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(self._logit(prob))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibration on predicted probabilities."""

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, prob: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(prob, y_true)
        return self

    def transform(self, prob: np.ndarray) -> np.ndarray:
        return self._model.predict(prob)


def apply_calibrators(
    source_prob: np.ndarray,
    source_true: np.ndarray,
    target_prob: np.ndarray,
) -> dict:
    """Fit each calibrator on source and apply to target."""
    platt = PlattScaler().fit(source_prob, source_true)
    isotonic = IsotonicCalibrator().fit(source_prob, source_true)
    return {
        "platt": platt.transform(target_prob),
        "isotonic": isotonic.transform(target_prob),
    }
