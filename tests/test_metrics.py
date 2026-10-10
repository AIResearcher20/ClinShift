"""Unit tests for metric implementations."""

import numpy as np
import pytest

from src.metrics import (
    auroc,
    brier,
    expected_calibration_error,
    reliability_curve,
)


def test_auroc_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(y_true, y_prob) == pytest.approx(1.0)


def test_auroc_chance():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    assert auroc(y_true, y_prob) == pytest.approx(0.5)


def test_brier_perfect_prediction():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.0, 1.0, 1.0, 0.0])
    assert brier(y_true, y_prob) == pytest.approx(0.0)


def test_ece_perfectly_calibrated():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)


def test_ece_maximum_miscalibration():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([1.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(1.0)


def test_ece_rejects_out_of_range_probabilities():
    y_true = np.array([0, 1])
    y_prob = np.array([-0.1, 1.1])
    with pytest.raises(ValueError):
        expected_calibration_error(y_true, y_prob)


def test_reliability_curve_shape():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.linspace(0.05, 0.95, 10)
    centers, mean_prob, obs_rate = reliability_curve(
        y_true, y_prob, n_bins=5
    )
    assert centers.shape == (5,)
    assert mean_prob.shape == (5,)
    assert obs_rate.shape == (5,)
