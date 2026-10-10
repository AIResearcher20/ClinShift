"""Unlabeled indicators computed from model predictions.

These indicators describe differences between the predicted probability
distributions of a source-trained model on two sites. They do not use
outcome labels at the target site. Their purpose is to complement the
feature-based indicators in shift_detection.py with signals that are
derived from the model's own behavior.
"""

import numpy as np


# Probability clipping range used before entropy and divergence
# computations. This prevents log(0) and keeps the estimates finite
# without changing the ordering of the probabilities.
PROB_EPS = 1e-7

# Smoothing added to histogram bins before computing Jensen-Shannon
# divergence. The value is small relative to 1 / n_bins and is applied
# uniformly, so it does not bias the comparison between sites.
HIST_EPS = 1e-9

# Shared number of bins for the histogram used in Jensen-Shannon
# divergence. The choice is documented in the analysis because the
# result is sensitive to this parameter.
JS_BINS = 50


def _clip(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob, PROB_EPS, 1.0 - PROB_EPS)


def prediction_js(
    source_prob: np.ndarray,
    target_prob: np.ndarray,
    n_bins: int = JS_BINS,
) -> float:
    """Jensen-Shannon divergence between two predicted probability
    distributions.

    Both arrays are discretized onto the same set of bins over [0, 1]
    before the divergence is computed. Using a shared bin grid is
    required for the comparison to be meaningful. A small uniform
    smoothing is applied to avoid zero-probability bins.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    source_hist, _ = np.histogram(source_prob, bins=edges)
    target_hist, _ = np.histogram(target_prob, bins=edges)

    source_density = source_hist.astype(float) + HIST_EPS
    target_density = target_hist.astype(float) + HIST_EPS

    source_density /= source_density.sum()
    target_density /= target_density.sum()

    mixture = 0.5 * (source_density + target_density)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)))

    js = 0.5 * _kl(source_density, mixture) + 0.5 * _kl(
        target_density, mixture
    )
    return float(js)


def confidence_shift(
    source_prob: np.ndarray,
    target_prob: np.ndarray,
) -> float:
    """Difference in mean predicted probability between target and
    source.

    A positive value indicates that the model is on average more
    confident on the target site than on the source site. The
    interpretation depends on the class balance at the target, which is
    not used here.
    """
    return float(target_prob.mean() - source_prob.mean())


def entropy_difference(
    source_prob: np.ndarray,
    target_prob: np.ndarray,
) -> float:
    """Difference in mean predictive entropy between target and source.

    Entropy is computed for a Bernoulli predictive distribution from
    the clipped probability. Higher entropy indicates less certainty
    in the model's predictions.
    """
    def _bernoulli_entropy(prob: np.ndarray) -> np.ndarray:
        prob = _clip(prob)
        return -(prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob))

    return float(
        _bernoulli_entropy(target_prob).mean()
        - _bernoulli_entropy(source_prob).mean()
    )


def compute_prediction_indicators(
    source_prob: np.ndarray,
    target_prob: np.ndarray,
) -> dict:
    """Compute every prediction-based indicator for one pair of sites."""
    return {
        "prediction_js": prediction_js(source_prob, target_prob),
        "confidence_shift": confidence_shift(source_prob, target_prob),
        "entropy_difference": entropy_difference(
            source_prob, target_prob
        ),
    }
