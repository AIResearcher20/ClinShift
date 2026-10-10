"""Descriptive comparison of learned decision boundaries across sites.

For each site, fit a logistic regression on a feature space defined by
a preprocessor fitted once on the source site. Report the distance
between coefficient vectors, the mean absolute difference between
predictions on the source evaluation set, and the full coefficient
table.

The measures are descriptive. They do not isolate concept divergence
from sample size, class balance, or measurement variation. The results
are stored alongside the covariate shift measures so that the two
views can be read together.
"""

import json
from pathlib import Path

from src.concept_shift import (
    coefficient_table,
    fit_site_models,
    pairwise_coefficient_distance,
    prediction_divergence,
)
from src.data_loader import load_all

SEED = 42
SOURCE_SITE = "cleveland"
RESULTS_DIR = Path("results")


def rows_to_records(frame, value_column: str) -> list:
    """Convert a pairwise table into a list of labeled records."""
    return [
        {
            "site_a": row.site_a,
            "site_b": row.site_b,
            value_column: getattr(row, value_column),
        }
        for row in frame.itertuples(index=False)
    ]


def main() -> None:
    data = load_all()

    fit = fit_site_models(data, SOURCE_SITE, SEED)

    coefficients = coefficient_table(fit)
    print("Coefficients by site (shared feature space)")
    print(coefficients.round(3).to_string())

    distances = pairwise_coefficient_distance(coefficients)
    print("\nPairwise coefficient distance")
    print(distances.round(3).to_string(index=False))

    divergence = prediction_divergence(fit, data, SOURCE_SITE)
    print("\nMean absolute prediction difference on source set")
    print(divergence.round(4).to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    coefficients.to_csv(RESULTS_DIR / "concept_coefficients.csv")
    distances.to_csv(RESULTS_DIR / "concept_pairwise.csv", index=False)
    divergence.to_csv(
        RESULTS_DIR / "concept_prediction_divergence.csv", index=False
    )

    summary = {
        "source_site": SOURCE_SITE,
        "coefficient_distance": rows_to_records(
            distances, "coefficient_distance"
        ),
        "prediction_divergence": rows_to_records(
            divergence, "mean_absolute_prediction_difference"
        ),
    }
    with (RESULTS_DIR / "concept_shift.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
