# Design decisions

This document records decisions that shape the analysis and that are
not obvious from the code alone. Each entry states what was decided,
on what evidence, and under what conditions the decision should be
revisited.

## Exclusion of cholesterol from the shared feature set

The UCI Heart Disease dataset contains a cholesterol measurement for
each patient. When comparing distributions across sites, the
standardized mean difference for cholesterol between Cleveland and
Switzerland is 6.74, and the Kolmogorov-Smirnov statistic is 1.0. A KS
statistic of 1.0 means the two samples do not overlap at any point of
the distribution. For comparison, the same statistic between Cleveland
and Hungary is 0.059, and between Cleveland and the VA site is 0.278.

A shift of this magnitude is not plausible as a physiological
difference between two patient populations. The more likely
explanation is that cholesterol was recorded in a different unit or
under a different measurement convention at the Switzerland site. The
original UCI documentation does not resolve this question.

Retaining cholesterol in the shared feature set would have two
undesirable consequences. First, the model would rely on a feature
whose values are not comparable across sites, which means any
degradation observed at the Switzerland site could be attributed to a
measurement artifact rather than a genuine distribution shift. Second,
the total variation distance and the standardized mean difference for
cholesterol would dominate the aggregate shift statistics, masking the
contribution of other features.

Cholesterol is therefore excluded from `SHARED_FEATURES`. This decision
affects all experiments. If the source dataset is later revised, or if
documentation of the recording convention at the Switzerland site
becomes available, the decision should be revisited and the analysis
repeated with cholesterol included.

## Treatment of isotonic calibration under distribution shift

Isotonic regression is one of two post-hoc calibration methods applied
in the transfer experiments. Across the initial runs, isotonic
calibration reduced AUROC on two of the three target sites when applied
to the gradient boosting model, while Platt scaling never altered
AUROC. The reduction reached 0.056 in the case of the Switzerland
target.

This behavior is consistent with a known property of isotonic
regression. When the calibration set is small, or when the target
distribution differs substantially from the source, isotonic regression
tends to collapse predicted probabilities into a small number of
distinct levels. Multiple target samples then receive the same
calibrated score, which degrades the ranking that AUROC depends on.

The decision is not to remove isotonic calibration, but to report it
alongside Platt scaling in all results. The failure mode itself is
treated as a finding: post-hoc calibration methods that are effective
in-distribution may behave unexpectedly when the evaluation
distribution differs from the calibration distribution. This is
relevant to any deployment setting where calibration is performed on a
source dataset and applied elsewhere.
