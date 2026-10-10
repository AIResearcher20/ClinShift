# Design decisions

This document records decisions that shape the analysis and that are
not obvious from the code alone. Each entry states what was decided,
on what evidence, and under what conditions the decision should be
revisited.

## Data source

The analysis uses the original UCI Heart Disease archive rather than
the Kaggle redistribution. The archive contains the four sites in the
format released by the UCI machine learning repository. This choice
avoids any preprocessing or unit conversion that may have been applied
in later redistributions.

## Exclusion of cholesterol from the shared feature set

Inspection of the raw UCI archive revealed that all 123 cholesterol
values recorded at the Switzerland site are zero. These are stored as
numerical values rather than missing entries, so a conventional
missing-value treatment does not detect them. Including cholesterol in
the shared feature set produces a standardized mean difference of 6.74
between Cleveland and Switzerland and a Kolmogorov-Smirnov statistic of
1.0, indicating no overlap between the two distributions.

The Switzerland site is not the only one with zero-valued cholesterol.
The VA site contains a smaller number of zero values that are worth
investigating separately, but the pattern there is not uniform and does
not exclude the site from analysis.

Cholesterol is therefore excluded from `SHARED_FEATURES` and from the
main cross-site comparison. The raw column is retained in the loaded
data frame so that the effect of this decision can be examined in a
later sensitivity analysis.

## Baseline versus transfer performance

Within-site cross-validation shows that the four sites differ
substantially in intrinsic difficulty. With logistic regression,
Cleveland and Hungary reach AUROC around 0.88-0.90, while Switzerland
and VA reach 0.70-0.73. The Switzerland site has a positive rate of
93.5% and substantial missingness in several features. The VA site has
missingness above 25% in several features.

Cross-site transfer from Cleveland does not uniformly reduce AUROC. In
Switzerland and VA, the transferred model achieves higher AUROC than a
model trained on the target site alone. This reversal is consistent
with the lower intrinsic quality of these two sites: a model trained
on the cleaner Cleveland data generalizes better than a model
overfitting to noisier target data.

Calibration does not follow the same pattern. In Switzerland, the
transferred model's ECE rises from 0.06 to 0.29 while AUROC rises by
0.05. The two metrics move in opposite directions. This decoupling
between discrimination and calibration is the central observation of
the study and motivates the analysis of unlabeled shift indicators.

## Treatment of isotonic calibration under distribution shift

Isotonic regression is one of two post-hoc calibration methods applied
in the transfer experiments. Isotonic calibration was observed to
reduce AUROC on some target sites when applied to the gradient boosting
model, while Platt scaling never altered AUROC. This behavior is
consistent with the known tendency of isotonic regression to collapse
predicted probabilities into a small number of levels when the
calibration set is small or when the target distribution differs
substantially from the source. Both methods are reported in the
results, and the failure mode is documented as a finding.

## Unlabeled shift indicators and calibration degradation

Two classes of unlabeled shift indicators were compared to the change
in Expected Calibration Error when a source-trained model was evaluated
on a target site.

Feature-based indicators quantify differences in the marginal
distribution of features. They include the AUROC of a classifier
trained to distinguish source from target, Maximum Mean Discrepancy
with an RBF kernel, and energy distance.

Prediction-based indicators quantify differences in the output
distribution of the source-trained model. They include Jensen-Shannon
divergence between predicted probability histograms, the difference in
mean predicted probability, and the difference in mean predictive
entropy.

The feature-based indicators reached a Spearman correlation of 0.5
with the change in ECE. The prediction-based indicators reached a
Spearman correlation of 1.0 with the same quantity. The difference is
consistent with what each class measures: prediction-based indicators
are computed from the model output and are therefore more directly
related to calibration.

With three target sites, a Spearman correlation of 1.0 has a
non-negligible probability of arising by chance under the null
hypothesis of no association. The result is therefore descriptive
rather than inferential. Replication on datasets with a larger number
of sites is required before these indicators can be recommended for
deployment.

## Source of out-of-fold predictions on Cleveland

The source probabilities used in the prediction-based indicators are
out-of-fold predictions from a five-fold cross-validation on
Cleveland. Each sample is predicted by a model that did not see it
during training. The target probabilities, by contrast, come from a
single model fitted on the full Cleveland site. The two sets therefore
come from slightly different models. This discrepancy is expected to
be small relative to the site-level differences that the indicators
are designed to measure, and it is documented here so that the
distinction is not lost.

## Descriptive analysis of concept divergence

Coefficient vectors of logistic regression models fitted
independently at each site, on a shared feature space defined by a
preprocessor fitted on Cleveland, have pairwise Euclidean distances
between 1.78 and 2.47. The mean absolute difference between the
predictions of these models on the Cleveland evaluation set ranges
from 0.11 to 0.37. The two measures agree on which pairs of sites are
most divergent.

Switzerland appears in all three of the largest pairwise distances in
both measures. The sign of the coefficient for sex reverses at
Switzerland relative to the other three sites. This reversal is
consistent with the site's positive rate of 93.5%, which changes the
interpretation of sex as a predictor in the sample.

For the three pairs of the form (Cleveland, target), prediction
divergence aligns with the change in ECE observed in the transfer
experiments: Hungary has the smallest divergence and the smallest
change in calibration, Switzerland has the largest of both, and VA
falls in between.

These measures are descriptive. They do not establish that concept
shift is the cause of calibration degradation, and they do not
separate concept shift from covariate shift. With three target sites,
no inferential claim is made. The measures are reported alongside the
covariate shift analysis so that the two views can be read together.




