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
the study and motivates the analysis of unlabeled shift indicators in
later sections.
