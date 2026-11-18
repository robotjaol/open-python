# Statistical foundations

## Estimating performance honestly

- A single test score is one draw. With 114 test rows (breast cancer project), a sensitivity of 0.95 can have a 95 percent bootstrap interval several points wide. Report it.
- Cross validation averages over splits. Report the mean and the standard deviation over folds. The fold standard deviation understates the true uncertainty because folds share data.
- Comparing two models on the same rows: use paired differences, not two separate intervals.

## Multiple comparisons and selection bias

The best of many candidates is biased upward. Choose on validation rows, report on test rows. Use nested cross validation when you tune and report on the same small dataset (`04_advanced_machine_learning/hyperparameter_optimization`).

## Class imbalance

Accuracy rewards predicting the majority class. Use PR-AUC, recall at a fixed precision, or a cost based threshold. The baseline PR-AUC of a random ranker equals the positive share.

## Correlation, association and causation

All supervised models here describe associations in observed data. A feature that predicts churn is not shown to cause it. Acting on it needs an experiment.

## Effect sizes and noise

When two models differ by less than the fold standard deviation, treat them as tied. State this in the notebook rather than naming a winner.
