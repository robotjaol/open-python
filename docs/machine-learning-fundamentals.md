# Machine learning fundamentals

## Three problem types

| Type | Data | Goal | Example here |
|---|---|---|---|
| Supervised | Inputs and known targets | Predict the target for new inputs | Churn (classification), fuel economy (regression) |
| Unsupervised | Inputs only | Find structure | Customer segments, anomalies |
| Reinforcement | An agent acts in an environment and receives reward | Learn a policy that maximises return | CartPole, FrozenLake |

## The standard workflow

1. State the decision the model supports and the cost of each error.
2. Split data before looking at the test rows. Use 60/20/20 or cross validation.
3. Build a trivial baseline (predict the mean or the majority class). A model must beat it.
4. Fit preprocessing and the model on training rows only.
5. Choose settings on validation rows or inside cross validation.
6. Score the test rows once. Report the metric with its spread.
7. Look at errors, not only the score.

## Bias, variance and overfitting

- **Underfitting:** error is high on both training and validation rows. Use a more flexible model or better features.
- **Overfitting:** training error is low and validation error is high. Use more data, regularisation or a simpler model.
- A learning curve (score versus training size) shows which case you are in. Supervised notebooks include one.

## Leakage

Information from outside the training rows reaches the model. Common sources: scaling or imputing on all rows, selecting features on all rows (see `04_advanced_machine_learning/feature_selection`), duplicate rows across splits, identifiers that encode the target, future information in time data.

## Reading further

`hastie2009` and `sutton2018` in the source register (both **not verified** in the build environment).
