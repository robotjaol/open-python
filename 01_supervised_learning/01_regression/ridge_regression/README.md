# Ridge Regression: Disease Progression with Correlated Predictors

Shrink coefficients with an L2 penalty and watch the bias variance trade-off on diabetes data.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `diabetes` (see `datasets/catalog`) |
| Notebook | [`03_ridge_regression.ipynb`](03_ridge_regression.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- What the L2 penalty does to coefficients
- Why scaling matters before penalised regression
- How to read a coefficient path

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 03_ridge_regression.ipynb
```

Results are saved to `experiments/results/ridge_regression.json`.
