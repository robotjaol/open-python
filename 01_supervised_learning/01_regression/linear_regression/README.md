# Linear Regression: Predicting Vehicle Fuel Economy

Fit and diagnose ordinary least squares on real car data, from the normal equations to residual checks.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Beginner |
| Dataset | `auto_mpg` (see `datasets/catalog`) |
| Notebook | [`01_linear_regression.ipynb`](01_linear_regression.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How least squares chooses coefficients and why the normal equations work
- Why we split the data before scaling and fitting
- How to read RMSE, MAE and R-squared and compare them with a dummy baseline
- How collinearity and nonlinearity show up in the diagnostics

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 01_linear_regression.ipynb
```

Results are saved to `experiments/results/linear_regression.json`.
