# Polynomial Regression: Fuel Economy versus Horsepower

Use one feature to see underfitting and overfitting as the polynomial degree grows.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Beginner |
| Dataset | `auto_mpg` (see `datasets/catalog`) |
| Notebook | [`02_polynomial_regression.ipynb`](02_polynomial_regression.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How polynomial features turn a linear model into a curve
- How the degree trades bias against variance
- How to pick the degree on validation data and not on the test set

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 02_polynomial_regression.ipynb
```

Results are saved to `experiments/results/polynomial_regression.json`.
