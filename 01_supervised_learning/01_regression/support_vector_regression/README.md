# Support Vector Regression: Fuel Economy with an Error Tube

Fit a flexible kernel regressor that ignores small errors.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `auto_mpg` (see `datasets/catalog`) |
| Notebook | [`06_support_vector_regression.ipynb`](06_support_vector_regression.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- What the epsilon insensitive loss is
- How the RBF kernel gives nonlinear fits
- How C, epsilon and gamma interact

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 06_support_vector_regression.ipynb
```

Results are saved to `experiments/results/support_vector_regression.json`.
