# Lasso Regression: Sparse Models for Disease Progression

Use the L1 penalty to set coefficients to exactly zero and select features.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `diabetes` (see `datasets/catalog`) |
| Notebook | [`04_lasso_regression.ipynb`](04_lasso_regression.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How the L1 penalty creates sparse models
- How coordinate descent and soft thresholding work
- How to count selected features along the path

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 04_lasso_regression.ipynb
```

Results are saved to `experiments/results/lasso_regression.json`.
