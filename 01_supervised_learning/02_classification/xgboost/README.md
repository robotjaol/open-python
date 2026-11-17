# XGBoost: Regularised Boosting for Churn

Use second order boosting with built-in regularisation and early stopping.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `telco_churn` (see `datasets/catalog`) |
| Notebook | [`07_xgboost.ipynb`](07_xgboost.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- What makes XGBoost different from plain gradient boosting
- How regularisation terms enter the split gain
- How to use early stopping without leaking the test set

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 07_xgboost.ipynb
```

Results are saved to `experiments/results/xgboost.json`.
