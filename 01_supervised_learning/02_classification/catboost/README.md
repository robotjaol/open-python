# CatBoost: Boosting with Categorical Features

Use ordered boosting and ordered target statistics to treat categories safely.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `telco_churn` (see `datasets/catalog`) |
| Notebook | [`09_catboost.ipynb`](09_catboost.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- Why target encoding can leak and how ordered statistics avoid it
- How symmetric trees work
- How to pass categorical columns natively

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 09_catboost.ipynb
```

Results are saved to `experiments/results/catboost.json`.
