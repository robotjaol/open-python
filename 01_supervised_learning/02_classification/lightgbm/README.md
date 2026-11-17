# LightGBM: Fast Leaf-Wise Boosting for Churn

Grow trees leaf by leaf on histogram bins and handle categories natively.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `telco_churn` (see `datasets/catalog`) |
| Notebook | [`08_lightgbm.ipynb`](08_lightgbm.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How leaf-wise growth differs from level-wise growth
- Why num_leaves is the main complexity control
- How native categorical features compare with one-hot encoding

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 08_lightgbm.ipynb
```

Results are saved to `experiments/results/lightgbm.json`.
