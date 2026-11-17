# Elastic Net: Mixing L1 and L2 Penalties

Balance sparsity and stability when predictors are correlated.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `diabetes` (see `datasets/catalog`) |
| Notebook | [`05_elastic_net.ipynb`](05_elastic_net.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How the mixing ratio blends lasso and ridge
- Why the elastic net handles correlated groups better than lasso
- How to tune two parameters sensibly

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 05_elastic_net.ipynb
```

Results are saved to `experiments/results/elastic_net.json`.
