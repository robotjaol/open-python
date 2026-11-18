# Temporal Difference Learning: TD(0) Value Estimation

Update value estimates after every step by bootstrapping.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Intermediate |
| Dataset | `FrozenLake-v1` (see `datasets/catalog`) |
| Notebook | [`03_temporal_difference.ipynb`](03_temporal_difference.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- The TD error
- Bootstrapping versus Monte Carlo
- The effect of the step size
- Bias and variance of the two estimators

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 03_temporal_difference.ipynb
```

Results are saved to `experiments/results/temporal_difference.json`.
