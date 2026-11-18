# Monte Carlo Control: Learning Blackjack from Episodes

Learn values by averaging complete episodes without a model.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Intermediate |
| Dataset | `Blackjack-v1` (see `datasets/catalog`) |
| Notebook | [`02_monte_carlo.ipynb`](02_monte_carlo.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- First visit Monte Carlo estimation
- Exploration with epsilon soft policies
- Why returns have high variance
- How to compare a policy with a baseline using a confidence interval

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 02_monte_carlo.ipynb
```

Results are saved to `experiments/results/monte_carlo.json`.
