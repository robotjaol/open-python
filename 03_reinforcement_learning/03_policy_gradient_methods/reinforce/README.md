# REINFORCE: Policy Gradient on CartPole

Adjust a stochastic policy directly in the direction of higher return.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Advanced |
| Dataset | `CartPole-v1` (see `datasets/catalog`) |
| Notebook | [`08_reinforce.ipynb`](08_reinforce.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- The policy gradient theorem
- Monte Carlo returns as the learning signal
- Why a baseline cuts variance
- Why episode based updates are sample inefficient

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 08_reinforce.ipynb
```

Results are saved to `experiments/results/reinforce.json`.
