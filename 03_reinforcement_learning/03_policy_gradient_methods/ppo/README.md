# Proximal Policy Optimization on CartPole

Take several safe gradient steps per batch by clipping the policy ratio.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Advanced |
| Dataset | `CartPole-v1` (see `datasets/catalog`) |
| Notebook | [`10_ppo.ipynb`](10_ppo.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- The clipped surrogate objective
- Generalised advantage estimation
- Why several epochs per batch are safe with clipping
- How to log the clip fraction

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 10_ppo.ipynb
```

Results are saved to `experiments/results/ppo.json`.
