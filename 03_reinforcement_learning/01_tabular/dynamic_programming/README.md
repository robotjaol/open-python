# Dynamic Programming: Planning in FrozenLake

Solve a known MDP exactly with policy iteration and value iteration.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Beginner |
| Dataset | `FrozenLake-v1` (see `datasets/catalog`) |
| Notebook | [`01_dynamic_programming.ipynb`](01_dynamic_programming.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- The Bellman optimality equation
- Policy evaluation, improvement and iteration
- Why DP needs a model of the environment
- How slippery transitions change the best policy

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 01_dynamic_programming.ipynb
```

Results are saved to `experiments/results/dynamic_programming.json`.
