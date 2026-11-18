# Q-Learning: Off-Policy Control on a Slippery Lake

Learn the optimal action values while behaving exploratively.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Intermediate |
| Dataset | `FrozenLake-v1` (see `datasets/catalog`) |
| Notebook | [`05_q_learning.ipynb`](05_q_learning.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- The Q-learning update and the max operator
- Why Q-learning is off policy
- Exploration schedules
- How to compare a learned policy with the DP optimum

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 05_q_learning.ipynb
```

Results are saved to `experiments/results/q_learning.json`.
