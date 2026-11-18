# Double DQN: Measuring Overestimation

Separate action choice from action evaluation to reduce optimistic value estimates.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Advanced |
| Dataset | `CartPole-v1` (see `datasets/catalog`) |
| Notebook | [`07_double_dqn.ipynb`](07_double_dqn.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- Maximisation bias
- The Double DQN target
- How to measure overestimation against an upper bound
- Whether a better value estimate means a better policy

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 07_double_dqn.ipynb
```

Results are saved to `experiments/results/double_dqn.json`.
