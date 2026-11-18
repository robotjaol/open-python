# Deep Q-Network: Balancing CartPole

Approximate Q values with a neural network, experience replay and a target network.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Advanced |
| Dataset | `CartPole-v1` (see `datasets/catalog`) |
| Notebook | [`06_dqn.ipynb`](06_dqn.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- Why a table cannot hold continuous states
- How replay and target networks stabilise training
- Epsilon schedules
- How to read noisy RL learning curves

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 06_dqn.ipynb
```

Results are saved to `experiments/results/dqn.json`.
