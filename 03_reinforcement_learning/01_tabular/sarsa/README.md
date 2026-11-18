# SARSA: Safe Paths on the Cliff

Learn the value of the policy you actually follow, including its exploration.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Intermediate |
| Dataset | `CliffWalking-v1` (see `datasets/catalog`) |
| Notebook | [`04_sarsa.ipynb`](04_sarsa.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- On policy TD control
- How exploration changes the learned path
- Comparing the online return with the greedy return
- Why SARSA picks the safer route

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 04_sarsa.ipynb
```

Results are saved to `experiments/results/sarsa.json`.
