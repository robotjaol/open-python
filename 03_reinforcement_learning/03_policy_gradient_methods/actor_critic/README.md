# Actor-Critic: Learning a Baseline Online

Update the policy after every step using a learned value function as critic.

| Item | Value |
|---|---|
| Family | Reinforcement learning |
| Level | Advanced |
| Dataset | `CartPole-v1` (see `datasets/catalog`) |
| Notebook | [`09_actor_critic.ipynb`](09_actor_critic.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- The TD error as an advantage estimate
- Why a critic lowers variance but adds bias
- Two learning rates
- Instability of one step actor critic

## Run it

```bash
pip install -e ".[reinforcement-learning,development]"
jupyter lab 09_actor_critic.ipynb
```

Results are saved to `experiments/results/actor_critic.json`.
