# Autonomous Control: Balancing a Cart-Pole with a Heuristic and PPO

Compare a hand written controller, a random policy and PPO, then test robustness to noise.

- Notebook: [`autonomous_control.ipynb`](autonomous_control.ipynb)
- Dataset: `CartPole-v1`
- Results: `experiments/results/autonomous_control.json`

## Problem

Keep a pole balanced on a cart for 500 steps, and find out how robust each controller is to sensor and actuator noise.

## Limits

Three seeds and a small training budget. A simple heuristic may beat PPO or not. The table shows which on this setup. The simulator is noise free by default and real systems have delay, friction and sensor faults that are not modelled.
