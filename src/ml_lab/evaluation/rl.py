"""Evaluation helpers for RL."""

import numpy as np


def evaluate_policy(env_factory, policy, episodes=20, seed=0, success_threshold=None):
    """Run policy and collect returns."""
    returns, lengths = [], []
    for episode in range(episodes):
        env = env_factory()
        state, _ = env.reset(seed=seed + episode)
        done, total, steps = False, 0.0, 0
        while not done:
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        env.close()
        returns.append(total)
        lengths.append(steps)
    returns = np.array(returns)
    summary = {
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "std_return": float(returns.std()),
        "mean_length": float(np.mean(lengths)),
    }
    if success_threshold is not None:
        summary["success_rate"] = float((returns >= success_threshold).mean())
    return summary


def aggregate_seeds(curves):
    """Stack curves from seeds."""
    length = min(len(c) for c in curves)
    stacked = np.array([c[:length] for c in curves])
    return stacked.mean(axis=0), stacked.std(axis=0), stacked


def smooth(values, window=10):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")
