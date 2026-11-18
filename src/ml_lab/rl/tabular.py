"""Tabular RL methods in NumPy."""

import numpy as np
import gymnasium as gym


def transition_model(env_id, **kwargs):
    """Return P, n_states, n_actions of a toy text env."""
    env = gym.make(env_id, **kwargs).unwrapped
    return env.P, env.observation_space.n, env.action_space.n


def policy_evaluation(P, policy, n_states, gamma=0.99, theta=1e-10):
    values = np.zeros(n_states)
    sweeps = 0
    while True:
        delta = 0.0
        for s in range(n_states):
            action = policy[s]
            new = sum(p * (r + gamma * values[s2] * (not done)) for p, s2, r, done in P[s][action])
            delta = max(delta, abs(new - values[s]))
            values[s] = new
        sweeps += 1
        if delta < theta:
            return values, sweeps


def q_from_values(P, values, n_states, n_actions, gamma=0.99):
    q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            q[s, a] = sum(p * (r + gamma * values[s2] * (not done)) for p, s2, r, done in P[s][a])
    return q


def value_iteration(P, n_states, n_actions, gamma=0.99, theta=1e-10):
    values = np.zeros(n_states)
    sweeps = 0
    while True:
        delta = 0.0
        for s in range(n_states):
            best = max(sum(p * (r + gamma * values[s2] * (not done)) for p, s2, r, done in P[s][a]) for a in range(n_actions))
            delta = max(delta, abs(best - values[s]))
            values[s] = best
        sweeps += 1
        if delta < theta:
            break
    policy = q_from_values(P, values, n_states, n_actions, gamma).argmax(axis=1)
    return values, policy, sweeps


def policy_iteration(P, n_states, n_actions, gamma=0.99):
    policy = np.zeros(n_states, dtype=int)
    rounds = 0
    while True:
        values, _ = policy_evaluation(P, policy, n_states, gamma)
        new_policy = q_from_values(P, values, n_states, n_actions, gamma).argmax(axis=1)
        rounds += 1
        if np.array_equal(new_policy, policy):
            return values, policy, rounds
        policy = new_policy


def epsilon_greedy(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def q_learning(env_id, episodes, alpha=0.5, gamma=0.99, epsilon=0.1, seed=0, decay=None, **env_kwargs):
    """Off policy TD control. Returns Q and the return of every episode."""
    env = gym.make(env_id, **env_kwargs)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = []
    state, _ = env.reset(seed=seed)
    for ep in range(episodes):
        eps = epsilon if decay is None else max(0.01, epsilon * decay ** ep)
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = epsilon_greedy(q[state], eps, rng)
            nxt, reward, term, trunc, _ = env.step(action)
            target = reward + gamma * q[nxt].max() * (not term)
            q[state, action] += alpha * (target - q[state, action])
            state, total, done = nxt, total + reward, term or trunc
        returns.append(total)
    env.close()
    return q, np.array(returns)


def sarsa(env_id, episodes, alpha=0.5, gamma=0.99, epsilon=0.1, seed=0, decay=None, **env_kwargs):
    """On policy TD control."""
    env = gym.make(env_id, **env_kwargs)
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = []
    env.reset(seed=seed)
    for ep in range(episodes):
        eps = epsilon if decay is None else max(0.01, epsilon * decay ** ep)
        state, _ = env.reset()
        action = epsilon_greedy(q[state], eps, rng)
        done, total = False, 0.0
        while not done:
            nxt, reward, term, trunc, _ = env.step(action)
            nxt_action = epsilon_greedy(q[nxt], eps, rng)
            target = reward + gamma * q[nxt, nxt_action] * (not term)
            q[state, action] += alpha * (target - q[state, action])
            state, action, total, done = nxt, nxt_action, total + reward, term or trunc
        returns.append(total)
    env.close()
    return q, np.array(returns)


def td0_prediction(env_id, policy, episodes, alpha=0.05, gamma=0.99, seed=0, **env_kwargs):
    """TD(0) value estimate of a fixed policy. Returns values after every episode."""
    env = gym.make(env_id, **env_kwargs)
    rng = np.random.default_rng(seed)
    values = np.zeros(env.observation_space.n)
    history = []
    env.reset(seed=seed)
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            nxt, reward, term, trunc, _ = env.step(int(policy[state]))
            values[state] += alpha * (reward + gamma * values[nxt] * (not term) - values[state])
            state, done = nxt, term or trunc
        history.append(values.copy())
    env.close()
    return values, np.array(history)


def greedy_success_rate(env_id, q, episodes=2000, seed=123, **env_kwargs):
    env = gym.make(env_id, **env_kwargs)
    wins = 0
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            state, reward, term, trunc, _ = env.step(int(q[state].argmax()))
            done = term or trunc
        wins += reward > 0
    env.close()
    return wins / episodes
