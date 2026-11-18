"""Deep RL agents in PyTorch. Written from scratch, no RL library."""

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))


def to_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class ReplayBuffer:
    def __init__(self, size, obs_dim):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.nxt = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros(size, dtype=np.int64)
        self.rew = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.size, self.pos, self.full = size, 0, False

    def add(self, o, a, r, n, d):
        i = self.pos
        self.obs[i], self.act[i], self.rew[i], self.nxt[i], self.done[i] = o, a, r, n, d
        self.pos = (self.pos + 1) % self.size
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.size if self.full else self.pos

    def sample(self, batch, rng):
        idx = rng.integers(0, len(self), batch)
        return (to_tensor(self.obs[idx]), torch.as_tensor(self.act[idx]), to_tensor(self.rew[idx]),
                to_tensor(self.nxt[idx]), to_tensor(self.done[idx]))


def greedy_policy(model):
    def policy(state):
        with torch.no_grad():
            return int(model(to_tensor(state)).argmax())
    return policy


def train_dqn(env_id="CartPole-v1", seed=0, total_steps=20000, double=False, lr=1e-3, gamma=0.99, buffer_size=20000,
              batch=64, target_every=250, eps_start=1.0, eps_end=0.05, eps_decay_steps=5000, learn_start=500, hidden=64):
    """DQN, or Double DQN when double=True. Bootstraps through time limit truncation."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q, q_target = mlp(obs_dim, n_actions, hidden), None
    q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, obs_dim)
    obs, _ = env.reset(seed=seed)
    returns, steps, q_log, total = [], [], [], 0.0
    for t in range(total_steps):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * t / eps_decay_steps)
        if rng.random() < eps:
            action = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                action = int(q(to_tensor(obs)).argmax())
        nxt, reward, term, trunc, _ = env.step(action)
        buffer.add(obs, action, reward, nxt, float(term))
        obs, total = nxt, total + reward
        if t >= learn_start:
            o, a, r, n, d = buffer.sample(batch, rng)
            with torch.no_grad():
                if double:
                    best = q(n).argmax(dim=1, keepdim=True)
                    next_q = q_target(n).gather(1, best).squeeze(1)
                else:
                    next_q = q_target(n).max(dim=1).values
                target = r + gamma * (1 - d) * next_q
            pred = q(o).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            if t % 500 == 0:
                with torch.no_grad():
                    q_log.append((t, float(q(o).max(dim=1).values.mean())))
        if t % target_every == 0:
            q_target.load_state_dict(q.state_dict())
        if term or trunc:
            returns.append(total)
            steps.append(t + 1)
            obs, _ = env.reset()
            total = 0.0
    env.close()
    return dict(returns=returns, steps=steps, policy=greedy_policy(q), model=q, q_log=q_log)


def categorical(model, state):
    logits = model(to_tensor(state))
    return torch.distributions.Categorical(logits=logits)


def argmax_policy(model):
    def policy(state):
        with torch.no_grad():
            return int(model(to_tensor(state)).argmax())
    return policy


def train_reinforce(env_id="CartPole-v1", seed=0, episodes=400, lr=5e-3, gamma=0.99, baseline="norm", hidden=64):
    """Monte Carlo policy gradient. baseline is 'none' or 'norm' (return normalisation)."""
    seed_everything(seed)
    env = gym.make(env_id)
    policy_net = mlp(env.observation_space.shape[0], env.action_space.n, hidden)
    opt = torch.optim.Adam(policy_net.parameters(), lr=lr)
    returns, steps, t = [], [], 0
    first = True
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed if first else None)
        first = False
        logps, rewards, done = [], [], False
        while not done:
            dist = categorical(policy_net, obs)
            action = dist.sample()
            obs, reward, term, trunc, _ = env.step(int(action))
            logps.append(dist.log_prob(action))
            rewards.append(reward)
            done = term or trunc
            t += 1
        g, discounted = 0.0, []
        for r in reversed(rewards):
            g = r + gamma * g
            discounted.append(g)
        g = torch.tensor(discounted[::-1], dtype=torch.float32)
        if baseline == "norm":
            g = (g - g.mean()) / (g.std() + 1e-8)
        loss = -(torch.stack(logps) * g).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        returns.append(sum(rewards))
        steps.append(t)
    env.close()
    return dict(returns=returns, steps=steps, policy=argmax_policy(policy_net), model=policy_net)


def train_actor_critic(env_id="CartPole-v1", seed=0, total_steps=60000, lr_actor=1e-3, lr_critic=5e-3, gamma=0.99, hidden=64):
    """One step actor critic with a learned state value baseline."""
    seed_everything(seed)
    env = gym.make(env_id)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    actor, critic = mlp(obs_dim, n_actions, hidden), mlp(obs_dim, 1, hidden)
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    obs, _ = env.reset(seed=seed)
    returns, steps, total = [], [], 0.0
    for t in range(total_steps):
        dist = categorical(actor, obs)
        action = dist.sample()
        nxt, reward, term, trunc, _ = env.step(int(action))
        value = critic(to_tensor(obs)).squeeze()
        with torch.no_grad():
            next_value = 0.0 if term else critic(to_tensor(nxt)).squeeze()
        td_error = reward + gamma * next_value - value
        loss_c = td_error.pow(2)
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()
        loss_a = -dist.log_prob(action) * td_error.detach()
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()
        obs, total = nxt, total + reward
        if term or trunc:
            returns.append(total)
            steps.append(t + 1)
            obs, _ = env.reset()
            total = 0.0
    env.close()
    return dict(returns=returns, steps=steps, policy=argmax_policy(actor), model=actor)


def train_ppo(env_id="CartPole-v1", seed=0, total_steps=60000, rollout=1024, epochs=10, minibatch=128, lr=1e-3,
              gamma=0.99, lam=0.95, clip=0.2, vf_coef=0.5, ent_coef=0.0, hidden=64):
    """PPO with clipped objective and GAE. Truncation is treated as episode end (small bias)."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    actor, critic = mlp(obs_dim, n_actions, hidden), mlp(obs_dim, 1, hidden)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    obs, _ = env.reset(seed=seed)
    returns, steps, total, t = [], [], 0.0, 0
    clip_log = []
    while t < total_steps:
        O, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(rollout):
            with torch.no_grad():
                dist = categorical(actor, obs)
                action = dist.sample()
                value = critic(to_tensor(obs)).squeeze()
            nxt, reward, term, trunc, _ = env.step(int(action))
            O.append(obs); A.append(int(action)); LP.append(float(dist.log_prob(action))); R.append(reward)
            D.append(float(term or trunc)); V.append(float(value))
            obs, total, t = nxt, total + reward, t + 1
            if term or trunc:
                returns.append(total)
                steps.append(t)
                obs, _ = env.reset()
                total = 0.0
        with torch.no_grad():
            last = float(critic(to_tensor(obs)).squeeze())
        adv, gae = np.zeros(rollout, dtype=np.float32), 0.0
        for i in reversed(range(rollout)):
            next_v = last if i == rollout - 1 else V[i + 1]
            delta = R[i] + gamma * next_v * (1 - D[i]) - V[i]
            gae = delta + gamma * lam * (1 - D[i]) * gae
            adv[i] = gae
        ret = adv + np.array(V, dtype=np.float32)
        O, A, LP = to_tensor(O), torch.as_tensor(A), to_tensor(LP)
        adv_t, ret_t = to_tensor(adv), to_tensor(ret)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        clipped = []
        for _ in range(epochs):
            perm = rng.permutation(rollout)
            for start in range(0, rollout, minibatch):
                idx = torch.as_tensor(perm[start:start + minibatch])
                dist = torch.distributions.Categorical(logits=actor(O[idx]))
                ratio = torch.exp(dist.log_prob(A[idx]) - LP[idx])
                surrogate = torch.min(ratio * adv_t[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[idx])
                loss = (-surrogate.mean() + vf_coef * (critic(O[idx]).squeeze(1) - ret_t[idx]).pow(2).mean()
                        - ent_coef * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
                clipped.append(float(((ratio - 1).abs() > clip).float().mean()))
        clip_log.append((t, float(np.mean(clipped))))
    env.close()
    return dict(returns=returns, steps=steps, policy=argmax_policy(actor), model=actor, clip_log=clip_log)


def curve_on_grid(steps, returns, grid, window=10):
    """Mean of the last `window` finished episodes at each grid step."""
    steps, returns = np.asarray(steps), np.asarray(returns, dtype=float)
    out = []
    for g in grid:
        k = np.searchsorted(steps, g, side="right")
        out.append(returns[max(0, k - window):k].mean() if k > 0 else np.nan)
    return np.array(out)
