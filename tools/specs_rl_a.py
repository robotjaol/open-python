"""RL specs, part A (tabular)."""

RL = []
BASE = "03_reinforcement_learning"

RL_SETUP = '''import warnings
warnings.filterwarnings("ignore")

import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from IPython.display import display

from ml_lab.rl import tabular, deep
from ml_lab.evaluation import evaluate_policy
from ml_lab.experiments import ExperimentRecord, save_figure
from ml_lab.utils import set_seed, print_environment
from ml_lab import visualization as viz

NOTEBOOK_ID = "__ID__"
SEED = 42
set_seed(SEED)
viz.apply_style()
print_environment()
print("gymnasium", gym.__version__)'''


def rl(num, sub, **kw):
    spec = dict(num=num, level="Intermediate", minutes=1, extras="reinforcement-learning", task="control", setup=RL_SETUP,
                skills=["Run several seeds", "Report learning curves with spread", "Compare with a baseline policy"],
                prereq=["Probability basics", "Markov decision processes"],
                exercises=["Repeat with a different set of seeds and report the spread.",
                           "Change the discount factor and describe what happens to the learned behaviour.",
                           "Change one hyperparameter by a factor of 10 and compare learning curves."],
                eval_md="Return is the sum of rewards in an episode. We report the mean return of the final policy over fresh evaluation episodes "
                        "(seeds not used in training), the spread across training seeds, and the same number for a random policy as a floor. "
                        "RL results change a lot between seeds, so every headline number comes with its spread.",
                lib_md="The algorithm is implemented in `src/ml_lab/rl` without an RL library. Gymnasium supplies only the environment.",
                limits_extra="Few seeds (see the notebook), small environments, and no hyperparameter search beyond the experiments shown.")
    spec.update(kw)
    spec["folder"] = f"{BASE}/{sub}/{spec['id']}"
    RL.append(spec)


TAB = "01_tabular"

rl("01", TAB, id="dynamic_programming", title="Dynamic Programming: Planning in FrozenLake", level="Beginner",
   tagline="Solve a known MDP exactly with policy iteration and value iteration.", name="ValueIteration", dataset="FrozenLake-v1",
   objectives=["The Bellman optimality equation", "Policy evaluation, improvement and iteration", "Why DP needs a model of the environment", "How slippery transitions change the best policy"],
   problem=dict(context="A robot crosses a frozen lake where the ice is slippery, so a move sometimes goes sideways.",
                problem="Find the policy that reaches the goal most often when the transition probabilities are known.",
                matters="Planning with a model is the reference that learning methods are measured against.",
                why_ml="The best action depends on uncertain outcomes several steps ahead.",
                why_algo="DP gives exact answers for small known MDPs."),
   papers=["sutton2018", "towers2024"],
   paper_relation="Sutton and Barto (not verified in the build environment, check before citing) present policy and value iteration. Towers et al. describe the Gymnasium API used for the environment.",
   math=r"""**Bellman expectation.** $V^\pi(s) = \sum_a \pi(a|s)\sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$.

**Bellman optimality.** $V^*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V^*(s')]$.

**Value iteration.** Apply the optimality backup until the largest change is below a threshold $\theta$.

**Policy iteration.** Alternate full policy evaluation and greedy improvement. It stops when the policy no longer changes, after a finite number of rounds.

**Limits.** Needs $p(s',r|s,a)$, cost grows with the number of states.""",
   dataset_md="**FrozenLake-v1** (Gymnasium, `toy_text`), 4 by 4 grid with 16 states and 4 actions. Reward is 1 only on reaching the goal. With `is_slippery=True` the agent moves in the intended direction with probability 1/3 and sideways otherwise. There is no dataset: the environment is simulated. The transition model is read from `env.unwrapped.P`. We also use the 8 by 8 map.",
   eda="""env = gym.make("FrozenLake-v1", is_slippery=True)
print("observation space:", env.observation_space, "| action space:", env.action_space)
print(env.unwrapped.desc.astype(str))
P, nS, nA = tabular.transition_model("FrozenLake-v1", is_slippery=True)
print("transitions from state 0, action 'right' (probability, next, reward, done):", P[0][2])""",
   prep_md="The environment is deterministic in its rules but random in its outcomes. We read the transition model once and use the discount factor 0.99 for planning. For the success probability we also evaluate with discount 1, which makes the value of the start state equal to the probability of reaching the goal.",
   prep='''GAMMA = 0.99
P, nS, nA = tabular.transition_model("FrozenLake-v1", is_slippery=True)
print("states:", nS, "| actions:", nA)''',
   concept="Because the model is known, we do not need to interact. We sweep over the states and back up values until they stop changing.",
   scratch_md="The implementation is in `ml_lab.rl.tabular`. We print the source of value iteration so that it can be read here.",
   scratch='''print(inspect.getsource(tabular.value_iteration))''',
   params="| Parameter | Meaning |\n|---|---|\n| `gamma` | Discount factor. |\n| `theta` | Stopping threshold for value change. |\n| `is_slippery` | Whether moves are random. |",
   train_md="Run value iteration and policy iteration on the 4 by 4 and 8 by 8 maps and compare the policies and the number of sweeps.",
   train='''results = {}
for size, name in [("4x4", "4x4"), ("8x8", "8x8")]:
    P_, nS_, nA_ = tabular.transition_model("FrozenLake-v1", map_name=size, is_slippery=True)
    V_vi, pol_vi, sweeps = tabular.value_iteration(P_, nS_, nA_, GAMMA)
    V_pi, pol_pi, rounds = tabular.policy_iteration(P_, nS_, nA_, GAMMA)
    results[size] = dict(P=P_, nS=nS_, V=V_vi, policy=pol_vi, sweeps=sweeps, rounds=rounds, same=bool(np.allclose(V_vi, V_pi, atol=1e-6)))
    print(size, "| VI sweeps:", sweeps, "| PI rounds:", rounds, "| same values:", results[size]["same"], "| same policy:", bool(np.array_equal(pol_vi, pol_pi)))''',
   eval_md="Exact success probability: evaluate the optimal policy with discount 1. Check it by simulating 5000 episodes (the simulation has a binomial standard error of about 0.7 percentage points). The floor is the uniformly random policy, also evaluated exactly.",
   eval='''table = {}
for size, r in results.items():
    n = r["nS"]
    exact, _ = tabular.policy_evaluation(r["P"], r["policy"], n, gamma=1.0)
    uniform = np.zeros(n)
    for _ in range(3000):
        for s in range(n):
            uniform[s] = np.mean([sum(p * (rw + uniform[s2] * (not d)) for p, s2, rw, d in r["P"][s][a]) for a in range(4)])
    env = gym.make("FrozenLake-v1", map_name=size, is_slippery=True)
    wins = 0
    for i in range(5000):
        s, _ = env.reset(seed=1000 + i)
        done = False
        while not done:
            s, rew, term, trunc, _ = env.step(int(r["policy"][s]))
            done = term or trunc
        wins += rew > 0
    table[size] = {"exact_success_optimal": float(exact[0]), "simulated_success_optimal": wins / 5000, "exact_success_random": float(uniform[0]), "VI_sweeps": r["sweeps"], "PI_rounds": r["rounds"]}
display(pd.DataFrame(table).T.round(4))
METRICS = {**{f"{k}_{size}": v for size, d in table.items() for k, v in d.items()}}
CONFIG = {"gamma": GAMMA, "theta": 1e-10, "maps": ["4x4", "8x8"]}
EXTRA = {"optimal_policy_4x4": [int(a) for a in results["4x4"]["policy"]]}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
for ax, size, n in [(axes[0], "4x4", 4), (axes[1], "8x8", 8)]:
    V = results[size]["V"].reshape(n, n)
    ax.imshow(V, cmap="viridis")
    desc = gym.make("FrozenLake-v1", map_name=size).unwrapped.desc.astype(str)
    for i in range(n):
        for j in range(n):
            tile = desc[i, j]
            ax.text(j, i, tile if tile in "HG" else arrows[int(results[size]["policy"][i * n + j])], ha="center", va="center", color="white", fontsize=12)
    ax.set_title(f"Optimal values and policy ({size})"); ax.axis("off")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "policy"); plt.show()''',
   interpret_md="The policy often moves away from holes even when that is not the direct route, because a slippery move may go sideways.",
   interpret='''print("optimal action in each 4x4 state (0 left, 1 down, 2 right, 3 up):")
print(results["4x4"]["policy"].reshape(4, 4))''',
   experiments=[
       dict(title="Slippery versus deterministic ice", md="Plan on the deterministic lake and compare the success probability.",
            code='''P_d, nS_d, nA_d = tabular.transition_model("FrozenLake-v1", is_slippery=False)
_, pol_d, _ = tabular.value_iteration(P_d, nS_d, nA_d, GAMMA)
exact_d, _ = tabular.policy_evaluation(P_d, pol_d, nS_d, gamma=1.0)
print("deterministic lake: success probability of the optimal policy", float(exact_d[0]))'''),
       dict(title="Discount factor", md="A smaller gamma makes the agent care less about distant reward.",
            code='''for g in [0.5, 0.9, 0.99, 1.0]:
    V, pol, sw = tabular.value_iteration(results["4x4"]["P"], 16, 4, g)
    exact, _ = tabular.policy_evaluation(results["4x4"]["P"], pol, 16, gamma=1.0)
    print(f"gamma={g}: sweeps {sw}, success probability of the resulting policy {float(exact[0]):.4f}")'''),
       dict(title="Cost of a sweep", md="Count how the work grows with the number of states.",
            code='''for size in ["4x4", "8x8"]:
    r = results[size]
    print(size, "states", r["nS"], "| backups until convergence", r["nS"] * r["sweeps"])''')],
   discussion="The textbook describes DP as exact but limited by model knowledge and state count. Our two map sizes show the second limit directly in the number of backups.",
   improvements="Use asynchronous or prioritised sweeps and compare with sampling based methods.",
   takeaways=["DP needs the model.", "Value and policy iteration reach the same optimum.", "Slippery ice lowers the best possible success rate."],
   use_cases="Planning in small known MDPs, reference solutions.", limitations="Needs a model, small state spaces.", next="Monte Carlo methods.")

rl("02", TAB, id="monte_carlo", title="Monte Carlo Control: Learning Blackjack from Episodes", level="Intermediate", minutes=2,
   tagline="Learn values by averaging complete episodes without a model.", name="MonteCarloControl", dataset="Blackjack-v1",
   objectives=["First visit Monte Carlo estimation", "Exploration with epsilon soft policies", "Why returns have high variance", "How to compare a policy with a baseline using a confidence interval"],
   problem=dict(context="Blackjack has a known rule set, but the transition probabilities depend on the shuffled deck.",
                problem="Learn when to hit or stick from sampled games.",
                matters="Model free learning works when the model is too complex to write down.",
                why_ml="The best action depends on player sum, dealer card and usable ace.",
                why_algo="Episodes are short and end with a clear reward, which suits Monte Carlo."),
   papers=["sutton2018"],
   paper_relation="Sutton and Barto (not verified in the build environment, check before citing) use blackjack as the standard Monte Carlo example.",
   math=r"""**Return.** $G_t = \sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}$.

**Value estimate.** $V(s) = \text{mean}\{G_t : S_t = s\}$ over first visits. Incremental form: $Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\big(G - Q(s,a)\big)$.

**Control.** Act $\varepsilon$-greedily with respect to $Q$ so that every action keeps being tried. Policy improvement is the greedy step on the estimated $Q$.

**Variance.** $G$ sums many random rewards, so the estimate converges slowly. Standard error falls as $1/\sqrt{N}$.

**Limits.** Needs episodes that end, high variance, slow in large state spaces.""",
   dataset_md="**Blackjack-v1** (Gymnasium, `toy_text`). State is (player sum 4 to 21, dealer showing card 1 to 10, usable ace 0 or 1), 2 actions (stick, hit). Reward +1 win, 0 draw, -1 loss, with no natural bonus. The environment samples from an infinite deck. There is no dataset.",
   eda="""env = gym.make("Blackjack-v1", sab=False)
print("observation space:", env.observation_space, "| actions:", env.action_space)
state, _ = env.reset(seed=0)
print("example start state (player sum, dealer card, usable ace):", state)""",
   prep_md="No preprocessing. States are tuples used as dictionary keys. Training uses seeds from 0, evaluation uses separate seeds from 10 000 000.",
   prep='''def new_env():
    return gym.make("Blackjack-v1", sab=False)
print("ready")''',
   concept="Play a full game, then credit every visited state action pair with the final reward. After many games the averages approach the true values.",
   scratch_md="First visit Monte Carlo control with an epsilon soft policy. In Blackjack each state action pair appears at most once per game, so first visit and every visit are the same.",
   scratch='''def mc_control(episodes, epsilon=0.1, seed=0):
    rng = np.random.default_rng(seed)
    env = new_env()
    Q, N = {}, {}
    env.reset(seed=seed)
    curve = []
    for ep in range(episodes):
        state, _ = env.reset()
        trajectory, done = [], False
        while not done:
            q = Q.setdefault(state, np.zeros(2))
            action = int(rng.integers(2)) if rng.random() < epsilon else int(q.argmax())
            nxt, reward, term, trunc, _ = env.step(action)
            trajectory.append((state, action))
            state, done = nxt, term or trunc
        for s, a in trajectory:
            N[(s, a)] = N.get((s, a), 0) + 1
            Q[s][a] += (reward - Q[s][a]) / N[(s, a)]
        curve.append(reward)
    env.close()
    return Q, np.array(curve)
print(inspect.getsource(mc_control)[:300], "...")''',
   params="| Parameter | Meaning |\n|---|---|\n| `episodes` | Number of games. |\n| `epsilon` | Exploration probability. |\n| `gamma` | 1 here, episodes are short. |",
   train_md="Train with 5 seeds and 200 000 games each. Epsilon is 0.1.",
   train='''import time
t0 = time.time()
runs = []
for seed in range(5):
    Q, rewards = mc_control(200000, seed=seed)
    runs.append((Q, rewards))
    print("seed", seed, "states seen", len(Q), "| mean reward of last 20000 training games (with exploration)", round(float(rewards[-20000:].mean()), 4))
print("seconds:", round(time.time() - t0, 1))''',
   eval='''def run_policy(policy, games=50000, seed0=10_000_000):
    env = new_env()
    rewards = []
    for i in range(games):
        s, _ = env.reset(seed=seed0 + i)
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(policy(s))
            done = term or trunc
        rewards.append(r)
    return np.array(rewards)
stick20 = run_policy(lambda s: 0 if s[0] >= 20 else 1)
rng_eval = np.random.default_rng(0)
random_pol = run_policy(lambda s: int(rng_eval.integers(2)))
learned = []
for Q, _ in runs:
    learned.append(run_policy(lambda s, Q=Q: int(Q[s].argmax()) if s in Q else 0))
means = np.array([r.mean() for r in learned])
se = np.mean([r.std(ddof=1) / np.sqrt(len(r)) for r in learned])
table = pd.DataFrame({"mean_return": [means.mean(), stick20.mean(), random_pol.mean()],
                      "std_error": [se, stick20.std(ddof=1) / np.sqrt(len(stick20)), random_pol.std(ddof=1) / np.sqrt(len(random_pol))]},
                     index=["learned (mean of 5 seeds)", "stick on 20 or more", "random"])
display(table.round(4))
print("learned policy mean return per seed:", np.round(means, 4), "| spread across seeds:", round(float(means.std(ddof=1)), 4))
METRICS = {"eval_mean_return": float(means.mean()), "eval_std_across_seeds": float(means.std(ddof=1)), "eval_std_error": float(se),
           "stick20_mean_return": float(stick20.mean()), "random_mean_return": float(random_pol.mean())}
CONFIG = {"episodes": 200000, "epsilon": 0.1, "seeds": 5, "eval_games": 50000}
EXTRA = {"per_seed_eval": [float(m) for m in means]}''',
   viz='''Q = runs[0][0]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, ace in zip(axes, [0, 1]):
    grid = np.full((18, 10), np.nan)
    for (p, d, a), q in Q.items():
        if a == ace and 4 <= p <= 21:
            grid[p - 4, d - 1] = int(q.argmax())
    im = ax.imshow(grid, origin="lower", cmap="coolwarm", extent=[0.5, 10.5, 3.5, 21.5], aspect="auto")
    ax.set_xlabel("Dealer showing card"); ax.set_ylabel("Player sum"); ax.set_title(f"Greedy action, usable ace = {ace} (0 stick, 1 hit)")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "policy"); plt.show()''',
   interpret_md="The learned rule should stick on high sums and hit on low sums, with the threshold depending on the dealer card.",
   interpret='''no_ace = {s[0]: int(q.argmax()) for s, q in Q.items() if s[1] == 10 and s[2] == 0}
print("action by player sum when the dealer shows 10, no usable ace (0 stick, 1 hit):", dict(sorted(no_ace.items())))''',
   experiments=[
       dict(title="Number of training games", md="Learned policy return against games, one seed, 20000 evaluation games.",
            code='''for n in [1000, 10000, 100000]:
    Qn, _ = mc_control(n, seed=0)
    r = run_policy(lambda s, Qn=Qn: int(Qn[s].argmax()) if s in Qn else 0, games=20000)
    print(f"{n} games: mean return {r.mean():.4f} +- {r.std(ddof=1) / np.sqrt(len(r)):.4f}")'''),
       dict(title="Exploration rate", md="Epsilon 0.01, 0.1 and 0.3 with 50 000 games.",
            code='''for eps in [0.01, 0.1, 0.3]:
    Qe, _ = mc_control(50000, epsilon=eps, seed=0)
    r = run_policy(lambda s, Qe=Qe: int(Qe[s].argmax()) if s in Qe else 0, games=20000)
    print(f"epsilon={eps}: mean return {r.mean():.4f}")'''),
       dict(title="Variance of the estimate", md="Standard deviation of a single game reward and the resulting number of games for a given precision.",
            code='''sd = learned[0].std(ddof=1)
print("std of one game reward:", round(float(sd), 3))
for target in [0.01, 0.005]:
    print(f"games needed for a standard error of {target}: about {int((sd / target) ** 2)}")''')],
   discussion="The textbook treatment predicts slow convergence from high variance. The games needed for a precise comparison, computed above, make that concrete: small differences between policies need many games to see.",
   improvements="Use off policy evaluation with importance sampling, or TD methods to cut variance.",
   takeaways=["MC averages complete episodes.", "Epsilon soft policies keep exploring.", "Report standard errors for policy comparisons."],
   use_cases="Games, simulators with no model.", limitations="High variance, episodic tasks only.", next="Temporal difference learning.")

rl("03", TAB, id="temporal_difference", title="Temporal Difference Learning: TD(0) Value Estimation", level="Intermediate", minutes=2,
   tagline="Update value estimates after every step by bootstrapping.", name="TD0", dataset="FrozenLake-v1",
   objectives=["The TD error", "Bootstrapping versus Monte Carlo", "The effect of the step size", "Bias and variance of the two estimators"],
   problem=dict(context="Waiting for the end of an episode is slow and noisy.",
                problem="Estimate the value of a fixed policy on FrozenLake from experience, without a model.",
                matters="TD learning is the base of Q-learning, SARSA and deep value methods.",
                why_ml="Values must be learned from sampled transitions.",
                why_algo="Exact values from DP give a ground truth to measure the error of TD."),
   papers=["sutton1988"],
   paper_relation="Sutton (1988) introduces TD methods and compares them with supervised prediction. We use a different task, the lake.",
   math=r"""**TD(0) update.** $V(S_t) \leftarrow V(S_t) + \alpha\,\delta_t$ with $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$.

**Monte Carlo update.** $V(S_t) \leftarrow V(S_t) + \alpha\,(G_t - V(S_t))$.

**Bias and variance.** TD uses the current estimate as target, so it is biased early but has low variance. MC is unbiased but has high variance.

**Convergence.** For a fixed policy and decreasing step sizes TD(0) converges to $V^\pi$ (tabular).""",
   dataset_md="**FrozenLake-v1** 4 by 4 slippery. The policy is fixed to the optimal policy from dynamic programming. The true values come from exact policy evaluation with discount 1, so the error of the estimates can be measured exactly.",
   eda="""P, nS, nA = tabular.transition_model("FrozenLake-v1", is_slippery=True)
_, optimal_policy, _ = tabular.value_iteration(P, nS, nA, 0.99)
true_values, _ = tabular.policy_evaluation(P, optimal_policy, nS, gamma=1.0)
print(np.round(true_values.reshape(4, 4), 3))""",
   prep_md="The policy is fixed and the discount is 1 (the task ends). Error is root mean squared error over all non terminal states.",
   prep='''nonterminal = np.array([s for s in range(nS) if not all(done for a in range(nA) for _, _, _, done in P[s][a])])
def rmse(values):
    return float(np.sqrt(np.mean((values[nonterminal] - true_values[nonterminal]) ** 2)))
print("non terminal states:", len(nonterminal))''',
   concept="After each step, move the value of the state you left a little towards reward plus the value of the state you reached.",
   scratch_md="The TD(0) function is in `ml_lab.rl.tabular`. We add an every visit Monte Carlo version here to compare.",
   scratch='''print(inspect.getsource(tabular.td0_prediction))
def mc_prediction(policy, episodes, alpha=0.05, seed=0):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    values = np.zeros(16); history = []
    env.reset(seed=seed)
    for _ in range(episodes):
        s, _ = env.reset(); states, done = [], False
        while not done:
            states.append(s)
            s, r, term, trunc, _ = env.step(int(policy[s])); done = term or trunc
        for st in states:
            values[st] += alpha * (r - values[st])      # return = final reward
        history.append(values.copy())
    return np.array(history)''',
   params="| Parameter | Meaning |\n|---|---|\n| `alpha` | Step size. |\n| `gamma` | Discount, 1 here. |\n| `episodes` | Training episodes. |",
   train_md="Run TD(0) and Monte Carlo prediction with step sizes 0.01, 0.05 and 0.2 for 3000 episodes and 5 seeds. Track the error after every 100 episodes.",
   train='''checkpoints = list(range(99, 3000, 100))
curves = {}
for method in ["TD(0)", "MC"]:
    for alpha in [0.01, 0.05, 0.2]:
        errs = []
        for seed in range(5):
            if method == "TD(0)":
                _, hist = tabular.td0_prediction("FrozenLake-v1", optimal_policy, 3000, alpha=alpha, gamma=1.0, seed=seed, is_slippery=True)
            else:
                hist = mc_prediction(optimal_policy, 3000, alpha=alpha, seed=seed)
            errs.append([rmse(hist[c]) for c in checkpoints])
        curves[(method, alpha)] = np.array(errs)
final = pd.DataFrame({f"{m} alpha={a}": [c[:, -1].mean(), c[:, -1].std(ddof=1)] for (m, a), c in curves.items()}, index=["rmse_mean", "rmse_std"]).T
display(final.round(4))''',
   eval_md="Root mean squared error against the exact values after 3000 episodes, mean and spread over 5 seeds. The floor is the error of an all zero estimate.",
   eval='''floor = rmse(np.zeros(16))
best_key = min(curves, key=lambda k: curves[k][:, -1].mean())
METRICS = {"all_zero_rmse": floor, "best_final_rmse": float(curves[best_key][:, -1].mean()), "best_final_rmse_std": float(curves[best_key][:, -1].std(ddof=1)),
           "td_alpha_0.05_rmse": float(curves[("TD(0)", 0.05)][:, -1].mean()), "mc_alpha_0.05_rmse": float(curves[("MC", 0.05)][:, -1].mean())}
CONFIG = {"episodes": 3000, "alphas": [0.01, 0.05, 0.2], "best": f"{best_key[0]} alpha={best_key[1]}", "seeds": 5}
EXTRA = {"final_table": final.round(5).reset_index().to_dict("records")}
print("best setting:", best_key)''',
   viz='''fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, alpha in zip(axes, [0.01, 0.05, 0.2]):
    for method in ["TD(0)", "MC"]:
        c = curves[(method, alpha)]
        ax.plot(np.array(checkpoints) + 1, c.mean(axis=0), label=method)
        ax.fill_between(np.array(checkpoints) + 1, c.mean(axis=0) - c.std(axis=0), c.mean(axis=0) + c.std(axis=0), alpha=0.2)
    ax.set_title(f"alpha = {alpha}"); ax.set_xlabel("Episodes"); ax.legend()
axes[0].set_ylabel("RMSE against exact values")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "curves"); plt.show()''',
   interpret_md="Compare the final values with the exact ones. Errors are larger in states far from the goal.",
   interpret='''_, hist = tabular.td0_prediction("FrozenLake-v1", optimal_policy, 3000, alpha=0.05, gamma=1.0, seed=0, is_slippery=True)
display(pd.DataFrame({"exact": true_values, "TD(0) estimate": hist[-1]}).round(3).T)''',
   experiments=[
       dict(title="Step size and noise", md="A larger step size learns faster but keeps more noise. Compare the spread of the last 10 checkpoints.",
            code='''for (m, a), c in curves.items():
    print(f"{m} alpha={a}: mean of last 10 checkpoints {c[:, -10:].mean():.4f}, std within run {c[:, -10:].std(axis=1).mean():.4f}")'''),
       dict(title="Speed in the first 200 episodes", md="Error after 100 and 300 episodes.",
            code='''for (m, a), c in curves.items():
    print(f"{m} alpha={a}: rmse after 100 episodes {c[:, 0].mean():.3f}, after 300 episodes {c[:, 2].mean():.3f}")'''),
       dict(title="Wrong policy", md="Evaluate a random policy with the same code. The values differ from the optimal policy values.",
            code='''random_policy = np.random.default_rng(0).integers(0, 4, nS)
v_true_random, _ = tabular.policy_evaluation(P, random_policy, nS, gamma=1.0)
print("start state value, optimal policy:", round(float(true_values[0]), 3), "| random fixed policy:", round(float(v_true_random[0]), 4))''')],
   discussion="Sutton (1988) argues that TD methods use experience more efficiently than methods that wait for the outcome. Our curves test this on one small task where the reward is sparse, so the comparison is limited to that case.",
   improvements="Try TD(lambda) and eligibility traces, and decaying step sizes.",
   takeaways=["TD bootstraps and updates every step.", "Step size trades speed for noise.", "Exact values allow exact error measurement."],
   use_cases="Prediction, base of control methods.", limitations="Fixed policy here, tabular only.", next="SARSA.")

rl("04", TAB, id="sarsa", title="SARSA: Safe Paths on the Cliff", level="Intermediate", minutes=1,
   tagline="Learn the value of the policy you actually follow, including its exploration.", name="SARSA", dataset="CliffWalking-v1",
   objectives=["On policy TD control", "How exploration changes the learned path", "Comparing the online return with the greedy return", "Why SARSA picks the safer route"],
   problem=dict(context="A robot walks along a cliff edge. A fall costs a lot and sends it back to the start.",
                problem="Learn a route from start to goal when exploratory moves may fall off the cliff.",
                matters="When mistakes are costly during learning, the learning rule decides how careful the agent is.",
                why_ml="The agent has no model of the grid.",
                why_algo="SARSA accounts for exploration in its updates."),
   papers=["sutton2018", "sutton1988"],
   paper_relation="Sutton and Barto (not verified in the build environment) introduce the cliff walking comparison of SARSA and Q-learning. We run our own version.",
   math=r"""**Update.** After the quintuple $(S, A, R, S', A')$:

$$Q(S,A) \leftarrow Q(S,A) + \alpha\,[R + \gamma Q(S',A') - Q(S,A)]$$

**On policy.** $A'$ is the action the epsilon greedy policy will take, so the learned values include the cost of exploring.

**Convergence.** Tabular SARSA converges to the optimal policy if the policy becomes greedy in the limit and every pair is visited infinitely often.""",
   dataset_md="**CliffWalking-v1** (Gymnasium, `toy_text`): 4 by 12 grid, 48 states, 4 actions. Every step costs -1, stepping into the cliff costs -100 and returns to the start. The best path along the cliff edge has return -13. There is no dataset.",
   eda="""env = gym.make("CliffWalking-v1")
print("observations:", env.observation_space, "| actions:", env.action_space)
print("start state:", env.reset(seed=0)[0], "| goal state: 47 | cliff states: 37 to 46")""",
   prep_md="Same settings for SARSA and Q-learning: step size 0.5, discount 1, epsilon 0.1, 500 episodes, 10 seeds. Greedy evaluation uses the Q table without exploration.",
   prep='''EPISODES, SEEDS = 500, range(10)
def greedy_return(q, max_steps=200):
    env = gym.make("CliffWalking-v1")
    s, _ = env.reset(seed=0)
    total, path = 0, [s]
    for _ in range(max_steps):
        s, r, term, trunc, _ = env.step(int(q[s].argmax()))
        total += r; path.append(s)
        if term or trunc:
            break
    return total, path''',
   concept="The agent learns from what it really does. If exploring near the cliff sometimes ends in a fall, the values of cliff edge cells go down.",
   scratch_md="Source of the SARSA update in `ml_lab.rl.tabular`:",
   scratch='''print(inspect.getsource(tabular.sarsa))''',
   params="| Parameter | Meaning |\n|---|---|\n| `alpha` | Step size. |\n| `gamma` | Discount. |\n| `epsilon` | Exploration rate. |",
   train_md="Train SARSA and Q-learning with 10 seeds each.",
   train='''runs = {}
for name, fn in [("SARSA", tabular.sarsa), ("Q-learning", tabular.q_learning)]:
    runs[name] = [fn("CliffWalking-v1", EPISODES, alpha=0.5, gamma=1.0, epsilon=0.1, seed=s) for s in SEEDS]
for name, rs in runs.items():
    print(name, "| online return over last 100 episodes (mean over seeds):", round(float(np.mean([r[1][-100:].mean() for r in rs])), 1))''',
   eval_md="Two numbers per method. Online return: the average reward while learning and exploring (last 100 episodes). Greedy return: reward of the final greedy path. SARSA is expected to have the higher online return and Q-learning the shorter greedy path. We check this, we do not assume it.",
   eval='''summary = {}
for name, rs in runs.items():
    online = np.array([r[1][-100:].mean() for r in rs])
    greedy = np.array([greedy_return(r[0])[0] for r in rs])
    summary[name] = dict(online_mean=online.mean(), online_std=online.std(ddof=1), greedy_mean=greedy.mean(), greedy_std=greedy.std(ddof=1))
display(pd.DataFrame(summary).T.round(2))
METRICS = {f"{k}_{n.lower().replace('-', '_')}": float(v) for n, d in summary.items() for k, v in d.items()}
CONFIG = {"alpha": 0.5, "gamma": 1.0, "epsilon": 0.1, "episodes": EPISODES, "seeds": 10}
EXTRA = {"optimal_return": -13}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, rs in runs.items():
    curves = np.array([np.convolve(r[1], np.ones(10) / 10, mode="valid") for r in rs])
    axes[0].plot(curves.mean(axis=0), label=name)
    axes[0].fill_between(range(curves.shape[1]), curves.mean(axis=0) - curves.std(axis=0), curves.mean(axis=0) + curves.std(axis=0), alpha=0.2)
axes[0].set_ylim(-150, 0); axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Online return (10 episode mean)"); axes[0].legend(); axes[0].set_title("Learning curves, 10 seeds")
grid = np.zeros((4, 12))
for name, color in [("SARSA", 1), ("Q-learning", 2)]:
    _, path = greedy_return(runs[name][0][0])
    for s in path:
        grid[s // 12, s % 12] += color
grid[3, 1:11] = -1
axes[1].imshow(grid, cmap="coolwarm"); axes[1].set_title("Greedy paths (seed 0): 1 SARSA, 2 Q-learning, 3 both"); axes[1].axis("off")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "paths"); plt.show()''',
   interpret_md="The SARSA path keeps a distance from the cliff because its values contain the cost of exploratory falls.",
   interpret='''for name in runs:
    ret, path = greedy_return(runs[name][0][0])
    rows_used = sorted({s // 12 for s in path})
    print(f"{name}: greedy return {ret}, path length {len(path) - 1}, rows used {rows_used} (row 3 is the cliff edge row)")''',
   experiments=[
       dict(title="Exploration rate", md="Epsilon 0.01, 0.1 and 0.3 for SARSA, online and greedy return, 5 seeds.",
            code='''for eps in [0.01, 0.1, 0.3]:
    rs = [tabular.sarsa("CliffWalking-v1", EPISODES, alpha=0.5, gamma=1.0, epsilon=eps, seed=s) for s in range(5)]
    print(f"epsilon={eps}: online {np.mean([r[1][-100:].mean() for r in rs]):.1f}, greedy {np.mean([greedy_return(r[0])[0] for r in rs]):.1f}")'''),
       dict(title="Decaying exploration", md="With epsilon decaying to 0.01 SARSA should approach the optimal edge path.",
            code='''rs = [tabular.sarsa("CliffWalking-v1", 2000, alpha=0.5, gamma=1.0, epsilon=0.5, decay=0.995, seed=s) for s in range(5)]
print("greedy returns with decaying epsilon:", [greedy_return(r[0])[0] for r in rs], "(optimal -13)")'''),
       dict(title="Step size", md="Alpha 0.1, 0.5, 0.9.",
            code='''for a in [0.1, 0.5, 0.9]:
    rs = [tabular.sarsa("CliffWalking-v1", EPISODES, alpha=a, gamma=1.0, epsilon=0.1, seed=s) for s in range(5)]
    print(f"alpha={a}: online {np.mean([r[1][-100:].mean() for r in rs]):.1f}")''')],
   discussion="The textbook story is that SARSA learns the safer path and Q-learning the shorter one. The online and greedy numbers above test both halves of that statement on our runs.",
   improvements="Try expected SARSA and decaying epsilon, and compare with more seeds.",
   takeaways=["SARSA is on policy.", "Exploration enters the learned values.", "Report online and greedy returns separately."],
   use_cases="Tasks where exploration is costly.", limitations="Tabular only.", next="Q-learning.")

rl("05", TAB, id="q_learning", title="Q-Learning: Off-Policy Control on a Slippery Lake", level="Intermediate", minutes=2,
   tagline="Learn the optimal action values while behaving exploratively.", name="QLearning", dataset="FrozenLake-v1",
   objectives=["The Q-learning update and the max operator", "Why Q-learning is off policy", "Exploration schedules", "How to compare a learned policy with the DP optimum"],
   problem=dict(context="The agent does not know the ice model, but can try crossing many times.",
                problem="Learn to cross the slippery 4 by 4 lake from interaction only.",
                matters="The exact optimum from DP is known, so we can measure how close learning gets.",
                why_ml="The model is unknown to the learner.",
                why_algo="Q-learning is the basic model free control method."),
   papers=["watkins1992", "towers2024"],
   paper_relation="Watkins and Dayan prove convergence of Q-learning for tabular problems. We check how close a finite run gets.",
   math=r"""**Update.** $Q(S,A) \leftarrow Q(S,A) + \alpha\,[R + \gamma \max_{a'} Q(S',a') - Q(S,A)]$.

**Off policy.** The target uses the greedy action in $S'$, whatever action the behaviour policy takes next. Q converges to $Q^*$ if every pair is visited infinitely often and step sizes decrease suitably.

**Exploration.** Epsilon greedy with decaying epsilon.

**Limits.** Maximisation bias, slow with sparse rewards.""",
   dataset_md="**FrozenLake-v1** 4 by 4 slippery, as in the DP notebook. The reward is sparse (1 only at the goal). Dynamic programming gave the optimal success probability, which is the reference.",
   eda="""P, nS, nA = tabular.transition_model("FrozenLake-v1", is_slippery=True)
V_opt, pol_opt, _ = tabular.value_iteration(P, nS, nA, 0.99)
exact_opt, _ = tabular.policy_evaluation(P, pol_opt, nS, gamma=1.0)
print("optimal success probability from dynamic programming:", round(float(exact_opt[0]), 4))""",
   prep_md="Step size 0.1, discount 0.99, epsilon decays from 1.0 with factor 0.9995 per episode (floor 0.01). Training 20 000 episodes, 5 seeds. Evaluation: greedy policy for 2000 episodes with seeds not used in training.",
   prep='''EPISODES = 20000
KW = dict(alpha=0.1, gamma=0.99, epsilon=1.0, decay=0.9995, is_slippery=True)''',
   concept="Each step moves the value of the chosen action toward the reward plus the best value available next. Over many tries the best actions stand out.",
   scratch_md="Source of the update:",
   scratch='''print(inspect.getsource(tabular.q_learning))''',
   params="| Parameter | Meaning |\n|---|---|\n| `alpha` | Step size. |\n| `gamma` | Discount. |\n| `epsilon`, `decay` | Exploration schedule. |",
   train_md="Train 5 seeds. After training we evaluate each greedy policy.",
   train='''runs = [tabular.q_learning("FrozenLake-v1", EPISODES, seed=s, **KW) for s in range(5)]
for s, (q, ret) in enumerate(runs):
    print("seed", s, "| training success rate over last 2000 episodes (with exploration):", round(float(ret[-2000:].mean()), 3))''',
   eval_md="Greedy success rate over 2000 evaluation episodes for each seed, compared with the simulated success rate of the DP optimal policy (same episodes) and a random policy.",
   eval='''def success_of(policy_fn, episodes=2000, seed0=50000):
    env = gym.make("FrozenLake-v1", is_slippery=True); wins = 0
    for i in range(episodes):
        s, _ = env.reset(seed=seed0 + i); done = False
        while not done:
            s, r, term, trunc, _ = env.step(policy_fn(s)); done = term or trunc
        wins += r > 0
    return wins / episodes
learned = np.array([success_of(lambda s, q=q: int(q[s].argmax())) for q, _ in runs])
dp_success = success_of(lambda s: int(pol_opt[s]))
rng_eval = np.random.default_rng(0)
random_success = success_of(lambda s: int(rng_eval.integers(4)))
display(pd.DataFrame({"success_rate": list(learned) + [dp_success, random_success]}, index=[f"Q-learning seed {i}" for i in range(5)] + ["DP optimal policy", "random policy"]).round(4))
METRICS = {"success_rate_mean": float(learned.mean()), "success_rate_std": float(learned.std(ddof=1)), "dp_optimal_success_simulated": float(dp_success),
           "dp_optimal_success_exact": float(exact_opt[0]), "random_success": float(random_success)}
CONFIG = {**{k: v for k, v in KW.items() if k != "is_slippery"}, "episodes": EPISODES, "seeds": 5}
EXTRA = {"per_seed_success": [float(x) for x in learned]}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for s, (q, ret) in enumerate(runs):
    axes[0].plot(np.convolve(ret, np.ones(500) / 500, mode="valid"), alpha=0.7, label=f"seed {s}")
axes[0].axhline(float(exact_opt[0]), color="black", linestyle="--", label="DP optimum"); axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Success rate (500 episode mean, with exploration)"); axes[0].legend(fontsize=7)
axes[1].imshow(runs[0][0].max(axis=1).reshape(4, 4), cmap="viridis"); axes[1].set_title("Learned state values (seed 0)"); axes[1].axis("off")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "curves"); plt.show()''',
   interpret_md="Compare the learned policy with the DP policy state by state.",
   interpret='''agree = [float(np.mean(q.argmax(axis=1) == pol_opt)) for q, _ in runs]
print("share of states where the learned action equals the DP action, by seed:", np.round(agree, 2))
print("states never updated (all zero row):", [int((q.max(axis=1) == 0).sum()) for q, _ in runs])''',
   experiments=[
       dict(title="Step size", md="Alpha 0.02, 0.1, 0.5, 3 seeds, greedy success rate.",
            code='''for a in [0.02, 0.1, 0.5]:
    res = [success_of(lambda s, q=q: int(q[s].argmax()), 1000) for q, _ in [tabular.q_learning("FrozenLake-v1", EPISODES, seed=s, **{**KW, "alpha": a}) for s in range(3)]]
    print(f"alpha={a}: success {np.mean(res):.3f}")'''),
       dict(title="Exploration schedule", md="Constant epsilon 0.1 versus decaying epsilon.",
            code='''for name, kw in [("constant 0.1", dict(epsilon=0.1, decay=None)), ("decaying", dict(epsilon=1.0, decay=0.9995))]:
    res = [success_of(lambda s, q=q: int(q[s].argmax()), 1000) for q, _ in [tabular.q_learning("FrozenLake-v1", EPISODES, seed=s, **{**KW, **kw}) for s in range(3)]]
    print(f"{name}: success {np.mean(res):.3f}")'''),
       dict(title="Amount of training", md="Success against the number of episodes.",
            code='''for n in [1000, 5000, 20000]:
    res = [success_of(lambda s, q=q: int(q[s].argmax()), 1000) for q, _ in [tabular.q_learning("FrozenLake-v1", n, seed=s, **{**KW, "decay": 0.9995 if n >= 20000 else 0.999}) for s in range(3)]]
    print(f"{n} episodes: success {np.mean(res):.3f}")''')],
   discussion="Watkins and Dayan prove convergence in the limit. Our runs are finite, so the gap to the DP optimum and the spread across seeds show how far a practical run gets.",
   improvements="Double Q-learning, optimistic initial values, and more seeds with confidence intervals.",
   takeaways=["Q-learning learns Q* off policy.", "Compare with DP when the model is small.", "Seeds matter."],
   use_cases="Small discrete control tasks.", limitations="Tabular, maximisation bias.", next="Deep Q-Networks.")
