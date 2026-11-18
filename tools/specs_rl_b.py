"""RL specs, part B (deep)."""

from specs_rl_a import rl

DV, PG = "02_deep_value_methods", "03_policy_gradient_methods"

CP_DATASET = ("**CartPole-v1** (Gymnasium, `classic_control`). A pole is hinged to a cart. State: cart position, cart velocity, pole angle, pole angular velocity "
              "(4 real numbers). Actions: push left or right. Reward +1 per step. An episode ends when the pole falls past about 12 degrees, the cart leaves the track, or after 500 steps. "
              "The maximum return is 500, and a mean of 475 is the usual solved threshold. There is no dataset, the environment is simulated.")
CP_EDA = """env = gym.make("CartPole-v1")
print("observation space:", env.observation_space)
print("action space:", env.action_space, "| max episode steps:", env.spec.max_episode_steps)
rng0 = np.random.default_rng(0)
random_returns = evaluate_policy(lambda: gym.make("CartPole-v1"), lambda s: int(rng0.integers(2)), episodes=50, seed=0)
print("random policy over 50 episodes:", {k: round(v, 1) for k, v in random_returns.items()})"""
CP_PREP_MD = ("No preprocessing: the 4 state values are fed directly to a network with two hidden layers of 64 units. "
              "Training uses seeds 0, 1 and 2. Evaluation uses 20 fresh episodes per seed with environment seeds from 1000, never seen in training. "
              "Only a CPU is used, so the budgets are small and results will be below long GPU runs.")
CP_PREP = '''SEEDS = [0, 1, 2]
make_env = lambda: gym.make("CartPole-v1")
GRID = np.linspace(1000, TOTAL_STEPS, 40)'''

DEEP_RUN = '''runs = {}
for seed in SEEDS:
    runs[seed] = __CALL__
    last = runs[seed]["returns"][-10:]
    print("seed", seed, "| episodes", len(runs[seed]["returns"]), "| mean training return of last 10 episodes:", round(float(np.mean(last)), 1))'''

DEEP_EVAL = '''rng_eval = np.random.default_rng(0)
random_eval = evaluate_policy(make_env, lambda s: int(rng_eval.integers(2)), episodes=20, seed=1000)
evals = {seed: evaluate_policy(make_env, runs[seed]["policy"], episodes=20, seed=1000, success_threshold=475) for seed in SEEDS}
table = pd.DataFrame(evals).T[["mean_return", "std_return", "success_rate"]]
display(table.round(1))
means = np.array([e["mean_return"] for e in evals.values()])
METRICS = {"eval_mean_return": float(means.mean()), "eval_std_across_seeds": float(means.std(ddof=1)), "eval_min_seed": float(means.min()), "eval_max_seed": float(means.max()),
           "random_policy_mean_return": float(random_eval["mean_return"]), "share_of_seeds_solved": float((means >= 475).mean())}
EXTRA = {"per_seed_eval_return": [float(m) for m in means], "training_steps_per_seed": [int(runs[s]["steps"][-1]) for s in SEEDS]}'''

DEEP_VIZ = '''fig, ax = plt.subplots(figsize=(7, 4))
curves = np.array([deep.curve_on_grid(runs[s]["steps"], runs[s]["returns"], GRID, window=10) for s in SEEDS])
for c in curves:
    ax.plot(GRID, c, alpha=0.35)
ax.plot(GRID, np.nanmean(curves, axis=0), color="black", linewidth=2, label="mean of seeds")
ax.axhline(random_eval["mean_return"], color="gray", linestyle=":", label="random policy"); ax.axhline(500, color="green", linestyle="--", label="maximum 500")
ax.set_xlabel("Environment steps"); ax.set_ylabel("Return (mean of last 10 episodes)"); ax.set_title("Learning curves, 3 seeds"); ax.legend()
save_figure(fig, NOTEBOOK_ID, "curves"); plt.show()'''

DEEP_INTERP = '''print("per seed evaluation return:", np.round(means, 1))
print("a seed counts as solved when its evaluation mean is at least 475; solved seeds:", int((means >= 475).sum()), "of", len(SEEDS))
final_train = np.array([np.mean(runs[s]["returns"][-10:]) for s in SEEDS])
print("training return of last 10 episodes versus evaluation return:", np.round(final_train, 1), "vs", np.round(means, 1))'''

DEEP_EXTRA_LIMITS = "Three seeds and a CPU sized step budget, so these runs show trends, not converged scores."


def curve_exp(title, md, variants, call_template, steps_note, seeds="[0, 1]"):
    code = f'''for name, kw in {variants}:
    rets = []
    for seed in {seeds}:
        out = {call_template}
        ev = evaluate_policy(make_env, out["policy"], episodes=10, seed=1000)
        rets.append(ev["mean_return"])
    print(f"{{name}}: evaluation return by seed {{np.round(rets, 1)}}, mean {{np.mean(rets):.1f}}")'''
    return dict(title=title, md=md + " " + steps_note, code=code)


rl("06", DV, id="dqn", title="Deep Q-Network: Balancing CartPole", level="Advanced", minutes=3,
   tagline="Approximate Q values with a neural network, experience replay and a target network.", name="DQN", dataset="CartPole-v1",
   objectives=["Why a table cannot hold continuous states", "How replay and target networks stabilise training", "Epsilon schedules", "How to read noisy RL learning curves"],
   problem=dict(context="Balancing an unstable system is a standard control benchmark.",
                problem="Learn to keep the pole up by pushing the cart, from state observations only.",
                matters="The same value approximation ideas drive agents for games and robots.",
                why_ml="The state is continuous, so a lookup table is impossible.",
                why_algo="DQN is the reference deep value method for discrete actions."),
   papers=["mnih2015", "henderson2018"],
   paper_relation="Mnih et al. train DQN on Atari images with a convolutional network and millions of frames. We train a small MLP on CartPole for 20 000 steps, so no result is comparable. Henderson et al. explain why seeds and hyperparameters matter.",
   math=r"""**Q-network.** $Q_\theta(s,a)$ approximates the optimal action value.

**Loss on a replay minibatch.** $L(\theta) = \mathbb{E}\big[\ell\big(y - Q_\theta(s,a)\big)\big]$ with target $y = r + \gamma(1-d)\max_{a'} Q_{\theta^-}(s',a')$, where $\theta^-$ are the parameters of a target network copied every $C$ steps and $\ell$ is the Huber loss.

**Replay.** Sampling past transitions at random breaks the correlation between consecutive samples.

**Exploration.** Epsilon decays linearly from 1.0 to 0.05.

**Limits.** Sample hungry, sensitive to hyperparameters, overestimates values.""",
   dataset_md=CP_DATASET, eda=CP_EDA, prep_md=CP_PREP_MD, prep="TOTAL_STEPS = 20000\n" + CP_PREP,
   concept="A network predicts the value of both actions. The agent mostly picks the larger one, and the network is trained to match one step lookahead targets.",
   scratch_md="The agent is in `ml_lab.rl.deep.train_dqn`. Source of the training function:",
   scratch='''print(inspect.getsource(deep.train_dqn))''',
   params="| Parameter | Meaning |\n|---|---|\n| `lr` | Adam learning rate. |\n| `buffer_size` | Replay capacity. |\n| `target_every` | Steps between target network copies. |\n| `eps_decay_steps` | Steps over which epsilon falls. |\n| `double` | Use the Double DQN target. |",
   train_md="Three seeds, 20 000 environment steps each, learning starts after 500 steps.",
   train=DEEP_RUN.replace("__CALL__", "deep.train_dqn(seed=seed, total_steps=TOTAL_STEPS)"),
   eval_md="Evaluation return of the greedy policy on 20 fresh episodes per seed. Floor: random policy.",
   eval=DEEP_EVAL + '''
CONFIG = {"total_steps": TOTAL_STEPS, "lr": 1e-3, "buffer_size": 20000, "batch": 64, "target_every": 250, "gamma": 0.99, "seeds": 3}''',
   viz=DEEP_VIZ, interpret_md="Training curves are noisy and a single seed can look great or poor.", interpret=DEEP_INTERP,
   experiments=[
       curve_exp("Target network update interval", "Update every 1, 250 and 1000 steps.",
                 '[("every 1 (no real target)", dict(target_every=1)), ("every 250", dict(target_every=250)), ("every 1000", dict(target_every=1000))]',
                 "deep.train_dqn(seed=seed, total_steps=12000, **kw)", "Two seeds and 12 000 steps."),
       curve_exp("Replay buffer size", "A tiny buffer keeps only recent, correlated samples.",
                 '[("buffer 500", dict(buffer_size=500)), ("buffer 20000", dict(buffer_size=20000))]',
                 "deep.train_dqn(seed=seed, total_steps=12000, **kw)", "Two seeds and 12 000 steps."),
       curve_exp("Learning rate", "A 100 fold change.",
                 '[("1e-4", dict(lr=1e-4)), ("1e-3", dict(lr=1e-3)), ("1e-2", dict(lr=1e-2))]',
                 "deep.train_dqn(seed=seed, total_steps=12000, **kw)", "Two seeds and 12 000 steps.")],
   discussion="Mnih et al. credit replay and a target network for stability. Our interval experiment tests the target network part on a different task. With two seeds per setting the comparison is only indicative, as Henderson et al. warn.",
   improvements="More seeds, longer training, prioritised replay, and a hyperparameter search with equal budgets.",
   takeaways=["DQN = Q-network + replay + target network.", "Curves are noisy, use several seeds.", "Small budgets give weak results."],
   use_cases="Discrete action control from states or images.", limitations="Sample hungry, unstable.", next="Double DQN.", limits_extra=DEEP_EXTRA_LIMITS)

rl("07", DV, id="double_dqn", title="Double DQN: Measuring Overestimation", level="Advanced", minutes=3,
   tagline="Separate action choice from action evaluation to reduce optimistic value estimates.", name="DoubleDQN", dataset="CartPole-v1",
   objectives=["Maximisation bias", "The Double DQN target", "How to measure overestimation against an upper bound", "Whether a better value estimate means a better policy"],
   problem=dict(context="Taking the maximum over noisy estimates makes them too high on average.",
                problem="Compare the value estimates and the returns of DQN and Double DQN on CartPole.",
                matters="Overoptimistic values can mislead the policy.",
                why_ml="Value estimates are learned and noisy.",
                why_algo="Double DQN is a one line change that targets this bias."),
   papers=["vanhasselt2016", "mnih2015"],
   paper_relation="van Hasselt et al. show reduced overestimation and better scores on Atari. We test the overestimation part directly because CartPole has a known upper bound on the return.",
   math=r"""**DQN target.** $y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')$. The same network picks and values the action, so noise in $Q$ is selected upwards.

**Double DQN target.** $y = r + \gamma\, Q_{\theta^-}\big(s', \arg\max_{a'} Q_\theta(s',a')\big)$. The online network chooses, the target network evaluates.

**Upper bound in CartPole.** Reward is 1 per step and episodes last at most 500 steps, so the discounted return is at most $\sum_{k=0}^{499}\gamma^k = (1-\gamma^{500})/(1-\gamma) \approx 99.3$ for $\gamma = 0.99$. Any mean max Q above this value is overestimation.

**Limits.** Reduces but does not remove the bias.""",
   dataset_md=CP_DATASET, eda=CP_EDA, prep_md=CP_PREP_MD, prep="TOTAL_STEPS = 20000\nUPPER = (1 - 0.99 ** 500) / (1 - 0.99)\nprint('upper bound of the discounted return:', round(UPPER, 2))\n" + CP_PREP,
   concept="Choose with one network and score with another, so errors in the choosing network are not rewarded twice.",
   scratch_md="Same training function with `double=True`:",
   scratch='''print([line for line in inspect.getsource(deep.train_dqn).split("\\n") if "double" in line or "best" in line or "next_q" in line])''',
   params="| Parameter | Meaning |\n|---|---|\n| `double` | True uses the Double DQN target. |\n| `target_every` | Target copy interval. |",
   train_md="Train DQN and Double DQN with identical settings and seeds. The mean max Q on replay minibatches is logged every 500 steps.",
   train='''dqn_runs, ddqn_runs = {}, {}
for seed in SEEDS:
    dqn_runs[seed] = deep.train_dqn(seed=seed, total_steps=TOTAL_STEPS, double=False)
    ddqn_runs[seed] = deep.train_dqn(seed=seed, total_steps=TOTAL_STEPS, double=True)
    print("seed", seed, "| last 10 training returns: DQN", round(float(np.mean(dqn_runs[seed]["returns"][-10:])), 1), "| Double DQN", round(float(np.mean(ddqn_runs[seed]["returns"][-10:])), 1))
runs = ddqn_runs''',
   eval_md="Evaluation return on 20 fresh episodes per seed for both methods, and the average of the logged mean max Q over the last third of training compared with the upper bound 99.3.",
   eval='''rng_eval = np.random.default_rng(0)
random_eval = evaluate_policy(make_env, lambda s: int(rng_eval.integers(2)), episodes=20, seed=1000)
def eval_runs(rs):
    return np.array([evaluate_policy(make_env, rs[s]["policy"], episodes=20, seed=1000)["mean_return"] for s in SEEDS])
dqn_eval, ddqn_eval = eval_runs(dqn_runs), eval_runs(ddqn_runs)
def late_q(rs):
    vals = []
    for s in SEEDS:
        log = np.array(rs[s]["q_log"])
        vals.append(log[len(log) * 2 // 3:, 1].mean())
    return np.array(vals)
q_dqn, q_ddqn = late_q(dqn_runs), late_q(ddqn_runs)
display(pd.DataFrame({"eval_return DQN": dqn_eval, "eval_return Double DQN": ddqn_eval, "mean max Q DQN": q_dqn, "mean max Q Double DQN": q_ddqn}, index=SEEDS).round(1))
print("upper bound of the true discounted return:", round(UPPER, 1))
means = ddqn_eval
METRICS = {"eval_mean_return_double_dqn": float(ddqn_eval.mean()), "eval_mean_return_dqn": float(dqn_eval.mean()), "eval_std_across_seeds_double_dqn": float(ddqn_eval.std(ddof=1)),
           "mean_max_q_double_dqn": float(q_ddqn.mean()), "mean_max_q_dqn": float(q_dqn.mean()), "discounted_return_upper_bound": float(UPPER), "random_policy_mean_return": float(random_eval["mean_return"])}
CONFIG = {"total_steps": TOTAL_STEPS, "seeds": 3, "target_every": 250, "gamma": 0.99}
EXTRA = {"per_seed_eval_double": [float(x) for x in ddqn_eval], "per_seed_eval_dqn": [float(x) for x in dqn_eval]}''',
   viz='''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, rs, color in [("DQN", dqn_runs, "C0"), ("Double DQN", ddqn_runs, "C1")]:
    curves = np.array([deep.curve_on_grid(rs[s]["steps"], rs[s]["returns"], GRID, window=10) for s in SEEDS])
    axes[0].plot(GRID, np.nanmean(curves, axis=0), color=color, label=name)
    axes[0].fill_between(GRID, np.nanmean(curves, axis=0) - np.nanstd(curves, axis=0), np.nanmean(curves, axis=0) + np.nanstd(curves, axis=0), color=color, alpha=0.15)
    ql = np.array([np.array(rs[s]["q_log"])[:, 1] for s in SEEDS]).mean(axis=0)
    axes[1].plot(np.array(rs[0]["q_log"])[:, 0], ql, color=color, label=name)
axes[1].axhline(UPPER, color="red", linestyle="--", label="upper bound of true value"); axes[1].set_xlabel("Environment steps"); axes[1].set_ylabel("Mean max Q on replay samples"); axes[1].legend()
axes[0].set_xlabel("Environment steps"); axes[0].set_ylabel("Return (10 episode mean)"); axes[0].legend(); axes[0].set_title("Returns")
plt.tight_layout(); save_figure(fig, NOTEBOOK_ID, "curves"); plt.show()''',
   interpret_md="If the mean max Q exceeds the upper bound the estimate is optimistic. Whether the policy return improves is a separate question.\n\n**What the committed run shows (3 seeds, 40 960 steps, values from `experiments/results/double_dqn.json`):** the mean max Q of both DQN (about 47) and Double DQN (about 46) stays below the analytic bound of 99.3, so **overestimation was not demonstrated in this setup**. The evaluation return of Double DQN (about 114) is lower than that of DQN (about 354), with a large spread across seeds for both. With three seeds and a short budget this is not evidence that Double DQN is worse in general. It does mean that this notebook does not reproduce the benefit reported in the paper, and that the result is sensitive to seeds and budget.",
   interpret='''print("share of logged points above the upper bound, DQN:", round(float(np.mean([np.array(dqn_runs[s]["q_log"])[:, 1] > UPPER for s in SEEDS])), 3))
print("share of logged points above the upper bound, Double DQN:", round(float(np.mean([np.array(ddqn_runs[s]["q_log"])[:, 1] > UPPER for s in SEEDS])), 3))
print("evaluation return DQN:", np.round(dqn_eval, 1), "| Double DQN:", np.round(ddqn_eval, 1))''',
   experiments=[
       curve_exp("Target update interval", "Double DQN with a faster target.", '[("every 100", dict(target_every=100)), ("every 500", dict(target_every=500))]',
                 "deep.train_dqn(seed=seed, total_steps=12000, double=True, **kw)", "Two seeds and 12 000 steps."),
       curve_exp("Discount factor", "A lower discount lowers the upper bound of the true value, so overestimation is easier to see.",
                 '[("gamma 0.9", dict(gamma=0.9)), ("gamma 0.99", dict(gamma=0.99))]',
                 "deep.train_dqn(seed=seed, total_steps=12000, double=True, **kw)", "Two seeds and 12 000 steps."),
       curve_exp("Network width", "Hidden units 32 and 128.", '[("32", dict(hidden=32)), ("128", dict(hidden=128))]',
                 "deep.train_dqn(seed=seed, total_steps=12000, double=True, **kw)", "Two seeds and 12 000 steps.")],
   discussion="van Hasselt et al. show that Q estimates become too high with plain DQN. CartPole gives an upper bound we can compare against, which turns the claim into a measurable check. The policy return comparison has high seed variance, so differences may be noise.",
   improvements="Use more seeds, dueling networks and prioritised replay, and test on a task with a harder overestimation setting.",
   takeaways=["Double DQN changes only the target.", "Check values against a known bound.", "Better values do not always give better returns."],
   use_cases="Any DQN setting.", limitations="Reduces, does not remove, the bias.", next="REINFORCE.", limits_extra=DEEP_EXTRA_LIMITS)

rl("08", PG, id="reinforce", title="REINFORCE: Policy Gradient on CartPole", level="Advanced", minutes=2,
   tagline="Adjust a stochastic policy directly in the direction of higher return.", name="REINFORCE", dataset="CartPole-v1",
   objectives=["The policy gradient theorem", "Monte Carlo returns as the learning signal", "Why a baseline cuts variance", "Why episode based updates are sample inefficient"],
   problem=dict(context="Some tasks are easier to solve by optimising the policy than by learning values.",
                problem="Learn a stochastic policy that balances CartPole.",
                matters="Policy gradients are the base of PPO and most modern deep RL.",
                why_ml="The policy is a neural network trained from sampled returns.",
                why_algo="REINFORCE is the simplest policy gradient method."),
   papers=["williams1992", "henderson2018"],
   paper_relation="Williams (1992) derives REINFORCE. Henderson et al. report that policy gradient results vary strongly with seeds.",
   math=r"""**Objective.** $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$.

**Gradient.** $\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\Big]$.

**Baseline.** Subtracting any $b(s)$ that does not depend on the action keeps the gradient unbiased and can lower its variance. We normalise returns within each episode (mean zero, unit std), which is a simple baseline and also rescales the gradient.

**Update.** Gradient ascent on $J$ using one episode at a time.

**Limits.** High variance, on policy, needs full episodes.""",
   dataset_md=CP_DATASET, eda=CP_EDA, prep_md=CP_PREP_MD.replace("Evaluation uses", "Training is counted in episodes and converted to environment steps. Evaluation uses"),
   prep="TOTAL_STEPS = 60000\n" + CP_PREP,
   concept="Increase the probability of actions that were followed by a high return and decrease it for those followed by a low return.",
   scratch_md="Source of the training function:",
   scratch='''print(inspect.getsource(deep.train_reinforce))''',
   params="| Parameter | Meaning |\n|---|---|\n| `lr` | Adam learning rate. |\n| `gamma` | Discount. |\n| `baseline` | `none` or `norm`. |\n| `episodes` | Training episodes. |",
   train_md="Three seeds, 400 episodes each, return normalisation on. The curve is plotted against environment steps, so the number of steps differs between seeds.",
   train=DEEP_RUN.replace("__CALL__", "deep.train_reinforce(seed=seed, episodes=400)"),
   eval_md="Greedy (argmax) policy evaluated on 20 fresh episodes per seed. Floor: random policy.",
   eval=DEEP_EVAL + '''
CONFIG = {"episodes": 400, "lr": 5e-3, "gamma": 0.99, "baseline": "norm", "seeds": 3}''',
   viz=DEEP_VIZ.replace("np.linspace", "np.linspace").replace("GRID", "GRID"), interpret_md="Look at the spread between seeds.", interpret=DEEP_INTERP,
   experiments=[
       curve_exp("With and without the baseline", "Return normalisation on or off.", '[("no baseline", dict(baseline="none")), ("normalised returns", dict(baseline="norm"))]',
                 "deep.train_reinforce(seed=seed, episodes=250, **kw)", "Two seeds and 250 episodes."),
       curve_exp("Learning rate", "1e-3, 5e-3 and 2e-2.", '[("1e-3", dict(lr=1e-3)), ("5e-3", dict(lr=5e-3)), ("2e-2", dict(lr=2e-2))]',
                 "deep.train_reinforce(seed=seed, episodes=250, **kw)", "Two seeds and 250 episodes."),
       curve_exp("Discount factor", "0.9, 0.99 and 1.0.", '[("0.9", dict(gamma=0.9)), ("0.99", dict(gamma=0.99)), ("1.0", dict(gamma=1.0))]',
                 "deep.train_reinforce(seed=seed, episodes=250, **kw)", "Two seeds and 250 episodes.")],
   discussion="Williams shows the estimator is unbiased. High variance is its known cost. The baseline experiment and the spread across seeds show this on CartPole. We cannot separate the variance effect from the gradient scale effect, because normalisation changes both.",
   improvements="Use a learned value baseline (actor critic), batch several episodes per update, and more seeds.",
   takeaways=["REINFORCE follows the gradient of expected return.", "Baselines cut variance.", "Learning is episode based and noisy."],
   use_cases="Teaching, small control tasks.", limitations="High variance, sample inefficient.", next="Actor-Critic.", limits_extra=DEEP_EXTRA_LIMITS)

rl("09", PG, id="actor_critic", title="Actor-Critic: Learning a Baseline Online", level="Advanced", minutes=2,
   tagline="Update the policy after every step using a learned value function as critic.", name="ActorCritic", dataset="CartPole-v1",
   objectives=["The TD error as an advantage estimate", "Why a critic lowers variance but adds bias", "Two learning rates", "Instability of one step actor critic"],
   problem=dict(context="Waiting for the end of an episode wastes information.",
                problem="Balance CartPole with updates after each step.",
                matters="Actor critic is the structure of most modern policy methods.",
                why_ml="Both the policy and the value function are learned.",
                why_algo="One step actor critic is the simplest online version."),
   papers=["konda2003", "sutton2018"],
   paper_relation="Konda and Tsitsiklis analyse actor critic convergence under conditions that our small network version does not meet. Sutton and Barto (not verified in the build environment) give the algorithm in textbook form.",
   math=r"""**Critic.** Learn $V_w(s)$ with the TD error $\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)$, minimising $\delta_t^2$.

**Actor.** $\theta \leftarrow \theta + \alpha\,\delta_t\,\nabla_\theta \log \pi_\theta(a_t|s_t)$. The TD error is an estimate of the advantage.

**Bias and variance.** The critic bootstraps, so the gradient is biased but has lower variance than REINFORCE.

**Limits.** Two learning rates to tune, instability when the critic is poor.""",
   dataset_md=CP_DATASET, eda=CP_EDA, prep_md=CP_PREP_MD, prep="TOTAL_STEPS = 60000\n" + CP_PREP,
   concept="The critic says how good the situation is, the actor moves toward actions that turned out better than the critic expected.",
   scratch_md="Source of the training function:",
   scratch='''print(inspect.getsource(deep.train_actor_critic))''',
   params="| Parameter | Meaning |\n|---|---|\n| `lr_actor`, `lr_critic` | Learning rates. |\n| `gamma` | Discount. |\n| `total_steps` | Environment steps. |",
   train_md="Three seeds, 60 000 steps each. One gradient step per environment step for each network.",
   train=DEEP_RUN.replace("__CALL__", "deep.train_actor_critic(seed=seed, total_steps=TOTAL_STEPS)"),
   eval_md="Greedy policy on 20 fresh episodes per seed. Floor: random policy.",
   eval=DEEP_EVAL + '''
CONFIG = {"total_steps": TOTAL_STEPS, "lr_actor": 1e-3, "lr_critic": 5e-3, "gamma": 0.99, "seeds": 3}''',
   viz=DEEP_VIZ, interpret_md="One step actor critic is often unstable, so check the individual seeds.", interpret=DEEP_INTERP,
   experiments=[
       curve_exp("Critic learning rate", "Slow, equal and fast critic.", '[("1e-3", dict(lr_critic=1e-3)), ("5e-3", dict(lr_critic=5e-3)), ("2e-2", dict(lr_critic=2e-2))]',
                 "deep.train_actor_critic(seed=seed, total_steps=25000, **kw)", "Two seeds and 25 000 steps."),
       curve_exp("Actor learning rate", "3e-4, 1e-3 and 3e-3.", '[("3e-4", dict(lr_actor=3e-4)), ("1e-3", dict(lr_actor=1e-3)), ("3e-3", dict(lr_actor=3e-3))]',
                 "deep.train_actor_critic(seed=seed, total_steps=25000, **kw)", "Two seeds and 25 000 steps."),
       curve_exp("Discount factor", "0.9 versus 0.99.", '[("0.9", dict(gamma=0.9)), ("0.99", dict(gamma=0.99))]',
                 "deep.train_actor_critic(seed=seed, total_steps=25000, **kw)", "Two seeds and 25 000 steps.")],
   discussion="Konda and Tsitsiklis show convergence under specific conditions (two time scales). Our learning rate experiments look at the ratio of the two rates in practice.",
   improvements="Use n step returns or GAE, batch several environments (A2C), entropy bonus and more seeds.",
   takeaways=["The critic gives a low variance advantage.", "Two learning rates must be balanced.", "Check seeds one by one."],
   use_cases="Online control.", limitations="Unstable, biased.", next="PPO.", limits_extra=DEEP_EXTRA_LIMITS)

rl("10", PG, id="ppo", title="Proximal Policy Optimization on CartPole", level="Advanced", minutes=2,
   tagline="Take several safe gradient steps per batch by clipping the policy ratio.", name="PPO", dataset="CartPole-v1",
   objectives=["The clipped surrogate objective", "Generalised advantage estimation", "Why several epochs per batch are safe with clipping", "How to log the clip fraction"],
   problem=dict(context="Large policy updates can destroy performance.",
                problem="Balance CartPole with a stable policy gradient method.",
                matters="PPO is the most used on policy deep RL algorithm.",
                why_ml="The policy and value networks are trained on batches.",
                why_algo="PPO is the practical default for many tasks."),
   papers=["schulman2017", "henderson2018", "agarwal2021"],
   paper_relation="Schulman et al. introduce PPO. Our version is minimal: one environment, no observation normalisation, no learning rate schedule. It is not a reproduction of their experiments. Agarwal et al. argue that few runs can mislead, and we use three seeds only.",
   math=r"""**Ratio.** $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{old}}(a_t|s_t)$.

**Clipped objective.** $L^{CLIP} = \mathbb{E}\big[\min\big(r_t \hat A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\big]$ with $\epsilon = 0.2$.

**GAE.** $\hat A_t = \sum_{l\ge 0}(\gamma\lambda)^l \delta_{t+l}$, $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

**Total loss.** $-L^{CLIP} + c_v (V - \hat R)^2 - c_e H(\pi)$.

**Limits.** Many implementation details matter, sensitive to the learning rate.""",
   dataset_md=CP_DATASET, eda=CP_EDA, prep_md=CP_PREP_MD, prep="TOTAL_STEPS = 60000\n" + CP_PREP,
   concept="Collect a batch with the current policy, then improve it for a few epochs, but stop pushing in directions where the new policy already differs a lot from the old one.",
   scratch_md="Source of the training function (`ml_lab.rl.deep.train_ppo`). Truncation at the time limit is treated like a terminal state, which adds a small bias.",
   scratch='''print(inspect.getsource(deep.train_ppo))''',
   params="| Parameter | Meaning |\n|---|---|\n| `rollout` | Steps per batch. |\n| `epochs`, `minibatch` | Optimisation passes and minibatch size. |\n| `clip` | Ratio clip range. |\n| `lam` | GAE parameter. |\n| `ent_coef` | Entropy bonus. |",
   train_md="Three seeds, 60 000 steps, rollouts of 1024 steps, 10 epochs, minibatch 128, learning rate 1e-3.",
   train=DEEP_RUN.replace("__CALL__", "deep.train_ppo(seed=seed, total_steps=TOTAL_STEPS)"),
   eval_md="Greedy policy on 20 fresh episodes per seed. Floor: random policy.",
   eval=DEEP_EVAL + '''
CONFIG = {"total_steps": TOTAL_STEPS, "rollout": 1024, "epochs": 10, "minibatch": 128, "lr": 1e-3, "clip": 0.2, "gamma": 0.99, "lambda": 0.95, "seeds": 3}''',
   viz=DEEP_VIZ, interpret_md="The clip fraction shows how often the ratio left the trust region.",
   interpret=DEEP_INTERP + '''
clip = np.array([[c for _, c in runs[s]["clip_log"]] for s in SEEDS])
print("mean clip fraction per rollout, by seed:", np.round(clip.mean(axis=1), 3))''',
   experiments=[
       curve_exp("Clip range", "0.05, 0.2 and 0.5.", '[("0.05", dict(clip=0.05)), ("0.2", dict(clip=0.2)), ("0.5", dict(clip=0.5))]',
                 "deep.train_ppo(seed=seed, total_steps=30720, **kw)", "Two seeds and about 30 000 steps."),
       curve_exp("Epochs per batch", "1, 4 and 10.", '[("1", dict(epochs=1)), ("4", dict(epochs=4)), ("10", dict(epochs=10))]',
                 "deep.train_ppo(seed=seed, total_steps=30720, **kw)", "Two seeds and about 30 000 steps."),
       curve_exp("GAE lambda", "0.0 (one step TD) versus 0.95.", '[("lambda 0.0", dict(lam=0.0)), ("lambda 0.95", dict(lam=0.95))]',
                 "deep.train_ppo(seed=seed, total_steps=30720, **kw)", "Two seeds and about 30 000 steps.")],
   discussion="Schulman et al. report PPO as stable and simple. Our small test cannot confirm their comparison with other methods, but the curves of three seeds show how stable the minimal version is on CartPole. Agarwal et al. recommend interval estimates, which need more seeds than we use.",
   improvements="Observation normalisation, learning rate decay, several parallel environments, and at least 10 seeds with intervals.",
   takeaways=["Clipping limits the policy change.", "GAE trades bias for variance.", "Details and seeds matter."],
   use_cases="General on policy deep RL.", limitations="Implementation details matter.", next="Fundamentals.", limits_extra=DEEP_EXTRA_LIMITS)
