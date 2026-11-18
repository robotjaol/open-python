import numpy as np

from ml_lab.rl import tabular


def test_value_iteration_frozenlake_optimal_success_probability():
    P, n_s, n_a = tabular.transition_model("FrozenLake-v1", is_slippery=True)
    values, policy, _ = tabular.value_iteration(P, n_s, n_a, gamma=1.0)
    # exact dynamic programming answer for the 4x4 slippery map is about 0.8235
    assert abs(float(values[0]) - 0.8235) < 0.005


def test_always_left_policy_is_much_worse_than_optimal():
    P, n_s, n_a = tabular.transition_model("FrozenLake-v1", is_slippery=True)
    values, _ = tabular.policy_evaluation(P, np.zeros(n_s, dtype=int), n_s, gamma=1.0)
    assert float(values[0]) < 0.1
