"""Summaries across runs."""

import numpy as np
import pandas as pd


def mean_std_table(runs):
    """List of metric dicts into mean and std."""
    frame = pd.DataFrame(runs)
    return pd.DataFrame({"mean": frame.mean(), "std": frame.std(ddof=1)}).round(4)


def bootstrap_interval(values, repeats=2000, seed=0, level=0.95):
    """Percentile interval of the mean."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(repeats)]
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))
