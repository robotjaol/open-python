"""Simple drift checks between training data and new data."""

import numpy as np
import pandas as pd


def population_stability_index(expected, actual, bins=10):
    """PSI for one numeric feature. Bins come from the expected data."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(train_frame, new_frame, bins=10):
    """PSI per numeric column. Common rule: below 0.1 small, above 0.25 large."""
    rows = []
    for col in train_frame.select_dtypes("number").columns:
        psi = population_stability_index(train_frame[col], new_frame[col], bins)
        level = "small" if psi < 0.1 else "moderate" if psi < 0.25 else "large"
        rows.append({"feature": col, "psi": round(psi, 4), "level": level})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
