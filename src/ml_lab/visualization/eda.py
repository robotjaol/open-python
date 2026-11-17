"""EDA charts."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_distributions(frame, columns=None, max_columns=12, bins=30):
    columns = (columns or frame.select_dtypes(include="number").columns.tolist())[:max_columns]
    cols = 4
    rows = int(np.ceil(len(columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows))
    for axis, column in zip(np.atleast_1d(axes).ravel(), columns):
        axis.hist(frame[column].dropna(), bins=bins, color="#1f4e79", alpha=0.85)
        axis.set_title(column, fontsize=9)
    for axis in np.atleast_1d(axes).ravel()[len(columns):]:
        axis.axis("off")
    fig.suptitle("Feature distributions (count per bin)", y=1.02)
    fig.tight_layout()
    return fig


def plot_correlation(frame, max_columns=15):
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] > max_columns:
        top = numeric.var().sort_values(ascending=False).index[:max_columns]
        numeric = numeric[top]
    fig, axis = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, annot=numeric.shape[1] <= 10, fmt=".2f", ax=axis)
    axis.set_title("Pearson correlation (unitless)")
    fig.tight_layout()
    return fig


def plot_target(series, task="classification"):
    fig, axis = plt.subplots(figsize=(5.5, 3.5))
    if task == "classification":
        series.value_counts().sort_index().plot.bar(ax=axis, color="#1f4e79")
        axis.set_ylabel("Rows")
    else:
        axis.hist(series, bins=30, color="#1f4e79")
        axis.set_ylabel("Rows")
    axis.set_xlabel(series.name)
    axis.set_title(f"Target distribution: {series.name}")
    fig.tight_layout()
    return fig


def outlier_table(frame, factor=1.5):
    """Count values outside IQR fences."""
    import pandas as pd

    rows = []
    for column in frame.select_dtypes(include="number").columns:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        spread = q3 - q1
        low, high = q1 - factor * spread, q3 + factor * spread
        count = int(((frame[column] < low) | (frame[column] > high)).sum())
        rows.append({"feature": column, "outliers": count, "percent": round(100 * count / len(frame), 2)})
    return pd.DataFrame(rows).sort_values("outliers", ascending=False).reset_index(drop=True)
