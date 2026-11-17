"""Model result charts."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import learning_curve


def plot_pred_vs_actual(y_true, y_pred, unit=""):
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(y_true, y_pred, alpha=0.6, s=18, color="#1f4e79")
    low, high = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    axis.plot([low, high], [low, high], "--", color="#c0504d", label="perfect prediction")
    axis.set_xlabel(f"Actual {unit}")
    axis.set_ylabel(f"Predicted {unit}")
    axis.set_title("Predicted versus actual (test set)")
    axis.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, unit=""):
    residual = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residual, alpha=0.6, s=18, color="#1f4e79")
    axes[0].axhline(0, color="#c0504d", linestyle="--")
    axes[0].set_xlabel(f"Predicted {unit}")
    axes[0].set_ylabel(f"Residual {unit}")
    axes[0].set_title("Residuals versus prediction")
    axes[1].hist(residual, bins=25, color="#1f4e79")
    axes[1].set_xlabel(f"Residual {unit}")
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Residual distribution")
    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, labels=None, normalize="true"):
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, normalize=normalize, cmap="Blues", ax=axis, colorbar=False
    )
    axis.grid(False)
    axis.set_title("Confusion matrix (row normalised)")
    fig.tight_layout()
    return fig


def plot_roc_pr(estimator, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_estimator(estimator, X_test, y_test, ax=axes[1])
    axes[1].set_title("Precision recall curve")
    fig.tight_layout()
    return fig


def plot_importance(names, values, title="Feature importance", top=15, xlabel="Importance"):
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, axis = plt.subplots(figsize=(7, 0.32 * len(order) + 1.5))
    axis.barh(np.array(names)[order], np.array(values)[order], color="#1f4e79")
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    fig.tight_layout()
    return fig


def plot_learning_curve(estimator, X, y, scoring, cv=5, seed=42):
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, train_sizes=np.linspace(0.15, 1.0, 7), random_state=seed, n_jobs=1
    )
    fig, axis = plt.subplots(figsize=(7, 4.2))
    for scores, label, color in [(train_scores, "train", "#c0504d"), (val_scores, "cross validation", "#1f4e79")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        axis.plot(sizes, mean, "o-", label=label, color=color)
        axis.fill_between(sizes, mean - std, mean + std, alpha=0.15, color=color)
    axis.set_xlabel("Training rows")
    axis.set_ylabel(scoring)
    axis.set_title("Learning curve (mean and one std)")
    axis.legend()
    fig.tight_layout()
    return fig


def plot_decision_boundary(estimator_factory, X, y, feature_pair=(0, 1), names=None, resolution=200):
    """Boundary in two chosen features."""
    X2 = np.asarray(X)[:, list(feature_pair)]
    model = estimator_factory().fit(X2, y)
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    zone = model.predict(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
    fig, axis = plt.subplots(figsize=(6, 4.8))
    axis.contourf(grid_x, grid_y, zone, alpha=0.25, cmap="coolwarm")
    axis.scatter(X2[:, 0], X2[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=22)
    if names:
        axis.set_xlabel(names[feature_pair[0]])
        axis.set_ylabel(names[feature_pair[1]])
    axis.set_title("Decision regions in two features")
    fig.tight_layout()
    return fig


def plot_clusters_2d(embedding, labels, title="Clusters", noise_label=-1):
    fig, axis = plt.subplots(figsize=(6.5, 5))
    labels = np.asarray(labels)
    for value in np.unique(labels):
        mask = labels == value
        color = "#bbbbbb" if value == noise_label else None
        name = "noise" if value == noise_label else f"cluster {value}"
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=14, alpha=0.8, label=name, color=color)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.set_title(title)
    if len(np.unique(labels)) <= 12:
        axis.legend(markerscale=1.5, fontsize=8)
    fig.tight_layout()
    return fig


def plot_reward_curves(curves_by_label, xlabel="Episode", ylabel="Episode return", window=10, title="Learning curves"):
    """Mean and std across seeds."""
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    for label, curves in curves_by_label.items():
        length = min(len(c) for c in curves)
        stacked = np.array([c[:length] for c in curves], dtype=float)
        if length >= window:
            kernel = np.ones(window) / window
            stacked = np.array([np.convolve(row, kernel, mode="valid") for row in stacked])
        mean, std = stacked.mean(axis=0), stacked.std(axis=0)
        x = np.arange(len(mean)) + (window - 1 if length >= window else 0)
        axis.plot(x, mean, label=f"{label} ({len(curves)} seeds)")
        axis.fill_between(x, mean - std, mean + std, alpha=0.2)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    return fig
