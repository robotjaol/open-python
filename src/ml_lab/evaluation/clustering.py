"""Metrics for clustering."""

import numpy as np
from sklearn import metrics


def internal_scores(X, labels):
    """Scores that need no ground truth."""
    mask = np.asarray(labels) != -1
    unique = np.unique(np.asarray(labels)[mask])
    result = {
        "clusters": int(len(unique)),
        "noise_fraction": float(1 - mask.mean()),
    }
    if len(unique) >= 2 and mask.sum() > len(unique):
        result["Silhouette"] = metrics.silhouette_score(X[mask], np.asarray(labels)[mask])
        result["Davies_Bouldin"] = metrics.davies_bouldin_score(X[mask], np.asarray(labels)[mask])
        result["Calinski_Harabasz"] = metrics.calinski_harabasz_score(X[mask], np.asarray(labels)[mask])
    return result


def external_scores(true_labels, labels):
    """Only use with independent labels."""
    return {
        "ARI": metrics.adjusted_rand_score(true_labels, labels),
        "NMI": metrics.normalized_mutual_info_score(true_labels, labels),
    }
