"""Metrics for regression and classification."""

import numpy as np
import pandas as pd
from sklearn import metrics


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = metrics.mean_squared_error(y_true, y_pred)
    result = {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": metrics.r2_score(y_true, y_pred),
    }
    if np.all(y_true != 0):
        result["MAPE_percent"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return result


def classification_metrics(y_true, y_pred, y_score=None):
    """Binary or multiclass scores."""
    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "macro"
    result = {
        "Accuracy": metrics.accuracy_score(y_true, y_pred),
        "Precision": metrics.precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": metrics.recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": metrics.f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_score is not None:
        if len(labels) == 2:
            score = y_score[:, 1] if np.ndim(y_score) == 2 else y_score
            result["ROC_AUC"] = metrics.roc_auc_score(y_true, score)
            result["PR_AUC"] = metrics.average_precision_score(y_true, score)
        else:
            result["ROC_AUC_ovr"] = metrics.roc_auc_score(y_true, y_score, multi_class="ovr")
    return result


def compare_models(results):
    """Dict of name to metrics into table."""
    return pd.DataFrame(results).T.round(4)


def predict_scores(model, X):
    """Probabilities or decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_estimator(model, X, y, task):
    """One call metric dict."""
    predictions = model.predict(X)
    if task == "regression":
        return regression_metrics(y, predictions)
    scores = predict_scores(model, X)
    return classification_metrics(y, predictions, scores)
