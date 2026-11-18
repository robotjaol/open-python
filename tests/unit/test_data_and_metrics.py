import numpy as np
import pytest

from ml_lab.data import list_datasets, load_dataset
from ml_lab.evaluation import classification_metrics, regression_metrics
from ml_lab.preprocessing import split_three_way


def test_registry_lists_datasets():
    assert "breast_cancer" in list_datasets()["name"].tolist()


def test_split_has_no_overlap_and_right_sizes():
    b = load_dataset("breast_cancer")
    Xtr, Xva, Xte, ytr, yva, yte = split_three_way(b.X, b.y, task="classification", seed=1)
    assert len(Xtr) + len(Xva) + len(Xte) == len(b.X)
    assert set(Xtr.index).isdisjoint(Xte.index) and set(Xtr.index).isdisjoint(Xva.index)
    assert abs(len(Xte) / len(b.X) - 0.2) < 0.02


def test_split_is_reproducible():
    b = load_dataset("wine")
    a = split_three_way(b.X, b.y, task="classification", seed=3)[2].index
    c = split_three_way(b.X, b.y, task="classification", seed=3)[2].index
    assert list(a) == list(c)


def test_regression_metrics_perfect_and_constant():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = regression_metrics(y, y)
    assert perfect["RMSE"] == 0 and perfect["R2"] == 1
    assert regression_metrics(y, np.full(4, y.mean()))["R2"] == pytest.approx(0, abs=1e-12)


def test_classification_metrics_known_values():
    m = classification_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert m["Accuracy"] == 0.75
