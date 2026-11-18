import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_lab.deployment import drift_report, load_artifact, predict_records, save_artifact
from ml_lab.deployment.monitoring import population_stability_index


def make_model(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = (X["a"] + X["b"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    save_artifact(model, tmp_path, "m", X.columns, "classification")
    return X


def test_roundtrip(tmp_path):
    X = make_model(tmp_path)
    model, card = load_artifact(tmp_path, "m")
    out = predict_records(model, card, X.head(5).to_dict("records"))
    assert len(out["prediction"]) == 5 and len(out["probability"]) == 5


def test_checksum_blocks_tampering(tmp_path):
    make_model(tmp_path)
    with open(tmp_path / "m.joblib", "ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(tmp_path, "m")


def test_missing_feature_is_rejected(tmp_path):
    make_model(tmp_path)
    model, card = load_artifact(tmp_path, "m")
    with pytest.raises(ValueError, match="missing"):
        predict_records(model, card, [{"a": 1.0, "b": 2.0}])


def test_psi_zero_for_same_data_and_large_for_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(size=5000)
    assert population_stability_index(base, base) < 1e-6
    assert population_stability_index(base, base + 2.0) > 0.25


def test_drift_report_flags_shifted_column():
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    new = train.copy()
    new["b"] = new["b"] + 3
    report = drift_report(train, new)
    assert report.iloc[0]["feature"] == "b" and report.iloc[0]["level"] == "large"
