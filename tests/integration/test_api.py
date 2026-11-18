import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "06_model_deployment"


@pytest.fixture(scope="module")
def client():
    subprocess.run([sys.executable, str(DEPLOY / "model_serialization" / "make_artifact.py")], check=True)
    spec = importlib.util.spec_from_file_location("api_app", DEPLOY / "fastapi_integration" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return TestClient(module.app), module.card


def test_health_and_card(client):
    c, card = client
    assert c.get("/health").json()["status"] == "ok"
    assert len(c.get("/model-card").json()["feature_names"]) == 30


def test_predict_valid_record(client):
    c, card = client
    record = {name: 1.0 for name in card["feature_names"]}
    r = c.post("/predict", json={"records": [record]})
    assert r.status_code == 200 and 0 <= r.json()["probability"][0] <= 1


def test_predict_rejects_missing_feature_and_empty(client):
    c, _ = client
    assert c.post("/predict", json={"records": [{"radius_mean": 1}]}).status_code == 422
    assert c.post("/predict", json={"records": []}).status_code == 422
