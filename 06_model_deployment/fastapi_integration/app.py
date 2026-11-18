"""Minimal prediction API. Run: uvicorn app:app --app-dir 06_model_deployment/fastapi_integration"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_lab.deployment import load_artifact, predict_records

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "model_serialization" / "artifacts"))
MODEL_NAME = os.environ.get("MODEL_NAME", "breast_cancer_logistic")

app = FastAPI(title="ml_lab prediction API", version="0.1.0")
model, card = load_artifact(MODEL_DIR, MODEL_NAME)


class Batch(BaseModel):
    records: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model": card["name"], "sha256": card["sha256"][:12]}


@app.get("/model-card")
def model_card():
    return card


@app.post("/predict")
def predict(batch: Batch):
    if not batch.records or len(batch.records) > 1000:
        raise HTTPException(status_code=422, detail="send 1 to 1000 records")
    try:
        return predict_records(model, card, batch.records)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
