# 06 Model deployment

A small, tested example: train, save with a model card and checksum, serve over HTTP, score a file in batch, check drift.

| Part | File | Status |
|---|---|---|
| Serialization with checksum | `model_serialization/make_artifact.py`, `src/ml_lab/deployment/artifact.py` | Completed, unit tested |
| FastAPI service | `fastapi_integration/app.py` | Completed, tested with `TestClient` (`tests/integration/test_api.py`). Not load tested |
| Batch inference | `batch_inference/batch_predict.py` | Written, run by hand only. No automated test |
| Drift monitoring (PSI) | `src/ml_lab/deployment/monitoring.py`, `model_monitoring/drift_check.py` | Function unit tested. Script run by hand only |
| Reproducible deployment (Docker) | `reproducible_deployment/` | **Not implemented**. Docker is not available in the build environment, so nothing was built or tested |

## Run it

```bash
python 06_model_deployment/model_serialization/make_artifact.py
uvicorn app:app --app-dir 06_model_deployment/fastapi_integration --port 8000
curl localhost:8000/health
```

## Security notes

- `joblib` files can execute code when loaded. The loader checks a SHA-256 stored next to the file, which detects accidental or later changes only. It does **not** protect against someone who replaces both files. Load only models you made.
- The API has no authentication, rate limit or input size limit beyond 1000 records. Do not expose it publicly as is.
- The example model is trained on a teaching dataset and must not be used for medical decisions.
