"""Save a fitted model with the metadata needed to use it safely."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn


def save_artifact(model, folder, name, feature_names, task, metrics=None, training_data="not stated"):
    """Write model.joblib and model_card.json into folder. Return both paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / f"{name}.joblib"
    joblib.dump(model, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    card = {
        "name": name,
        "task": task,
        "feature_names": list(feature_names),
        "metrics": metrics or {},
        "training_data": training_data,
        "sklearn_version": sklearn.__version__,
        "sha256": digest,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Load only files you created yourself. joblib files can run code when loaded.",
    }
    card_path = folder / f"{name}_card.json"
    card_path.write_text(json.dumps(card, indent=2))
    return model_path, card_path


def load_artifact(folder, name):
    """Load model and card. Refuse the file if the checksum does not match."""
    folder = Path(folder)
    model_path = folder / f"{name}.joblib"
    card = json.loads((folder / f"{name}_card.json").read_text())
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != card["sha256"]:
        raise ValueError("checksum mismatch: model file changed after saving")
    return joblib.load(model_path), card


def predict_records(model, card, records):
    """Predict for a list of dicts. Check that every feature is present."""
    frame = pd.DataFrame(records)
    missing = [c for c in card["feature_names"] if c not in frame.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    frame = frame[card["feature_names"]]
    result = {"prediction": model.predict(frame).tolist()}
    if hasattr(model, "predict_proba"):
        result["probability"] = model.predict_proba(frame)[:, -1].round(6).tolist()
    return result
