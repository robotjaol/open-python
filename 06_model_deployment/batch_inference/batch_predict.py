"""Score a CSV file. Usage: python batch_predict.py input.csv output.csv"""

import sys
from pathlib import Path

import pandas as pd

from ml_lab.deployment import load_artifact, predict_records

model_dir = Path(__file__).resolve().parents[1] / "model_serialization" / "artifacts"
model, card = load_artifact(model_dir, "breast_cancer_logistic")
data = pd.read_csv(sys.argv[1])
scores = predict_records(model, card, data.to_dict("records"))
data["prediction"] = scores["prediction"]
data["probability"] = scores["probability"]
data.to_csv(sys.argv[2], index=False)
print("rows scored:", len(data))
