"""Train a small model on breast_cancer and save it with a model card."""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ml_lab.data import load_dataset
from ml_lab.deployment import save_artifact
from ml_lab.preprocessing import make_model_pipeline, split_three_way

bundle = load_dataset("breast_cancer")
X, y = bundle.X, bundle.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="classification", seed=42)
model = make_model_pipeline(Xtr, LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
out = Path(__file__).parent / "artifacts"
paths = save_artifact(model, out, "breast_cancer_logistic", X.columns, "classification",
                      {"test_ROC_AUC": round(float(auc), 4)}, "UCI Breast Cancer Wisconsin Diagnostic (sklearn copy)")
print("saved", [str(p.name) for p in paths], "test ROC-AUC", round(float(auc), 4))
