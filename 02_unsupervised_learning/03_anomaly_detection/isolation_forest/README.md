# Isolation Forest: Finding Malignant Outliers Among Benign Samples

Isolate anomalies with random splits and score by path length.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Intermediate |
| Dataset | `breast_cancer` (see `datasets/catalog`) |
| Notebook | [`10_isolation_forest.ipynb`](10_isolation_forest.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- Why anomalies are isolated in few splits
- How the anomaly score is normalised
- How to evaluate anomaly detection with rare labels
- How contamination affects the threshold

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 10_isolation_forest.ipynb
```

Results are saved to `experiments/results/isolation_forest.json`.
