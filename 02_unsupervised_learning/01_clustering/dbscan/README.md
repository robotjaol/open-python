# DBSCAN: Density Clusters and Noise in Digit Images

Find dense regions of any shape and mark the rest as noise.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Intermediate |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`03_dbscan.ipynb`](03_dbscan.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- Core, border and noise points
- How eps and min_samples are chosen
- Why DBSCAN needs scaled, low dimensional data
- How noise is reported

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 03_dbscan.ipynb
```

Results are saved to `experiments/results/dbscan.json`.
