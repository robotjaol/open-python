# K-Means: Segmenting Wines by Chemistry

Group rows around centres and learn how to pick k without labels.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Beginner |
| Dataset | `wine` (see `datasets/catalog`) |
| Notebook | [`01_kmeans.ipynb`](01_kmeans.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How Lloyd's algorithm alternates assignment and update
- Why k-means needs scaled features
- How to choose k with the elbow and silhouette
- Why results depend on initialisation

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 01_kmeans.ipynb
```

Results are saved to `experiments/results/kmeans.json`.
