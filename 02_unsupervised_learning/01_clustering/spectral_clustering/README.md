# Spectral Clustering: Graph Cuts on Digit Images

Cluster with the eigenvectors of a neighbour graph.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Advanced |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`06_spectral_clustering.ipynb`](06_spectral_clustering.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How a similarity graph defines clusters
- What the graph Laplacian eigenvectors do
- How the eigengap suggests k
- Why spectral methods need a good graph

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 06_spectral_clustering.ipynb
```

Results are saved to `experiments/results/spectral_clustering.json`.
