# HDBSCAN: Clusters of Varying Density

Build a hierarchy of density clusters and keep the stable ones.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Advanced |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`04_hdbscan.ipynb`](04_hdbscan.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- Mutual reachability distance
- How stability selects clusters
- Why min_cluster_size replaces eps
- How soft probabilities describe membership

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 04_hdbscan.ipynb
```

Results are saved to `experiments/results/hdbscan.json`.
