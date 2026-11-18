# Hierarchical Clustering: A Tree of Wines

Build a dendrogram and cut it at the level you need.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Beginner |
| Dataset | `wine` (see `datasets/catalog`) |
| Notebook | [`02_hierarchical_clustering.ipynb`](02_hierarchical_clustering.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How agglomerative merging works
- How linkage choice changes the tree
- How to read a dendrogram
- What the cophenetic correlation measures

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 02_hierarchical_clustering.ipynb
```

Results are saved to `experiments/results/hierarchical_clustering.json`.
