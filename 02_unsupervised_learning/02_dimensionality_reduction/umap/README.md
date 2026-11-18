# UMAP: A Faster Map That Can Transform New Rows

Embed with a fuzzy neighbour graph and apply the map to unseen data.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Intermediate |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`09_umap.ipynb`](09_umap.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How UMAP builds a fuzzy graph
- How n_neighbors and min_dist shape the map
- How to transform held out rows
- How UMAP compares with t-SNE and PCA

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 09_umap.ipynb
```

Results are saved to `experiments/results/umap.json`.
