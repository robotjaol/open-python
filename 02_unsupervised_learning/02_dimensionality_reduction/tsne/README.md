# t-SNE: Mapping Digit Images in 2D

Embed high dimensional data so that neighbours stay neighbours.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Intermediate |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`08_tsne.ipynb`](08_tsne.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How t-SNE matches neighbour probabilities
- What perplexity means
- Why distances between clusters are not meaningful
- How to measure embedding quality

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 08_tsne.ipynb
```

Results are saved to `experiments/results/tsne.json`.
