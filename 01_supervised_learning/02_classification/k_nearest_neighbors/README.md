# K-Nearest Neighbors: Classifying Wine Cultivars

Classify by voting among the closest examples and see why feature scaling decides everything.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Beginner |
| Dataset | `wine` (see `datasets/catalog`) |
| Notebook | [`02_k_nearest_neighbors.ipynb`](02_k_nearest_neighbors.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How the nearest neighbour rule works
- Why unscaled features break distance based models
- How k trades bias against variance
- Why high dimensions hurt KNN

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 02_k_nearest_neighbors.ipynb
```

Results are saved to `experiments/results/k_nearest_neighbors.json`.
