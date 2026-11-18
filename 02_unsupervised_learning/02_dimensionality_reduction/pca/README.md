# PCA: Compressing Breast Cancer Measurements

Find the directions of largest variance and keep the few that matter.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Beginner |
| Dataset | `breast_cancer` (see `datasets/catalog`) |
| Notebook | [`07_pca.ipynb`](07_pca.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How PCA relates to the SVD
- How to read explained variance
- Why scaling decides the result
- How to fit PCA without leakage

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 07_pca.ipynb
```

Results are saved to `experiments/results/pca.json`.
