# Gaussian Mixture Models: Soft Clusters of Cars

Fit a mixture of Gaussians with EM and choose the model with BIC.

| Item | Value |
|---|---|
| Family | Unsupervised learning |
| Level | Intermediate |
| Dataset | `auto_mpg` (see `datasets/catalog`) |
| Notebook | [`05_gaussian_mixture.ipynb`](05_gaussian_mixture.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How EM alternates responsibilities and parameter updates
- How covariance type changes the shape of clusters
- How BIC balances fit and complexity
- How soft assignments differ from hard ones

## Run it

```bash
pip install -e ".[unsupervised,development]"
jupyter lab 05_gaussian_mixture.ipynb
```

Results are saved to `experiments/results/gaussian_mixture.json`.
