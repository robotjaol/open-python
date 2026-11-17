# HistGradientBoosting: Fast Boosting on Fuel Economy

Bin features into histograms to train boosted trees quickly and handle missing values natively.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `auto_mpg` (see `datasets/catalog`) |
| Notebook | [`08_hist_gradient_boosting.ipynb`](08_hist_gradient_boosting.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How histogram binning speeds up split search
- Early stopping with an internal validation split
- Native handling of missing values

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 08_hist_gradient_boosting.ipynb
```

Results are saved to `experiments/results/hist_gradient_boosting.json`.
