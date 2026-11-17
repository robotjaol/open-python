# Gradient Boosting: Sequential Trees on Disease Progression

Build an additive model by fitting each tree to the errors of the previous ones.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `diabetes` (see `datasets/catalog`) |
| Notebook | [`07_gradient_boosting_regressor.ipynb`](07_gradient_boosting_regressor.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How boosting fits residuals stage by stage
- The roles of learning rate, depth and number of trees
- How to detect overfitting with staged predictions

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 07_gradient_boosting_regressor.ipynb
```

Results are saved to `experiments/results/gradient_boosting_regressor.json`.
