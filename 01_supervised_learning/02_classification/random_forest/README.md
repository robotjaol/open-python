# Random Forest: Stable Churn Prediction

Average many randomised trees to cut variance, and use out-of-bag data as a free validation set.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `telco_churn` (see `datasets/catalog`) |
| Notebook | [`04_random_forest.ipynb`](04_random_forest.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- Why bagging reduces variance
- What random feature subsets add
- How out of bag error works
- Why impurity importance can mislead

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 04_random_forest.ipynb
```

Results are saved to `experiments/results/random_forest.json`.
