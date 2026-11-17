# Decision Tree: Readable Rules for Customer Churn

Grow a tree of if-then rules, then prune it so it generalises.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `telco_churn` (see `datasets/catalog`) |
| Notebook | [`03_decision_tree.ipynb`](03_decision_tree.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How a tree chooses splits with Gini impurity
- Why unpruned trees overfit
- How to read the rules of a small tree
- How to prune with depth limits and cost complexity

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 03_decision_tree.ipynb
```

Results are saved to `experiments/results/decision_tree.json`.
