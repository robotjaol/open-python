# Extra Trees: Randomised Splits on Digit Images

Pick split thresholds at random to push variance down even further than a random forest.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`11_extra_trees.ipynb`](11_extra_trees.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How extremely randomised trees choose splits
- How the bias variance balance differs from random forests
- How to visualise importance on image pixels

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 11_extra_trees.ipynb
```

Results are saved to `experiments/results/extra_trees.json`.
