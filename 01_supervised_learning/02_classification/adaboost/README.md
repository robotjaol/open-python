# AdaBoost: Boosting Stumps on Breast Cancer Data

Reweight the hard examples after each round and combine weak learners.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `breast_cancer` (see `datasets/catalog`) |
| Notebook | [`10_adaboost.ipynb`](10_adaboost.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How example weights shift towards mistakes
- How the weak learner votes are weighted
- Why AdaBoost is sensitive to label noise

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 10_adaboost.ipynb
```

Results are saved to `experiments/results/adaboost.json`.
