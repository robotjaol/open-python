# Logistic Regression: Diagnosing Breast Tumours

Model the probability of malignancy and learn how regularisation and thresholds change decisions.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Beginner |
| Dataset | `breast_cancer` (see `datasets/catalog`) |
| Notebook | [`01_logistic_regression.ipynb`](01_logistic_regression.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How the sigmoid turns a linear score into a probability
- What log loss and maximum likelihood mean
- How the decision threshold changes precision and recall
- How regularisation strength C acts

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 01_logistic_regression.ipynb
```

Results are saved to `experiments/results/logistic_regression.json`.
