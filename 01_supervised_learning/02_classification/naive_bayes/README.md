# Naive Bayes: Handwritten Digits from Pixel Statistics

Classify with Bayes rule and an independence assumption, and learn why it still works.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `mnist_5000` (see `datasets/catalog`) |
| Notebook | [`06_naive_bayes.ipynb`](06_naive_bayes.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How Bayes rule gives a classifier
- What the naive independence assumption means
- Why Laplace smoothing is needed
- Why naive Bayes can classify well with poor probabilities

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 06_naive_bayes.ipynb
```

Results are saved to `experiments/results/naive_bayes.json`.
