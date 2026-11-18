# Wine Cultivar Authentication with Abstention

Classify wine cultivar from chemistry, find the smallest useful feature set, and abstain when the model is unsure.

- Notebook: [`wine_authentication.ipynb`](wine_authentication.ipynb)
- Dataset: `wine`
- Results: `experiments/results/wine_authentication.json`

## Problem

Identify the cultivar of a wine from chemical analysis, and decide when a human should re-check.

## Limits

178 samples from one region and three cultivars. The task is easy, so high scores say little about harder authentication problems. Confidence cut-offs come from cross validated probabilities on the same data.
