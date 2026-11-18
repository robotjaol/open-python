# Interpreting a Small Medical Regression: Diabetes Progression

Which baseline measurements relate to progression, how stable is that answer, and what does a sparse model keep?

- Notebook: [`diabetes_interpretation.ipynb`](diabetes_interpretation.ipynb)
- Dataset: `diabetes`
- Results: `experiments/results/diabetes_interpretation.json`

## Problem

Predict disease progression one year ahead and identify measurements that matter, with a stability check.

## Limits

442 patients, one study, features pre-scaled by the provider, and `sex` is a standardised number. Associations, not causes. Rankings of correlated serum measures are unstable, as the bootstrap table shows.
