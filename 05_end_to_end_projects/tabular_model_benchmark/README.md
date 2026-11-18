# Tabular Model Benchmark Across Four Datasets

Compare six model families on four datasets with repeated cross validation and report average ranks, not a single winner.

- Notebook: [`tabular_model_benchmark.ipynb`](tabular_model_benchmark.ipynb)
- Dataset: `multiple`
- Results: `experiments/results/tabular_model_benchmark.json`

## Problem

Which model families work best across several small tabular problems?

## Limits

Four small datasets, fixed light settings, one metric, Telco shortened to 4000 rows. Differences of a few thousandths are noise. This benchmark cannot confirm or refute large scale benchmark claims.
