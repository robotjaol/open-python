# Customer Churn Prediction and Retention Targeting

From a ranked churn model to a cost based contact list, with calibration and an honest baseline.

- Notebook: [`customer_churn.ipynb`](customer_churn.ipynb)
- Dataset: `telco_churn`
- Results: `experiments/results/customer_churn.json`

## Problem

Predict which telecom customers will leave and decide whom to contact with a retention offer.

## Limits

One snapshot with no time dimension, so we cannot test drift. The cost numbers are assumptions. Duplicates were kept. The data has no information about what offers work, so the model ranks risk but does not say which action to take (that needs an experiment with a control group).
