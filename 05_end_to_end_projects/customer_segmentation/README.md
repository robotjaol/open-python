# Customer Segmentation with Clustering

Segment telecom customers on usage and tenure, check stability, and read the segments with churn as an outside check.

- Notebook: [`customer_segmentation.ipynb`](customer_segmentation.ipynb)
- Dataset: `telco_churn`
- Results: `experiments/results/customer_segmentation.json`

## Problem

Group customers into segments that a marketing team can act on, using account and service information only.

## Limits

Four numeric features only, so segments mostly separate by tenure and price. Segment churn rates are an association, not a causal effect. Silhouette values are low for this kind of data, which means segments overlap.
