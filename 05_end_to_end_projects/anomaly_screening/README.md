# Anomaly Screening: Comparing Detectors on Rare Positives

Compare three anomaly detectors over many random draws of the rare class, instead of trusting one split. A proxy for industrial anomaly detection, for which no open data could be downloaded.

- Notebook: [`anomaly_screening.ipynb`](anomaly_screening.ipynb)
- Dataset: `breast_cancer`
- Results: `experiments/results/anomaly_screening.json`

## Problem

Rank rows by how unusual they are when only a small share of rows are abnormal.

## Limits

Constructed anomalies from a real class, one dataset, 30 repetitions of a 30 percent split. Real industrial anomalies differ in kind and frequency.
