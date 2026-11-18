# Model Calibration: Do Predicted Probabilities Match Observed Rates?

Compare raw, sigmoid and isotonic calibrated probabilities with the Brier score and reliability curves.

- Notebook: [`model_calibration.ipynb`](model_calibration.ipynb)
- Dataset: `telco_churn`
- Results: `experiments/results/model_calibration.json`

## Limits

Single split. ECE with 10 bins on about 1400 test rows is noisy. Isotonic regression can overfit with few rows.
