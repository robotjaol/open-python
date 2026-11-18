# Model Robustness: Noise, Missing Values and Distribution Shift

How fast does accuracy fall when test inputs are degraded, and which models fall slowest?

- Notebook: [`model_robustness.ipynb`](model_robustness.ipynb)
- Dataset: `digits`
- Results: `experiments/results/model_robustness.json`

## Limits

Corruptions are synthetic and applied once with one seed. The results say nothing about natural shift (new writers, new scanners).
