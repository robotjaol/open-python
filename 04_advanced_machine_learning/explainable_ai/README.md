# Explainable Models: Permutation Importance, Partial Dependence and Their Limits

Explain a gradient boosting model on fuel efficiency and test whether the explanation is stable.

- Notebook: [`explainable_ai.ipynb`](explainable_ai.ipynb)
- Dataset: `auto_mpg`
- Results: `experiments/results/explainable_ai.json`

## Limits

Permutation importance treats correlated features as independent, which can create unrealistic rows. Partial dependence averages over such rows. Both describe the model, not causal effects.
