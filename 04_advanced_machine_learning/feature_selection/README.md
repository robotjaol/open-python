# Feature Selection Without Leakage

Show how selecting features before cross validation inflates scores, and the correct nested version.

- Notebook: [`feature_selection.ipynb`](feature_selection.ipynb)
- Dataset: `breast_cancer`
- Results: `experiments/results/feature_selection.json`

## Limits

Synthetic noise is used by design to expose leakage and is not a real-world claim. Only one selector (univariate F) is tested.
