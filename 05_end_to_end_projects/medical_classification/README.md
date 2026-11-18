# Medical Classification: Breast Tumour Diagnosis Support

Choose a model by sensitivity at a fixed specificity target, with intervals and an error review.

- Notebook: [`medical_classification.ipynb`](medical_classification.ipynb)
- Dataset: `breast_cancer`
- Results: `experiments/results/medical_classification.json`

## Problem

Classify tumours as malignant or benign from nucleus measurements and report how the model errs.

## Limits

One institution, 569 rows, only 114 test rows. No external validation. The data are measurements from images, not images. The result must not be used for patient care.
