# Handwritten Digit Recognition with Error Analysis

Compare five classifiers, study the errors, and test robustness to noise.

- Notebook: [`digit_recognition.ipynb`](digit_recognition.ipynb)
- Dataset: `digits`
- Results: `experiments/results/digit_recognition.json`

## Problem

Recognise handwritten digits from 8 by 8 pixel images and find out which digits get confused and how robust the model is.

## Limits

8 by 8 images from one collection, 1797 rows. High accuracy here does not transfer to different writers, scanners or larger images.
