# Fuel Efficiency Regression

Predict miles per gallon from engine and weight, with a model comparison and residual review. Substitutes the energy forecasting case for which no open data could be downloaded.

- Notebook: [`fuel_efficiency_regression.ipynb`](fuel_efficiency_regression.ipynb)
- Dataset: `auto_mpg`
- Results: `experiments/results/fuel_efficiency_regression.json`

## Problem

Predict a car's fuel economy from its design.

## Limits

Cars from 1970 to 1982 only, 392 rows, 79 test rows. Weight, displacement and horsepower are strongly correlated, so single coefficients are unstable.
