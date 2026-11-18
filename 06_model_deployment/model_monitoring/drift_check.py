"""Compare a new CSV with the training data. Usage: python drift_check.py new.csv"""

import sys

import pandas as pd

from ml_lab.data import load_dataset
from ml_lab.deployment import drift_report

train = load_dataset("breast_cancer").X
new = pd.read_csv(sys.argv[1])[train.columns]
print(drift_report(train, new).head(10).to_string(index=False))
