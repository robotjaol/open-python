"""Download datasets that are not bundled with Python packages."""

import sys

from ml_lab.data import load_dataset


def main(names):
    names = names or ["telco_churn"]
    for name in names:
        bundle = load_dataset(name)
        print(f"{name}: {bundle.X.shape[0]} rows, {bundle.X.shape[1]} features, target {bundle.target_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
