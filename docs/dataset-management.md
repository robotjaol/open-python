# Dataset management

## Rules

1. Every dataset has an entry in `datasets/catalog/*.json` with source, licence, DOI or URL, row and feature count, and known limits.
2. Raw files are not committed unless they are small and the licence allows it. Downloads go to `datasets/cache/`, which git ignores.
3. Downloads are checked with SHA-256 when a checksum is known (`ml_lab.data.download`).
4. The licence is read from the source page, not assumed. If it cannot be read, write "not stated" and do not redistribute.

## Datasets used

See [../datasets/README.md](../datasets/README.md). Datasets bundled with scikit-learn and mlxtend are loaded from the installed package. Telco churn is downloaded from a public GitHub repository, and its terms are not stated clearly there, so it is not redistributed here.

## Adding a dataset

1. Add an entry to `src/ml_lab/data/registry.py` and a loader in `loaders.py`.
2. Run `python scripts/generate_catalog.py`.
3. Add a test that loads it and checks row count and target.
4. Document leakage risks (duplicates, time order, identifiers) in the notebook.
