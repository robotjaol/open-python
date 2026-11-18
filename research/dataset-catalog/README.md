# Dataset catalog

Machine readable entries live in `datasets/catalog/*.json`. This page explains what is and is not in the repository.

## Substitutions

The build environment could not reach UCI, OpenML, Kaggle, Hugging Face or DOI hosts. Datasets that ship inside installed Python packages (scikit-learn, mlxtend) were used, and the Telco churn file was downloaded from a GitHub raw URL and checked with SHA-256. Their original sources and licences are recorded in each catalog entry.

## Not available

The directive lists predictive maintenance, credit risk, fraud detection, energy forecasting and recommendation projects. Their usual open datasets sit on blocked hosts, so those projects are **Blocked**, not simulated. See `05_end_to_end_projects/BLOCKED.md`.
