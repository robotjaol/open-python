# Datasets

Catalog files: `catalog/supervised.json`, `catalog/unsupervised.json`, `catalog/reinforcement_learning.json`. Rules: [../docs/dataset-management.md](../docs/dataset-management.md).

| Dataset | Used for | Access | Licence (as recorded) |
|---|---|---|---|
| Auto MPG | regression, clustering, project | bundled in `mlxtend` | CC BY 4.0 |
| Breast Cancer Wisconsin (Diagnostic) | classification, anomaly, project | bundled in scikit-learn | CC BY 4.0 |
| Wine | classification, clustering | bundled in scikit-learn | CC BY 4.0 |
| Optical digits | classification, dimensionality reduction | bundled in scikit-learn | CC BY 4.0 |
| MNIST subset (5000 rows) | classification | bundled in `mlxtend` | see catalog |
| Diabetes progression | regression | bundled in scikit-learn | see catalog |
| Telco customer churn | classification, clustering, projects | downloaded, SHA-256 checked | terms not clearly stated, not redistributed |
| Gymnasium environments | reinforcement learning | `pip install gymnasium` | MIT |

`cache/` is ignored by git. Run `python scripts/download_datasets.py` to fill it.

**Check before use:** licences were read from source pages on 2026-11-17 where the catalog says so. Confirm them again before redistributing any data.
