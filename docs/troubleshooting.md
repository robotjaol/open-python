# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: ml_lab` | Package not installed | `pip install -e .` in the repo root |
| `ValueError: underlying array is read-only` | pandas 3 copy-on-write | Use `.to_numpy(copy=True)` before changing an array in place |
| Telco notebooks fail on download | No internet or GitHub blocked | Download the CSV by hand to `datasets/cache/telco_churn.csv`, then run again. The loader checks the checksum |
| `ImportError` for `umap`, `xgboost`, `lightgbm`, `catboost`, `gymnasium` or `torch` | Optional extras not installed | `pip install -e ".[all]"` |
| RL notebook takes more than 10 minutes | Many threads fighting each other | `export OMP_NUM_THREADS=1` |
| Results differ in the third decimal | Library or CPU differences | Expected, see [experiment-reproducibility.md](experiment-reproducibility.md) |
| `fastapi` import fails | Deployment extra missing | `pip install -e ".[deployment]"` |
| Notebook shows outputs from an old run | Stale outputs | Restart kernel, Run All, save |
