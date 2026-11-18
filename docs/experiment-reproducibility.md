# Experiment reproducibility

**What is fixed.** Seed 42 for splits and models. Splits are 60/20/20 stratified for classification. Package versions are recorded in every result file (`environment` key).

**What is stored.** Each notebook writes `experiments/results/<id>.json` with config, metrics, seed and environment, and figures to `experiments/figures/<id>/`.

**How to reproduce one result.**
```bash
python tools/run_notebooks.py 01_supervised_learning/02_classification/logistic_regression
```
Then compare the new JSON with the committed one. Small differences (third decimal) can appear across scikit-learn or NumPy versions and across CPUs for multi-threaded methods.

**Limits.** Deep RL results depend on library versions and threads. RL notebooks use single thread CPU torch and three seeds. `uv.lock` records one resolved dependency tree (`uv sync --frozen` can install it). It was resolved but not install-tested, and CI currently installs with pip.

**Rule against leakage.** Test rows are scored once, after model choice on validation rows.
