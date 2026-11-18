# open-python

A paper-driven Python laboratory for learning and researching machine learning. Every notebook states its problem, cites the paper behind the algorithm, runs from a clean kernel, and writes its numbers to a results file that the preview app reads.

[![python-tests](https://github.com/robotjaol/open-python/actions/workflows/python-tests.yml/badge.svg)](https://github.com/robotjaol/open-python/actions/workflows/python-tests.yml)
[![notebook-validation](https://github.com/robotjaol/open-python/actions/workflows/notebook-validation.yml/badge.svg)](https://github.com/robotjaol/open-python/actions/workflows/notebook-validation.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

**Preview app:** after you enable GitHub Pages (Settings, Pages, Source: GitHub Actions) the `preview-app` workflow publishes it at `https://robotjaol.github.io/open-python/`. It lists every notebook with saved metrics, figures, and the executed notebook. To build it locally: `make app`, then serve `app/dist`.

## What is inside

| Folder | Content | Notebooks |
|---|---|---|
| `00_fundamentals` | Python, NumPy and pandas, plotting, linear algebra, probability, optimisation, ML concepts, feature engineering | 8 |
| `01_supervised_learning` | 8 regression and 12 classification algorithms on real datasets | 20 |
| `02_unsupervised_learning` | Clustering, dimensionality reduction, anomaly detection | 10 |
| `03_reinforcement_learning` | Dynamic programming, Monte Carlo, TD, SARSA, Q-learning, DQN, Double DQN, REINFORCE, actor-critic, PPO | 10 |
| `04_advanced_machine_learning` | Imbalance, calibration, leakage-free feature selection, explainability, nested tuning, robustness | 6 |
| `05_end_to_end_projects` | Ten projects with 13 parts each, from problem statement to recommendation | 10 |
| `06_model_deployment` | Model card with checksum, FastAPI service, batch scoring, drift check | scripts |
| `src/ml_lab` | Shared library: data, preprocessing, evaluation, RL, deployment | |
| `research`, `datasets`, `docs` | Source register, comparisons, dataset catalog, 13 documentation pages | |

## Quick start

```bash
git clone https://github.com/robotjaol/open-python.git && cd open-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,development]"
make test
jupyter lab 00_fundamentals/01_python_for_ml/01_python_for_ml.ipynb
```

Details: [docs/getting-started.md](docs/getting-started.md). Study order: [docs/learning-roadmap.md](docs/learning-roadmap.md).

## How the notebooks are built

1. Each supervised notebook follows a 15 section template (problem, why this algorithm, paper, mathematics, data, preprocessing, from-scratch version, library version, training, evaluation, visualisation, interpretation, experiments, exercises, limitations).
2. Splits are 60/20/20. Preprocessing is fitted on training rows only. One setting is chosen on validation rows, the model is refitted on training plus validation rows, and the test rows are scored once, next to a dummy baseline and 5-fold cross validation.
3. Reinforcement learning notebooks use evaluation seeds that training never saw, several training seeds, a random-policy floor, and exact dynamic programming references where they exist.
4. Notebooks are executed by `tools/run_notebooks.py` from a clean kernel. `tests/notebook_validation` checks that every notebook is valid, executed, error free, and free of local paths.

## Examples of saved results (single runs, see the notebooks for limits)

| Project | Result | Baseline |
|---|---|---|
| Customer churn (Telco, 7043 rows) | PR-AUC 0.661 on test rows | 0.265 (predicting the churn share) |
| Breast tumour diagnosis (569 rows) | ROC-AUC 0.995. Sensitivity 0.857 at a threshold meant for 0.95, 95% interval 0.735 to 0.958 | |
| CartPole control, 3 seeds | PPO 496.6 mean return, hand written rule 500, random 21.0 | 500 is the maximum |
| Anomaly screening, 30 draws | Isolation Forest ROC-AUC 0.957 (std 0.037) | Random PR-AUC 0.048 |

The medical row is shown because it fails its own target on test rows. That is a useful result, not a hidden one.

## Honest status

Not everything in the original plan exists. The full table is in the preview app and in [CHANGELOG.md](CHANGELOG.md).

- **Blocked:** projects on predictive maintenance, credit risk, fraud detection, energy forecasting and recommendation. Their usual open datasets were on hosts the build environment could not reach. Nothing was simulated in their place. See [05_end_to_end_projects/BLOCKED.md](05_end_to_end_projects/BLOCKED.md).
- **Not implemented:** A2C, SAC, DDPG, TD3, offline RL, time series notebooks, Docker deployment.
- **Not validated:** the GitHub Actions workflows were written and parsed but have not run on GitHub yet.
- **Partly verified:** most references were checked against Crossref or arXiv. Four are marked "not verified" in [research/source-register.md](research/source-register.md).
- **Lock file:** `uv.lock` was resolved with `uv lock` (230 packages) but nothing was installed from it and CI does not use it yet.
- **Small scale:** deep RL uses 3 seeds and a short CPU budget. Datasets are small. Read each notebook's limitations.

## Datasets

Bundled datasets come from installed packages. Telco churn is downloaded and checked with SHA-256. Licences and limits are in [datasets/README.md](datasets/README.md). You are responsible for following each dataset's terms.

## Contributing, security, citation

[CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Cite with [CITATION.cff](CITATION.cff). Licensed under [MIT](LICENSE).
