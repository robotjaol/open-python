# Changelog

Format follows Keep a Changelog. Versions follow semantic versioning once tagged.

## [0.1.0] - Unreleased

### Added
- 8 fundamentals, 20 supervised (8 regression, 12 classification), 10 unsupervised, 10 reinforcement learning, 6 advanced and 10 end-to-end project notebooks, all executed from a clean kernel with results stored in `experiments/results`.
- Shared library `ml_lab` (data, preprocessing, evaluation, RL, deployment), 15 unit and integration tests.
- Research register with verification status, dataset catalog, 13 documentation pages.
- Example deployment: model card with checksum, FastAPI service, batch scoring, drift check.
- `uv.lock` (resolved, not install-tested).
- GitHub Actions for tests, notebook validation, dependency audit and the preview app.

### Not included
- Projects on predictive maintenance, credit risk, fraud, energy forecasting and recommendation (data hosts unreachable when built). See `05_end_to_end_projects/BLOCKED.md`.
- RL extras: A2C, SAC, DDPG, TD3, offline RL. Time series notebooks. Docker deployment.
