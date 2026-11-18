# Getting started

Goal: run your first notebook in about 10 minutes.

**Requirements.** Python 3.10 or newer (tested on 3.13), 4 GB RAM, about 2 GB disk. No GPU is needed. Internet is needed once to install packages and, for Telco churn, to download one CSV.

## Steps

1. Clone and enter the repository.
   ```bash
   git clone https://github.com/robotjaol/open-python.git
   cd open-python
   ```
2. Create an environment and install. Expected result: no error and `ml_lab` importable.
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[all,development]"
   python -c "import ml_lab; print('ok')"
   ```
3. Check the environment.
   ```bash
   python scripts/setup_environment.py
   ```
4. Open the first notebook.
   ```bash
   jupyter lab 00_fundamentals/01_python_for_ml/01_python_for_ml.ipynb
   ```
5. Run all cells (Kernel, Restart and Run All). Expected result: no red error cell.
6. Optional: download the Telco file used by churn notebooks.
   ```bash
   python scripts/download_datasets.py
   ```
7. Run tests. Expected result: all pass.
   ```bash
   make test
   ```

## Where to go next

Follow [learning-roadmap.md](learning-roadmap.md). Reasoning behind the layout: [experiment-reproducibility.md](experiment-reproducibility.md).

If something fails, see [troubleshooting.md](troubleshooting.md).
