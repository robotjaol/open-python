# Research gaps in this repository

These are gaps in the repository, not claims about the field.

| Gap | Why it matters | How to close it |
|---|---|---|
| Required real world datasets (maintenance, credit, fraud, energy, recommendation) are missing | Projects cannot show domain specific issues such as time based leakage or extreme imbalance | Run `scripts/download_datasets.py` on a machine with access, add loaders, build the project notebooks |
| RL runs use 3 seeds and small budgets | `agarwal2021` and `henderson2018` show that few seeds give wide uncertainty | Use 10 or more seeds, report confidence intervals, longer training |
| Benchmark has four datasets | Cannot test claims from `grinsztajn2022` | Extend to 20 or more datasets with nested tuning |
| Anomaly detection data are constructed from a labelled class | Real faults differ in type and rate | Use an open industrial anomaly dataset |
| Unverified references (`ke2017`, `vandermaaten2008`, `sutton2018`, `hastie2009`) | Bibliographic details may contain errors | Check against the publisher page |
| Advanced ML and deployment folders hold small examples | They are not production patterns | Add authentication, load tests, monitoring in production |
