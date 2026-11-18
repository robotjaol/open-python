# Literature review (scope and status)

This is a reading map, not a systematic review. No search protocol, database query log or inclusion criteria were run, so do not cite it as a systematic review.

## Themes covered by verified sources

- **Tabular supervised learning.** Tree ensembles and boosting (`breiman2001`, `friedman2001`, `chen2016`, `prokhorenkova2018`), kernel methods (`cortes1995`), penalised linear models (`hoerl1970`, `tibshirani1996`, `zou2005`, `efron2004`). `grinsztajn2022` reports that tree based models stayed ahead of deep networks on a benchmark of medium sized tabular datasets. The repository benchmark has four small datasets, so it neither confirms nor refutes that result.
- **Unsupervised learning.** Partitioning, density and model based clustering (`lloyd1982`, `schubert2017`, `campello2013`, `dempster1977`, `vonluxburg2007`), projection (`pearson1901`, `mcinnes2018`) and anomaly detection (`liu2008`, `breunig2000`).
- **Reinforcement learning.** Temporal difference and Q-learning (`sutton1988`, `watkins1992`), deep value methods (`mnih2015`, `vanhasselt2016`), policy gradient (`williams1992`, `konda2003`, `schulman2017`), and reporting practice (`henderson2018`, `agarwal2021`).

## Not covered

Soft actor-critic, DDPG, TD3 and offline RL have verified sources in the register but **no notebooks**. Time series, causal inference, fairness and large language model topics are not covered.
