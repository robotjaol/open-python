# Glossary

- **Baseline:** a trivial model (mean, majority class, random policy) a real model must beat.
- **Bellman equation:** recursive relation between the value of a state and the values of the next states.
- **Calibration:** match between predicted probabilities and observed frequencies.
- **Cross validation:** repeated train and validation splits to average out luck.
- **Data leakage:** information from outside training rows used during training.
- **Discount factor (gamma):** weight on future reward, between 0 and 1.
- **Epsilon greedy:** act randomly with probability epsilon, otherwise take the best known action.
- **Hyperparameter:** a setting chosen before training, such as tree depth.
- **Off-policy / on-policy:** learning about a policy different from, or the same as, the one that collects data.
- **PR-AUC:** area under the precision recall curve.
- **Regularisation:** a penalty that discourages complex models.
- **Return:** discounted sum of rewards in an episode.
- **Seed:** number that fixes a random generator so a run can be repeated.
- **Stratified split:** split that keeps class proportions.
- **Target leakage:** a feature that contains the answer or is only known after it.
- **Transductive:** fitted on all rows including those later evaluated (as for t-SNE).
