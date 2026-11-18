# Evaluation metrics

| Task | Metric | Formula or meaning | Use when | Watch out |
|---|---|---|---|---|
| Regression | RMSE | square root of mean squared error | Large errors cost more | Same unit as target. Sensitive to outliers |
| Regression | MAE | mean absolute error | Robust summary | Does not punish large errors more |
| Regression | R2 | 1 minus residual variance over total variance | Compare with predicting the mean | Can be negative on test rows |
| Classification | Accuracy | share correct | Balanced classes | Misleading under imbalance |
| Classification | Precision, recall, F1 | TP over predicted positive, TP over actual positive, their harmonic mean | Positive class matters | Depends on threshold |
| Classification | ROC-AUC | probability a random positive scores above a random negative | Ranking quality | Hides poor precision when positives are rare |
| Classification | PR-AUC (average precision) | area under precision versus recall | Rare positives | Baseline equals positive share |
| Probabilities | Brier score, ECE | squared probability error, gap between predicted and observed rates | When you use the probability itself | ECE is noisy with few rows |
| Clustering | Silhouette | cohesion versus separation, from minus 1 to 1 | Internal check | Favours round clusters |
| Clustering | ARI, NMI | agreement with reference labels | Only when labels exist | Labels may not match structure |
| Anomaly | ROC-AUC, PR-AUC on a labelled test set | ranking of anomalies | Labels available | Constructed anomalies may not match real ones |
| RL | Mean return over evaluation episodes, per seed | average episode reward | Always with several seeds | Single seeds mislead (`henderson2018`) |

Implementations: `src/ml_lab/evaluation` and `src/ml_lab/metrics`.
