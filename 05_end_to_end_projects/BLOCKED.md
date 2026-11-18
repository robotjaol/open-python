# Projects that are Blocked

The directive lists projects whose usual open data could not be obtained in the build environment. The environment reached package indexes and GitHub only. UCI, OpenML, Kaggle, Hugging Face and DOI hosts were refused by the network allowlist. **No synthetic stand-in was created for these**, because the directive forbids synthetic data for required real-world cases.

| Directive project | Status | Reason | What to do |
|---|---|---|---|
| Predictive maintenance | Blocked | Candidate public datasets (for example NASA C-MAPSS, AI4I 2020) are on blocked hosts | Download on a machine with access, add a loader and catalog entry, then write the 13 part notebook |
| Credit risk | Blocked | Usual datasets are on Kaggle or UCI | Same |
| Fraud detection | Blocked | Usual dataset is on Kaggle or OpenML | Same. Needs time ordered splits and precision at low false positive rate |
| Energy consumption forecasting | Blocked | Usual datasets are on UCI or Kaggle | Same. Needs a time series split |
| Recommendation system | Blocked | MovieLens and similar are on GroupLens or Kaggle | Same |

Dataset names above are examples of where to look. Their licences and availability were **not verified** here.

## Substitutions that are in the repository

| Directive project | Substitute | Honest difference |
|---|---|---|
| Customer churn | `customer_churn` (Telco churn, real data) | Direct match |
| Customer segmentation | `customer_segmentation` (Telco, real data) | Direct match |
| Medical classification | `medical_classification` (Breast Cancer Wisconsin, real data) | Direct match |
| Autonomous navigation with RL | `autonomous_control` (CartPole) | Control, not navigation |
| Industrial anomaly detection | `anomaly_screening` (constructed anomalies from a real labelled class) | Not industrial data |
| Energy forecasting | `fuel_efficiency_regression` (Auto MPG regression) | Regression, not forecasting. Not a replacement |
