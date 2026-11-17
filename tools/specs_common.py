"""Shared text for specs."""

DS_MPG = ("**Auto MPG** (Quinlan, 1993), UCI Machine Learning Repository, licence CC BY 4.0, DOI 10.24432/C5859H. "
          "392 cars from 1970 to 1982 without missing horsepower (as shipped in `mlxtend`). Target `mpg`. Features: `cylinders`, "
          "`displacement`, `horsepower`, `weight`, `acceleration`, `model_year`, `origin` (1 US, 2 Europe, 3 Japan). All numeric. "
          "Limit: old cars only, so conclusions do not transfer to modern vehicles.")
DS_DIAB = ("**Diabetes progression** from Efron, Hastie, Johnstone and Tibshirani (2004), shipped with scikit-learn. 442 patients, "
           "10 baseline features (`age`, `sex`, `bmi`, `bp`, and six blood serum measures `s1` to `s6`), target is a quantitative measure of "
           "disease progression one year after baseline. The provider centred and scaled the features. Data terms of the original source are not "
           "stated by the provider. Limits: small sample, no time dimension, `sex` is a standardised number.")
DS_BC = ("**Breast Cancer Wisconsin (Diagnostic)**, UCI, licence CC BY 4.0, DOI 10.24432/C5DW2B. 569 rows, 30 numeric features "
         "(mean, standard error and worst value of ten nucleus properties computed from digitised fine needle aspirate images). "
         "Target `malignant` (1 malignant, 0 benign). No missing values. Limits: one institution, small sample.")
DS_WINE = ("**Wine**, UCI, licence CC BY 4.0, DOI 10.24432/C5PC7J. 178 rows, 13 chemical measurements (alcohol, malic acid, ash, "
           "magnesium, phenols, flavanoids, colour intensity, proline and others). Target `cultivar`, three classes. Limits: one region, tiny sample.")
DS_DIGITS = ("**Optical Recognition of Handwritten Digits**, UCI, licence CC BY 4.0, DOI 10.24432/C50P49. scikit-learn ships 1797 images "
             "(the test part of the UCI data) as 8 by 8 grids, so 64 integer pixel features from 0 to 16. Target `digit`, ten classes. "
             "Limit: far smaller than MNIST.")
DS_MNIST = ("**MNIST subset** with 5000 images of 28 by 28 pixels (784 features, values 0 to 255) from `mlxtend.data.mnist_data`. "
            "Target `digit`, ten classes. The licence is stated as CC BY-SA 3.0 on the Hugging Face card for MNIST. Verify before reuse.")
DS_TELCO = ("**Telco customer churn**, IBM sample data from the repository `IBM/telco-customer-churn-on-icp4d` (code licence Apache-2.0, "
            "dataset terms not stated). 7043 customers, 20 features: demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account "
            "(`tenure` in months, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`) and services "
            "(`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, "
            "`StreamingTV`, `StreamingMovies`). Target `Churn` (1 = customer left). `TotalCharges` has blank strings that become missing "
            "values. The file is downloaded once and checked against a SHA-256 hash. Limit: one snapshot, no time dimension. "
            "Duplicate rows are reported in the EDA but kept, because without an identifier we cannot tell whether they are different customers.")

REG_METRICS = ("Regression metrics: MAE (average absolute error in target units), RMSE (square root of the mean squared error, same unit, "
               "punishes large errors), R-squared (share of variance explained, can be negative on new data) and MAPE when the target is never zero.")
CLS_METRICS = ("Classification metrics: accuracy, precision, recall, F1, ROC-AUC and PR-AUC, plus the confusion matrix. "
               "With imbalanced classes accuracy alone misleads, so the model is chosen on F1 or PR-AUC as stated in the training cell.")
REG_INTERP = ("Permutation importance shuffles one feature and measures how much the error grows. Correlated features share importance, so read it as "
              "a ranking of groups. Check the residual plot for curvature or funnel shapes, which mean the model misses structure or the error "
              "variance changes with the prediction.")
CLS_INTERP = ("Permutation importance shows which inputs the model relies on. The confusion matrix shows which classes are mixed up, and "
              "the list of misclassified rows is the starting point for error analysis. A model is only as good as the cases it fails on.")
