"""Build advanced machine learning notebooks (compact, 9 sections)."""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "04_advanced_machine_learning"

PRE = '''import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from ml_lab.data import load_dataset
from ml_lab.preprocessing import split_three_way, make_model_pipeline
from ml_lab.experiments import ExperimentRecord, save_figure
from ml_lab.utils import set_seed, print_environment
from ml_lab import visualization as viz
NB_ID = "__ID__"
SEED = 42
set_seed(SEED); viz.apply_style(); print_environment()'''

SAVE = '''record = ExperimentRecord(notebook=NB_ID, family="advanced", algorithm=ALGORITHM, dataset="__DS__", task="__TASK__", seed=SEED, config=CONFIG, metrics=METRICS, extra=EXTRA)
print("saved:", record.save())'''

NB = []


def nb(**kw):
    NB.append(kw)


def render(p):
    folder = BASE / p["id"]
    folder.mkdir(parents=True, exist_ok=True)
    s = lambda t: t.replace("__ID__", p["id"]).replace("__DS__", p["dataset"]).replace("__TASK__", p["task"])  # noqa: E731
    cells = [new_markdown_cell(f"# {p['title']}\n\n**Dataset:** `{p['dataset']}` | **Level:** advanced\n\n{p['tagline']}\n\n"
                               "Sections: 1 Problem, 2 Method, 3 Setup and data, 4 Experiment, 5 Results, 6 Interpretation, 7 Limitations, 8 Exercises, 9 References."),
             new_markdown_cell("## 1. Problem\n\n" + p["problem"]),
             new_markdown_cell("## 2. Method\n\n" + p["method"]),
             new_markdown_cell("## 3. Setup and data\n\n" + p["data_md"]),
             new_code_cell(s(PRE)), new_code_cell(p["load"]),
             new_markdown_cell("## 4. Experiment\n\n" + p["exp_md"]), new_code_cell(p["exp"]),
             new_markdown_cell("## 5. Results\n\n" + p["res_md"]), new_code_cell(p["res"] + "\n\n" + s(SAVE)),
             new_markdown_cell("## 6. Interpretation\n\n" + p["interp"]),
             new_markdown_cell("## 7. Limitations\n\n" + p["limits"]),
             new_markdown_cell("## 8. Exercises\n\n" + "\n".join(f"{i}. {e}" for i, e in enumerate(p["exercises"], 1))),
             new_markdown_cell("## 9. References\n\n" + p["refs"])]
    nbk = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.validate(nbk)
    path = folder / f"{p['id']}.ipynb"
    nbformat.write(nbk, path)
    (folder / "README.md").write_text(f"# {p['title']}\n\n{p['tagline']}\n\n- Notebook: [`{path.name}`]({path.name})\n- Dataset: `{p['dataset']}`\n- Results: `experiments/results/{p['id']}.json`\n\n## Limits\n\n{p['limits']}\n")
    (folder / "references.md").write_text(f"# References: {p['title']}\n\n{p['refs']}\n\nVerification status of every key is in `research/source-register.md`.\n")
    print("built", path.relative_to(ROOT))


TELCO = "Telco customer churn (7043 rows, 21 columns, about 26.5 percent churn). Source and checksum are documented in `datasets/catalog/supervised.json` and `01_supervised_learning/` notebooks."
LOAD_TELCO = '''b = load_dataset("telco_churn"); X, y = b.X, b.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="classification", seed=SEED)
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, brier_score_loss
print(len(Xtr), len(Xva), len(Xte), "| churn share", round(float(y.mean()), 3))'''

nb(id="imbalanced_learning", title="Imbalanced Learning: Class Weights, Resampling by Weight and Thresholds", dataset="telco_churn", task="classification",
   tagline="Does reweighting help more than moving the decision threshold?",
   problem="Churn is the minority class (about 26.5 percent). Accuracy at threshold 0.5 hides poor recall. We test three ways to deal with imbalance and compare them on ranking and on F1.",
   method="(a) plain model, (b) class weights (`class_weight=balanced`), (c) plain model with a threshold chosen on validation rows to maximise F1. The ranking metric PR-AUC is not changed by a threshold, so any PR-AUC difference comes from the model itself. No synthetic oversampling is used because it must be done inside each cross validation fold and adds a dependency.",
   data_md=TELCO, load=LOAD_TELCO,
   exp_md="Fit three logistic regression variants on training rows and a random forest with and without class weights.",
   exp='''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
variants = {"logistic plain": LogisticRegression(max_iter=2000), "logistic balanced": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "forest plain": RandomForestClassifier(300, min_samples_leaf=5, n_jobs=2, random_state=SEED),
            "forest balanced": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=2, random_state=SEED)}
fit = {n: make_model_pipeline(Xtr, m, scale="logistic" in n).fit(Xtr, ytr) for n, m in variants.items()}
def best_threshold(p, y_true):
    grid = np.linspace(0.05, 0.9, 86); f = [f1_score(y_true, p >= t) for t in grid]; return float(grid[int(np.argmax(f))])
print("fitted", list(fit))''',
   res_md="Test rows, scored once. F1 is reported at threshold 0.5 and at the threshold tuned on validation rows.",
   res='''rows = []
for n, m in fit.items():
    pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    t = best_threshold(pv, yva)
    rows.append({"model": n, "PR_AUC": average_precision_score(yte, pt), "F1_at_0.5": f1_score(yte, pt >= 0.5), "tuned_threshold": t, "F1_at_tuned": f1_score(yte, pt >= t), "recall_at_0.5": ((pt >= 0.5) & (yte == 1)).sum() / (yte == 1).sum()})
tab = pd.DataFrame(rows).set_index("model"); display(tab.round(3))
METRICS = {f"{n.replace(' ', '_')}_{k}": float(v) for n, r in tab.iterrows() for k, v in r.items()}
CONFIG = {"threshold_grid": "0.05 to 0.9 on validation"}; EXTRA = {"table": tab.round(4).reset_index().to_dict("records")}; ALGORITHM = "class weights and threshold tuning"''',
   interp="Read the table by column. If PR-AUC is nearly equal across plain and balanced variants while F1 at 0.5 differs, weighting acts mostly like a threshold shift. The tuned threshold column shows how much of the gain a simple threshold gives without changing the model.",
   limits="One dataset, one split, one seed. The differences are single run values with no interval. No resampling methods (SMOTE and similar) were tested.",
   exercises=["Repeat with five random splits and report the spread of F1.", "Add a cost matrix (missed churn costs five times a wasted offer) and choose the threshold by expected cost.", "Try SMOTE inside a cross validation pipeline and compare."],
   refs="He, H. and Garcia, E. (2009), *Learning from imbalanced data*, IEEE TKDE, is the usual survey. It is **not verified in the build environment** and is listed for reading only. For model references see `research/source-register.md` (`breiman2001`).")

nb(id="model_calibration", title="Model Calibration: Do Predicted Probabilities Match Observed Rates?", dataset="telco_churn", task="classification",
   tagline="Compare raw, sigmoid and isotonic calibrated probabilities with the Brier score and reliability curves.",
   problem="A churn score is used to estimate how many customers will leave and to apply a cost threshold. That needs probabilities that match observed frequencies. Tree ensembles are often miscalibrated.",
   method="Fit a random forest and a histogram gradient boosting model. Calibrate each with sigmoid and isotonic regression using 5-fold cross fitting on the training rows. Compare Brier score (mean squared error of the probability, lower is better) and expected calibration error (ECE) with 10 equal count bins on the test rows.",
   data_md=TELCO, load=LOAD_TELCO,
   exp_md="Fit and calibrate.",
   exp='''from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from ml_lab.preprocessing import build_preprocessor
base = {"forest": RandomForestClassifier(300, min_samples_leaf=3, n_jobs=2, random_state=SEED), "hist_gb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)}
models = {}
for n, m in base.items():
    pre = build_preprocessor(Xtr, scale=False)
    models[(n, "raw")] = Pipeline([("prep", pre), ("model", clone(m))]).fit(Xtr, ytr)
    for method in ["sigmoid", "isotonic"]:
        models[(n, method)] = Pipeline([("prep", build_preprocessor(Xtr, scale=False)), ("model", CalibratedClassifierCV(clone(m), method=method, cv=5))]).fit(Xtr, ytr)
def ece(y_true, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[0], q[-1] = -1, 2
    idx = np.digitize(p, q[1:-1]); return float(sum(abs(y_true[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))
print(len(models), "models")''',
   res_md="Test rows.",
   res='''rows = []
for (n, method), m in models.items():
    p = m.predict_proba(Xte)[:, 1]
    rows.append({"model": n, "calibration": method, "Brier": brier_score_loss(yte, p), "ECE": ece(yte.values, p), "ROC_AUC": roc_auc_score(yte, p)})
tab = pd.DataFrame(rows); display(tab.round(4))
fig, ax = plt.subplots(figsize=(5, 4.5)); ax.plot([0, 1], [0, 1], "--", color="gray")
for (n, method), m in models.items():
    if n == "forest":
        f, mp = calibration_curve(yte, m.predict_proba(Xte)[:, 1], n_bins=8, strategy="quantile"); ax.plot(mp, f, marker="o", label=method)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed share"); ax.legend(); ax.set_title("Random forest reliability"); save_figure(fig, NB_ID, "reliability"); plt.show()
METRICS = {f"{r.model}_{r.calibration}_{k}": float(getattr(r, k)) for r in tab.itertuples() for k in ["Brier", "ECE", "ROC_AUC"]}
CONFIG = {"cv": 5, "bins": 10, "methods": ["raw", "sigmoid", "isotonic"]}; EXTRA = {"table": tab.round(4).to_dict("records")}; ALGORITHM = "CalibratedClassifierCV"''',
   interp="Compare Brier and ECE within each model. Calibration should leave ROC-AUC almost unchanged because it is a monotone transformation (isotonic can create ties). A lower Brier with similar ROC-AUC indicates better probabilities, not a better ranking.",
   limits="Single split. ECE with 10 bins on about 1400 test rows is noisy. Isotonic regression can overfit with few rows.",
   exercises=["Bootstrap the test rows to put an interval on ECE.", "Calibrate on a separate validation set instead of cross fitting.", "Check calibration inside each contract type."],
   refs="Calibration methods are implemented in scikit-learn (`CalibratedClassifierCV`). The usual paper references (Platt 1999, Zadrozny and Elkan 2002, Guo et al. 2017) are **not verified in the build environment**; check them before citing.")

nb(id="feature_selection", title="Feature Selection Without Leakage", dataset="breast_cancer", task="classification",
   tagline="Show how selecting features before cross validation inflates scores, and the correct nested version.",
   problem="Selecting features on all rows and then cross validating leaks label information into every fold. With many features and few rows the bias can be large.",
   method="Compare (a) the wrong way: select the top k features by ANOVA F on the full data, then cross validate, and (b) the right way: select inside a pipeline fitted on each training fold. To make the effect visible we add 300 random noise columns to the real data and repeat 20 times. The noise columns are synthetic and are used only for this leakage demonstration.",
   data_md="Breast Cancer Wisconsin Diagnostic, 569 rows, 30 features, plus 300 synthetic noise columns for the leakage demonstration. The model comparison at the end uses only the real 30 features.",
   load='''b = load_dataset("breast_cancer"); X, y = b.X, b.y
print(X.shape)''',
   exp_md="Sample 100 rows, add noise columns, and score with leaky and correct selection.",
   exp='''from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
leaky, correct = [], []
for rep in range(20):
    rng = np.random.default_rng(rep)
    rows = rng.choice(len(X), 100, replace=False)
    Xs = np.hstack([X.values[rows], rng.normal(size=(100, 300))]); ys = y.values[rows]
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
    Xsel = SelectKBest(f_classif, k=10).fit_transform(Xs, ys)
    leaky.append(cross_val_score(LogisticRegression(max_iter=2000), StandardScaler().fit_transform(Xsel), ys, cv=cv).mean())
    correct.append(cross_val_score(make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10), LogisticRegression(max_iter=2000)), Xs, ys, cv=cv).mean())
print("done")''',
   res_md="Compare accuracy of both procedures. Then, on pure noise labels, the leaky procedure should look better than chance and the correct one should not.",
   res='''noise_leaky, noise_correct = [], []
for rep in range(20):
    rng = np.random.default_rng(100 + rep)
    Xn, yn = rng.normal(size=(100, 300)), rng.integers(0, 2, 100)
    cv = StratifiedKFold(5, shuffle=True, random_state=rep)
    noise_leaky.append(cross_val_score(LogisticRegression(max_iter=2000), StandardScaler().fit_transform(SelectKBest(f_classif, k=10).fit_transform(Xn, yn)), yn, cv=cv).mean())
    noise_correct.append(cross_val_score(make_pipeline(StandardScaler(), SelectKBest(f_classif, k=10), LogisticRegression(max_iter=2000)), Xn, yn, cv=cv).mean())
tab = pd.DataFrame({"real data + noise columns": [np.mean(leaky), np.mean(correct)], "pure noise labels (true accuracy about 0.5)": [np.mean(noise_leaky), np.mean(noise_correct)]}, index=["leaky selection", "selection inside pipeline"])
display(tab.round(3))
METRICS = {"real_leaky": float(np.mean(leaky)), "real_correct": float(np.mean(correct)), "noise_leaky": float(np.mean(noise_leaky)), "noise_correct": float(np.mean(noise_correct)), "noise_leaky_std": float(np.std(noise_leaky, ddof=1)), "noise_correct_std": float(np.std(noise_correct, ddof=1))}
CONFIG = {"k": 10, "noise_columns": 300, "rows": 100, "repetitions": 20}; EXTRA = {"table": tab.round(4).reset_index().to_dict("records")}; ALGORITHM = "SelectKBest in pipeline"''',
   interp="On pure noise the true accuracy is 0.5. A leaky score above 0.5 is optimism created by the procedure, not signal. The in-pipeline score stays near 0.5 as it should.",
   limits="Synthetic noise is used by design to expose leakage and is not a real-world claim. Only one selector (univariate F) is tested.",
   exercises=["Replace SelectKBest with L1 logistic regression selection inside the pipeline.", "Vary k and the number of noise columns and plot the leakage bias.", "Do the same for scaling fitted on all rows."],
   refs="Ambroise and McLachlan (2002), *Selection bias in gene extraction on the basis of microarray gene-expression data*, PNAS, is the standard reference for this effect. It is **not verified in the build environment**.")

nb(id="explainable_ai", title="Explainable Models: Permutation Importance, Partial Dependence and Their Limits", dataset="auto_mpg", task="regression",
   tagline="Explain a gradient boosting model on fuel efficiency and test whether the explanation is stable.",
   problem="A tree ensemble predicts mpg well but has no coefficients. We need global explanations and a check that they do not change with the random seed or the data sample.",
   method="Permutation importance on held out rows (drop in R2 when one column is shuffled), partial dependence for the top feature, and a stability check over 20 bootstrap fits. No SHAP library is used because it was not available in the build environment.",
   data_md="Auto MPG, 392 rows, 7 features (see `datasets/catalog/supervised.json`).",
   load='''b = load_dataset("auto_mpg"); X, y = b.X, b.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="regression", seed=SEED)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import r2_score
Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
model = make_model_pipeline(Xtv, HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED), scale=False).fit(Xtv, ytv)
print("test R2:", round(r2_score(yte, model.predict(Xte)), 3))''',
   exp_md="Permutation importance on the test rows, 20 repeats.",
   exp='''imp = permutation_importance(model, Xte, yte, scoring="r2", n_repeats=20, random_state=SEED)
tab = pd.DataFrame({"importance": imp.importances_mean, "std": imp.importances_std}, index=X.columns).sort_values("importance", ascending=False)
display(tab.round(3))''',
   res_md="Partial dependence of the top feature, and rank stability over 20 bootstrap refits of the whole model.",
   res='''top = tab.index[0]
pdp = partial_dependence(model, Xte, [top], grid_resolution=20)
fig, ax = plt.subplots(figsize=(5, 3.5)); ax.plot(pdp["grid_values"][0], pdp["average"][0]); ax.set_xlabel(top); ax.set_ylabel("Predicted mpg"); ax.set_title("Partial dependence"); save_figure(fig, NB_ID, "pdp"); plt.show()
ranks = []
rng = np.random.default_rng(SEED)
for _ in range(20):
    i = rng.integers(0, len(Xtv), len(Xtv))
    m = make_model_pipeline(Xtv.iloc[i], HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=int(rng.integers(1e6))), scale=False).fit(Xtv.iloc[i], ytv.iloc[i])
    pi = permutation_importance(m, Xte, yte, scoring="r2", n_repeats=5, random_state=0).importances_mean
    ranks.append(pd.Series(pi, index=X.columns).rank(ascending=False))
rank_tab = pd.DataFrame(ranks).agg(["mean", "std"]).T.sort_values("mean"); display(rank_tab.round(2))
METRICS = {"test_R2": float(r2_score(yte, model.predict(Xte))), "top_feature_importance": float(tab.iloc[0]["importance"]), "top_feature_rank_std": float(rank_tab.loc[top, "std"]), "n_bootstrap": 20.0}
CONFIG = {"model": "HistGradientBoostingRegressor", "repeats": 20}; EXTRA = {"importance": tab.round(4).reset_index().to_dict("records"), "top_feature": top}; ALGORITHM = "permutation importance, PDP"''',
   interp="A feature with a large importance and a small rank standard deviation is a reliable driver. Features whose rank moves a lot (large std) are correlated with others (weight, displacement, horsepower), so importance is shared between them and should not be read one by one.",
   limits="Permutation importance treats correlated features as independent, which can create unrealistic rows. Partial dependence averages over such rows. Both describe the model, not causal effects.",
   exercises=["Group correlated features and permute them together.", "Add individual conditional expectation curves.", "Install SHAP and compare its ranking with permutation importance."],
   refs="Breiman (2001) `breiman2001` introduced permutation importance for forests. Friedman (2001) `friedman2001` introduced partial dependence plots. Both are in `research/source-register.md`.")

nb(id="hyperparameter_optimization", title="Hyperparameter Optimization: Random Search vs Grid Search with Nested Validation", dataset="digits", task="classification",
   tagline="Does a larger search budget help, and how optimistic is the score of the best configuration?",
   problem="The best score found by a search is biased upward because it was selected as the maximum. Nested cross validation measures that bias.",
   method="Support vector machine with C and gamma. Inner loop: random search with 5, 20 and 60 configurations versus a 6 by 6 grid. Outer loop: 5-fold stratified. We report the inner best score and the outer score on untouched folds.",
   data_md="Optical digits, 1797 rows, 64 features (see catalog).",
   load='''b = load_dataset("digits"); X, y = b.X, b.y
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
pipe = make_pipeline(StandardScaler(), SVC())
print(X.shape)''',
   exp_md="Run the outer loop for each search strategy.",
   exp='''outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
strategies = {"random 5": lambda: RandomizedSearchCV(pipe, {"svc__C": loguniform(1e-2, 1e3), "svc__gamma": loguniform(1e-5, 1e-1)}, n_iter=5, cv=3, random_state=SEED, n_jobs=2),
              "random 20": lambda: RandomizedSearchCV(pipe, {"svc__C": loguniform(1e-2, 1e3), "svc__gamma": loguniform(1e-5, 1e-1)}, n_iter=20, cv=3, random_state=SEED, n_jobs=2),
              "random 60": lambda: RandomizedSearchCV(pipe, {"svc__C": loguniform(1e-2, 1e3), "svc__gamma": loguniform(1e-5, 1e-1)}, n_iter=60, cv=3, random_state=SEED, n_jobs=2),
              "grid 36": lambda: GridSearchCV(pipe, {"svc__C": np.logspace(-2, 3, 6), "svc__gamma": np.logspace(-5, -1, 6)}, cv=3, n_jobs=2)}
res_rows = []
for name, make in strategies.items():
    inner, outer_s = [], []
    for tr, te in outer.split(X, y):
        s = make().fit(X.iloc[tr], y.iloc[tr]); inner.append(s.best_score_); outer_s.append(s.score(X.iloc[te], y.iloc[te]))
    res_rows.append({"strategy": name, "inner_best_score": np.mean(inner), "outer_score": np.mean(outer_s), "outer_std": np.std(outer_s, ddof=1), "optimism": np.mean(inner) - np.mean(outer_s)})
print("done")''',
   res_md="Inner best versus outer score.",
   res='''tab = pd.DataFrame(res_rows).set_index("strategy"); display(tab.round(4))
default = cross_val_score(pipe, X, y, cv=outer).mean()
print("default SVC (no search) outer accuracy:", round(float(default), 4))
METRICS = {f"{n.replace(' ', '_')}_{k}": float(v) for n, r in tab.iterrows() for k, v in r.items()}
METRICS["default_svc_accuracy"] = float(default)
CONFIG = {"outer_cv": 5, "inner_cv": 3, "model": "SVC RBF"}; EXTRA = {"table": tab.round(4).reset_index().to_dict("records")}; ALGORITHM = "random search vs grid search"''',
   interp="The inner score is lower than the outer score here (negative optimism). The reason is that the inner folds train on 3-fold subsets of 80 percent of the data, so they see less data than the outer fit. The usual upward selection bias exists but is smaller than this effect, so the gap is not a clean estimate of optimism. All four strategies and the default SVC land within one outer standard deviation of each other, so on this dataset the search brings no measurable gain. That is a result about this easy dataset, not about search in general.",
   limits="One model and one dataset. Digits is an easy task, so the search has little room. Search costs are counted in configurations, not in measured seconds.",
   exercises=["Add successive halving (`HalvingRandomSearchCV`).", "Repeat with five outer seeds and average.", "Try the same on a harder dataset such as Telco."],
   refs="Bergstra and Bengio (2012), *Random search for hyper-parameter optimization*, JMLR, motivates random search. It is **not verified in the build environment**.")

nb(id="model_robustness", title="Model Robustness: Noise, Missing Values and Distribution Shift", dataset="digits", task="classification",
   tagline="How fast does accuracy fall when test inputs are degraded, and which models fall slowest?",
   problem="Test accuracy on clean data from the same source overstates field performance. We degrade test inputs in three controlled ways and track accuracy.",
   method="Train logistic regression, random forest, an RBF SVM with standardised inputs and the same SVM without scaling on clean data. Test under (a) Gaussian pixel noise, (b) random pixel dropout (set to zero), and (c) a shift: shifting every image by one pixel to the right. Corruptions are applied only to test rows.",
   data_md="Optical digits, 1797 rows, 8 by 8 images (see catalog).",
   load='''b = load_dataset("digits"); X, y = b.X, b.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="classification", seed=SEED)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
models = {"logistic": (LogisticRegression(max_iter=3000), True), "random_forest": (RandomForestClassifier(300, n_jobs=2, random_state=SEED), True),
          "svm_rbf": (SVC(C=10), True), "svm_rbf_no_scaling": (SVC(C=10), False)}
fit = {n: make_model_pipeline(Xtv, m, scale=sc).fit(Xtv, ytv) for n, (m, sc) in models.items()}
print({n: round(float((f.predict(Xte) == yte).mean()), 3) for n, f in fit.items()})''',
   exp_md="Define corruptions and evaluate.",
   exp='''rng = np.random.default_rng(SEED)
def noise(a, s): return np.clip(a + rng.normal(0, s, a.shape), 0, 16)
def dropout(a, p): return a * (rng.random(a.shape) > p)
def shift(a):
    im = a.reshape(-1, 8, 8); out = np.zeros_like(im); out[:, :, 1:] = im[:, :, :-1]; return out.reshape(len(a), 64)
tests = {"clean": Xte.values, "noise 2": noise(Xte.values, 2), "noise 4": noise(Xte.values, 4), "dropout 20%": dropout(Xte.values, 0.2), "dropout 40%": dropout(Xte.values, 0.4), "shift 1 px": shift(Xte.values)}
rows = {n: {t: float((f.predict(pd.DataFrame(a, columns=Xte.columns)) == yte.values).mean()) for t, a in tests.items()} for n, f in fit.items()}
tab = pd.DataFrame(rows).T; display(tab.round(3))''',
   res_md="Accuracy table and drop from clean.",
   res='''drop = tab.sub(tab["clean"], axis=0).drop(columns="clean"); display(drop.round(3))
fig, ax = plt.subplots(figsize=(8, 3.8)); tab.T.plot.bar(ax=ax, rot=20); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy under corruption"); save_figure(fig, NB_ID, "robustness"); plt.show()
METRICS = {f"{n}_{t.replace(' ', '_').replace('%', 'pct')}": float(v) for n, r in tab.iterrows() for t, v in r.items()}
CONFIG = {"corruptions": list(tests)}; EXTRA = {"table": tab.round(4).reset_index().to_dict("records")}; ALGORITHM = "robustness test of three classifiers"''',
   interp="**Observation:** the scaled SVM has the best clean accuracy and the worst noise accuracy, while the SVM without scaling is far more stable. **Inference (checked in a separate run during development, not in this notebook):** standardisation divides by the training standard deviation, and a few pixels are nearly constant in training data (some have standard deviation 0), so noise on those pixels becomes a very large standardised value that dominates the RBF distance. Check this by counting pixels with a small training standard deviation. The lesson is general: the model with the best clean accuracy is not necessarily the most robust, and preprocessing can create fragility. The shift test hurts every model because all rely on pixel positions.",
   limits="Corruptions are synthetic and applied once with one seed. The results say nothing about natural shift (new writers, new scanners).",
   exercises=["Train with augmented images and compare.", "Repeat each corruption with 10 seeds and add intervals.", "Add a small convolutional network."],
   refs="Hendrycks and Dietterich (2019), *Benchmarking neural network robustness to common corruptions and perturbations*, ICLR, is the usual reference. It is **not verified in the build environment**.")

if __name__ == "__main__":
    ids = sys.argv[1:]
    for p in NB:
        if not ids or p["id"] in ids:
            render(p)
