"""Build end-to-end project notebooks."""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).parent))
from refs import REFS, bibtex  # noqa: E402
import specs_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_end_to_end_projects"

PRE = '''import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from ml_lab.data import load_dataset
from ml_lab.preprocessing import split_three_way, make_model_pipeline, build_preprocessor
from ml_lab.evaluation import evaluate_estimator, internal_scores, external_scores
from ml_lab.experiments import ExperimentRecord, save_figure
from ml_lab.utils import set_seed, print_environment
from ml_lab import visualization as viz
PROJECT_ID = "__ID__"
SEED = 42
set_seed(SEED)
viz.apply_style()
print_environment()'''

SAVE = '''record = ExperimentRecord(notebook=PROJECT_ID, family="project", algorithm=ALGORITHM, dataset="__DATASET__", task="__TASK__", seed=SEED, config=CONFIG, metrics=METRICS, extra=EXTRA)
print("saved:", record.save())
display(pd.Series(METRICS, name="metric").round(4).to_frame())'''

PROJECTS = []


def project(**kw):
    PROJECTS.append(kw)


def refs_md(keys):
    out = []
    for k in keys:
        r = REFS[k]
        out.append(f"- {r['authors']} ({r['year']}). *{r['title']}*. {r['venue']}. {r['url'] or ''} Verification: {r['verified']}")
    return "\n".join(out)


def render(p):
    folder = BASE / p["id"]
    folder.mkdir(parents=True, exist_ok=True)
    kw = dict(ID=p["id"], DATASET=p["dataset"], TASK=p["task"])
    sub = lambda t: t.replace("__ID__", kw["ID"]).replace("__DATASET__", kw["DATASET"]).replace("__TASK__", kw["TASK"])  # noqa: E731
    toc = ["1. Problem statement", "2. Business or research context", "3. Dataset documentation", "4. Research references", "5. Exploratory analysis",
           "6. Preprocessing", "7. Model selection", "8. Model training", "9. Evaluation", "10. Comparison", "11. Interpretation", "12. Limitations", "13. Final recommendations"]
    cells = [new_markdown_cell(f"# {p['title']}\n\n**Family:** End-to-end project | **Dataset:** `{p['dataset']}` | **Task:** {p['task']}\n\n{p['tagline']}\n\n## Table of Contents\n\n" + "\n".join(f"- {t}" for t in toc)),
             new_markdown_cell("## 1. Problem statement\n\n" + p["problem"]),
             new_markdown_cell("## 2. Business or research context\n\n" + p["context"]),
             new_markdown_cell("## 3. Dataset documentation\n\n" + p["data_md"]),
             new_markdown_cell("## 4. Research references\n\nVerification status is from the build environment on 2026-11-17. Entries marked *not verified* must be checked before citing.\n\n" + refs_md(p["refs"]) + "\n\n" + p["refs_note"]),
             new_code_cell(sub(PRE)),
             new_code_cell(p["load"]),
             new_markdown_cell("## 5. Exploratory analysis\n\n" + p["eda_md"]),
             new_code_cell(p["eda"]),
             new_markdown_cell("## 6. Preprocessing\n\n" + p["prep_md"]),
             new_code_cell(p["prep"]),
             new_markdown_cell("## 7. Model selection\n\n" + p["select_md"]),
             new_code_cell(p["select"]),
             new_markdown_cell("## 8. Model training\n\n" + p["train_md"]),
             new_code_cell(p["train"]),
             new_markdown_cell("## 9. Evaluation\n\n" + p["eval_md"]),
             new_code_cell(p["eval"] + "\n\n" + sub(SAVE)),
             new_markdown_cell("## 10. Comparison\n\n" + p["compare_md"]),
             new_code_cell(p["compare"]),
             new_markdown_cell("## 11. Interpretation\n\n" + p["interp_md"]),
             new_code_cell(p["interp"]),
             new_markdown_cell("## 12. Limitations\n\n" + p["limits"]),
             new_markdown_cell("## 13. Final recommendations\n\n" + p["recs"])]
    nbk = new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.validate(nbk)
    path = folder / f"{p['id']}.ipynb"
    nbformat.write(nbk, path)
    (folder / "README.md").write_text(f"# {p['title']}\n\n{p['tagline']}\n\n- Notebook: [`{path.name}`]({path.name})\n- Dataset: `{p['dataset']}`\n- Results: `experiments/results/{p['id']}.json`\n\n"
                                      f"## Problem\n\n{p['problem']}\n\n## Limits\n\n{p['limits']}\n")
    (folder / "references.md").write_text(f"# References: {p['title']}\n\n{refs_md(p['refs'])}\n\n```bibtex\n" + "\n\n".join(bibtex(k) for k in p["refs"]) + "\n```\n")
    print("built", path.relative_to(ROOT))


TELCO_DOC = C.DS_TELCO
CLS_LOAD_TELCO = '''bundle = load_dataset("telco_churn")
X, y = bundle.X, bundle.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="classification", seed=SEED)
print("train / validation / test rows:", len(Xtr), len(Xva), len(Xte), "| churn share in train:", round(float(ytr.mean()), 3))
display(pd.Series(bundle.describe_source(), name="source").to_frame())'''

# ------------------------------------------------------------------ 1 churn
project(id="customer_churn", title="Customer Churn Prediction and Retention Targeting", dataset="telco_churn", task="classification",
        tagline="From a ranked churn model to a cost based contact list, with calibration and an honest baseline.",
        problem="Predict which telecom customers will leave and decide whom to contact with a retention offer.",
        context="Keeping a customer usually costs less than winning a new one. A retention team has a limited budget, so a good ranking and a justified cut-off matter more than accuracy. "
                "The cost numbers below are **assumptions for illustration** (a missed churner costs 5 units, a wasted offer costs 1 unit). Replace them with the real figures of a business before using the cut-off.",
        data_md=TELCO_DOC, refs=["breiman2001", "friedman2001", "grinsztajn2022"],
        refs_note="These papers cover the model families compared here. None of them uses this dataset, so no number is comparable.",
        load=CLS_LOAD_TELCO,
        eda_md="Churn rate, missing values and the link between contract type, tenure and churn.",
        eda='''print("overall churn rate:", round(float(y.mean()), 3), "| duplicate rows:", int(X.duplicated().sum()), "| missing TotalCharges:", int(X["TotalCharges"].isna().sum()))
display(pd.concat([X["Contract"], y.rename("churn")], axis=1).groupby("Contract")["churn"].agg(["mean", "count"]).round(3))
fig, ax = plt.subplots(figsize=(6, 3.5))
for v, label in [(0, "stayed"), (1, "churned")]:
    ax.hist(X.loc[y == v, "tenure"], bins=24, alpha=0.6, label=label)
ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Customers"); ax.legend(); ax.set_title("Churners leave early"); plt.show()''',
        prep_md="One preprocessor (median imputation, one hot encoding, scaling for linear models) fitted on training rows only. The test rows are used once at the end.",
        prep='''prep_demo = build_preprocessor(Xtr)
print("encoded feature count:", prep_demo.fit(Xtr).transform(Xva).shape[1])''',
        select_md="Candidates: a logistic regression, a random forest and a histogram gradient boosting model. The primary metric is PR-AUC (area under the precision recall curve) because churn is the minority class. The dummy model gives the floor.",
        select='''from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
candidates = {"dummy": (DummyClassifier(strategy="prior"), True),
              "logistic": (LogisticRegression(max_iter=2000, C=0.5), True),
              "random_forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=2, random_state=SEED), False),
              "hist_gradient_boosting": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED), False)}
fitted, rows = {}, []
for name, (model, scale) in candidates.items():
    pipe = make_model_pipeline(Xtr, model, scale=scale).fit(Xtr, ytr)
    fitted[name] = pipe
    p = pipe.predict_proba(Xva)[:, 1]
    rows.append({"model": name, "validation_PR_AUC": average_precision_score(yva, p), "validation_ROC_AUC": roc_auc_score(yva, p)})
sel = pd.DataFrame(rows).set_index("model")
display(sel.round(4))
BEST = sel.drop("dummy")["validation_PR_AUC"].idxmax()
print("selected on validation PR-AUC:", BEST)''',
        train_md="The selected model is refit on training plus validation rows. Five fold cross validation on those rows gives the spread of the score.",
        train='''Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
model, scale = candidates[BEST]
final = make_model_pipeline(Xtv, clone(model), scale=scale).fit(Xtv, ytv)
cv = cross_val_score(make_model_pipeline(Xtv, clone(model), scale=scale), Xtv, ytv, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="average_precision")
print(f"5-fold PR-AUC on train plus validation: {cv.mean():.3f} +- {cv.std(ddof=1):.3f}")''',
        eval_md="Test set, scored once. Metrics at the default 0.5 threshold and ranking metrics. Then a cost based threshold chosen on the validation rows (with the model trained on the training rows only) and applied to the test rows.",
        eval='''from sklearn.metrics import precision_score, recall_score, f1_score
p_test = final.predict_proba(Xte)[:, 1]
base_p = fitted["dummy"].predict_proba(Xte)[:, 1]
FN_COST, FP_COST = 5.0, 1.0
pv = fitted[BEST].predict_proba(Xva)[:, 1]
grid = np.linspace(0.05, 0.95, 91)
val_cost = [FN_COST * ((pv < t) & (yva == 1)).sum() + FP_COST * ((pv >= t) & (yva == 0)).sum() for t in grid]
THRESH = float(grid[int(np.argmin(val_cost))])
def cost(y_true, score, t):
    return FN_COST * ((score < t) & (y_true == 1)).sum() + FP_COST * ((score >= t) & (y_true == 0)).sum()
test_cost = {"model at cost threshold": cost(yte, p_test, THRESH), "contact nobody": cost(yte, np.zeros(len(yte)), 0.5), "contact everybody": cost(yte, np.ones(len(yte)), 0.5)}
METRICS = {"test_PR_AUC": average_precision_score(yte, p_test), "test_ROC_AUC": roc_auc_score(yte, p_test), "baseline_PR_AUC": average_precision_score(yte, base_p),
           "precision_at_threshold": precision_score(yte, p_test >= THRESH), "recall_at_threshold": recall_score(yte, p_test >= THRESH), "F1_at_threshold": f1_score(yte, p_test >= THRESH),
           "cv_PR_AUC_mean": float(cv.mean()), "cv_PR_AUC_std": float(cv.std(ddof=1)), "cost_model": float(test_cost["model at cost threshold"]), "cost_contact_nobody": float(test_cost["contact nobody"]), "cost_contact_everybody": float(test_cost["contact everybody"])}
CONFIG = {"selected_model": BEST, "threshold": THRESH, "fn_cost": FN_COST, "fp_cost": FP_COST}
EXTRA = {"validation_table": sel.round(4).reset_index().to_dict("records"), "cost_note": "costs are assumptions"}
ALGORITHM = BEST
print("threshold from validation:", round(THRESH, 2)); display(pd.Series(test_cost, name="test cost (assumed units)").round(1).to_frame())''',
        compare_md="All candidates on the same test rows (this table is for comparison only, the choice was made on validation rows) and a calibration check of the selected model.",
        compare='''rows = []
for name, pipe in fitted.items():
    pr = pipe.predict_proba(Xte)[:, 1]
    rows.append({"model": name, "test_PR_AUC": average_precision_score(yte, pr), "test_ROC_AUC": roc_auc_score(yte, pr)})
display(pd.DataFrame(rows).set_index("model").round(4))
from sklearn.calibration import calibration_curve
frac, mean_pred = calibration_curve(yte, p_test, n_bins=8, strategy="quantile")
fig, ax = plt.subplots(figsize=(4.5, 4)); ax.plot(mean_pred, frac, marker="o"); ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Observed churn share"); ax.set_title("Calibration, test rows"); save_figure(fig, PROJECT_ID, "calibration"); plt.show()''',
        interp_md="Permutation importance on the test rows (drop in PR-AUC when one column is shuffled) shows what the ranking depends on.",
        interp='''imp = permutation_importance(final, Xte, yte, scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=1)
display(pd.Series(imp.importances_mean, index=Xte.columns).sort_values(ascending=False).head(8).round(4).to_frame("PR-AUC drop"))''',
        limits="One snapshot with no time dimension, so we cannot test drift. The cost numbers are assumptions. Duplicates were kept. The data has no information about what offers work, so the model ranks risk but does not say which action to take (that needs an experiment with a control group).",
        recs="Use the model to rank customers, not to decide actions alone. Validate the cost ratio with the retention team, check calibration on fresh data every month, and run a controlled retention test before scaling.")

# ------------------------------------------------------------------ 2 segmentation
project(id="customer_segmentation", title="Customer Segmentation with Clustering", dataset="telco_churn", task="clustering",
        tagline="Segment telecom customers on usage and tenure, check stability, and read the segments with churn as an outside check.",
        problem="Group customers into segments that a marketing team can act on, using account and service information only.",
        context="Segments let a business tailor offers. A segmentation is useful only if it is stable under resampling and the segments differ in ways people can describe. Churn is **not** used to fit the clusters. It is used afterwards as an external check.",
        data_md=TELCO_DOC, refs=["lloyd1982", "rousseeuw1987", "dempster1977"],
        refs_note="K-means (Lloyd), the silhouette (Rousseeuw) and Gaussian mixtures (Dempster et al.) are compared.",
        load='''bundle = load_dataset("telco_churn")
X, y = bundle.X, bundle.y
print(X.shape)''',
        eda_md="We use tenure, monthly charges, total charges and the count of add-on services.",
        eda='''services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
F = pd.DataFrame({"tenure": X["tenure"], "MonthlyCharges": X["MonthlyCharges"], "TotalCharges": X["TotalCharges"].fillna(X["TotalCharges"].median()),
                  "n_addons": X[services].eq("Yes").sum(axis=1)})
display(F.describe().round(1))
print("correlation of tenure and TotalCharges:", round(float(F["tenure"].corr(F["TotalCharges"])), 3))''',
        prep_md="Standardise the four features. There is no train and test split because we describe existing customers, but we test stability by resampling.",
        prep='''from sklearn.preprocessing import StandardScaler
Z = StandardScaler().fit_transform(F)
print(Z.shape)''',
        select_md="Compare k-means and a Gaussian mixture for k from 2 to 8 using silhouette (k-means) and BIC (mixture).",
        select='''from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    gm = GaussianMixture(k, n_init=3, random_state=SEED).fit(Z)
    rows.append({"k": k, "kmeans_silhouette": internal_scores(Z, km.labels_)["Silhouette"], "gmm_BIC": gm.bic(Z)})
sel = pd.DataFrame(rows).set_index("k")
display(sel.round(3))
K = int(sel["kmeans_silhouette"].idxmax()); print("k-means k by silhouette:", K, "| GMM k by BIC:", int(sel["gmm_BIC"].idxmin()))''',
        train_md="Fit k-means with the chosen k and test stability on 20 pairs of 80 percent subsamples.",
        train='''from sklearn.metrics import adjusted_rand_score
model = KMeans(K, n_init=10, random_state=SEED).fit(Z)
labels = model.labels_
rng = np.random.default_rng(SEED); aris = []
for _ in range(20):
    a, b = rng.choice(len(Z), int(0.8 * len(Z)), replace=False), rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
    common = np.intersect1d(a, b)
    aris.append(adjusted_rand_score(KMeans(K, n_init=5, random_state=0).fit(Z[a]).predict(Z[common]), KMeans(K, n_init=5, random_state=1).fit(Z[b]).predict(Z[common])))
print("stability ARI over 20 resample pairs: mean", round(float(np.mean(aris)), 3), "std", round(float(np.std(aris, ddof=1)), 3))''',
        eval_md="Internal scores, the stability result and the churn rate of each segment (an outside check, not a fit target).",
        eval='''s = internal_scores(Z, labels)
seg = F.assign(segment=labels, churn=y.values)
profile = seg.groupby("segment").agg(customers=("churn", "size"), churn_rate=("churn", "mean"), tenure=("tenure", "mean"), monthly=("MonthlyCharges", "mean"), addons=("n_addons", "mean")).round(2)
display(profile)
METRICS = {**s, "stability_ARI_mean": float(np.mean(aris)), "stability_ARI_std": float(np.std(aris, ddof=1)), "churn_rate_min_segment": float(profile.churn_rate.min()), "churn_rate_max_segment": float(profile.churn_rate.max())}
CONFIG = {"k": K, "features": list(F.columns), "n_init": 10}
EXTRA = {"profile": profile.reset_index().to_dict("records")}
ALGORITHM = "KMeans"''',
        compare_md="Compare k-means with a Gaussian mixture at the same k, using agreement (ARI) and silhouette.",
        compare='''gm = GaussianMixture(K, n_init=3, random_state=SEED).fit(Z)
print("agreement k-means vs GMM (ARI):", round(adjusted_rand_score(labels, gm.predict(Z)), 3), "| GMM silhouette:", round(internal_scores(Z, gm.predict(Z))["Silhouette"], 3), "| k-means silhouette:", round(s["Silhouette"], 3))''',
        interp_md="Plot the segments in the plane of tenure and monthly charges and read the profile table.",
        interp='''fig, ax = plt.subplots(figsize=(6, 4.5))
sc = ax.scatter(F["tenure"], F["MonthlyCharges"], c=labels, cmap="tab10", s=6, alpha=0.5)
ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Monthly charges"); ax.set_title("Segments"); save_figure(fig, PROJECT_ID, "segments"); plt.show()''',
        limits="Four numeric features only, so segments mostly separate by tenure and price. Segment churn rates are an association, not a causal effect. Silhouette values are low for this kind of data, which means segments overlap.",
        recs="Use the segments as a descriptive tool. Check stability on new months of data, add behavioural features (usage, support calls) when available, and test an offer per segment with a control group.")

# ------------------------------------------------------------------ 3 medical
project(id="medical_classification", title="Medical Classification: Breast Tumour Diagnosis Support", dataset="breast_cancer", task="classification",
        tagline="Choose a model by sensitivity at a fixed specificity target, with intervals and an error review.",
        problem="Classify tumours as malignant or benign from nucleus measurements and report how the model errs.",
        context="In screening, a missed malignancy is worse than a false alarm. We therefore fix a sensitivity target of 0.95 on validation data and report specificity at that point. This is a **teaching example, not a medical device**.",
        data_md=C.DS_BC, refs=["wolberg1990", "cortes1995", "breiman2001"],
        refs_note="Wolberg and Mangasarian describe cytology based diagnosis from the same research group. The other two define the models compared.",
        load='''bundle = load_dataset("breast_cancer")
X, y = bundle.X, bundle.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="classification", seed=SEED)
print(len(Xtr), len(Xva), len(Xte), "| malignant share:", round(float(y.mean()), 3))''',
        eda_md="Class balance and the strongest single features.",
        eda='''from sklearn.metrics import roc_auc_score
single = pd.Series({c: roc_auc_score(ytr, Xtr[c]) for c in X.columns})
single = (single - 0.5).abs().sort_values(ascending=False) + 0.5
display(single.head(6).round(3).to_frame("single feature ROC-AUC"))''',
        prep_md="Scaling inside the pipeline. No imputation needed (no missing values).",
        prep='''print("missing values:", int(X.isna().sum().sum()))''',
        select_md="Logistic regression, RBF support vector machine and random forest. The model is chosen by validation ROC-AUC. The threshold is then set so that validation sensitivity is at least 0.95.",
        select='''from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
cands = {"logistic": LogisticRegression(max_iter=3000, C=1.0), "svm_rbf": SVC(probability=True, C=3, random_state=SEED), "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=2)}
fitted = {n: make_model_pipeline(Xtr, m).fit(Xtr, ytr) for n, m in cands.items()}
sel = pd.Series({n: roc_auc_score(yva, f.predict_proba(Xva)[:, 1]) for n, f in fitted.items()}, name="validation ROC-AUC")
display(sel.round(4).to_frame()); BEST = sel.idxmax(); print("selected:", BEST)''',
        train_md="Refit the selected model on training plus validation rows. The threshold comes from the validation scores of the model trained on the training rows.",
        train='''pv = fitted[BEST].predict_proba(Xva)[:, 1]
cands_t = np.unique(pv)
ok = [t for t in cands_t if ((pv >= t) & (yva == 1)).sum() / (yva == 1).sum() >= 0.95]
THRESH = float(max(ok)) if ok else 0.5
Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
final = make_model_pipeline(Xtv, clone(cands[BEST])).fit(Xtv, ytv)
print("threshold giving at least 0.95 validation sensitivity:", round(THRESH, 3))''',
        eval_md="Test results with 95 percent bootstrap intervals (1000 resamples). With 114 test rows the intervals are wide, and that width is the honest uncertainty.",
        eval='''from sklearn.metrics import confusion_matrix
pt = final.predict_proba(Xte)[:, 1]
pred = (pt >= THRESH).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
rng = np.random.default_rng(SEED); sens, spec, auc = [], [], []
yv = yte.values
for _ in range(1000):
    i = rng.integers(0, len(yv), len(yv))
    if yv[i].min() == yv[i].max():
        continue
    s_ = ((pred[i] == 1) & (yv[i] == 1)).sum() / (yv[i] == 1).sum(); p_ = ((pred[i] == 0) & (yv[i] == 0)).sum() / (yv[i] == 0).sum()
    sens.append(s_); spec.append(p_); auc.append(roc_auc_score(yv[i], pt[i]))
ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
METRICS = {"test_ROC_AUC": roc_auc_score(yte, pt), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "false_negatives": int(fn), "false_positives": int(fp),
           "sensitivity_ci_low": ci(sens)[0], "sensitivity_ci_high": ci(sens)[1], "specificity_ci_low": ci(spec)[0], "specificity_ci_high": ci(spec)[1]}
CONFIG = {"selected_model": BEST, "threshold": THRESH, "target_validation_sensitivity": 0.95}
EXTRA = {"validation_ROC_AUC": sel.round(4).to_dict(), "test_rows": int(len(yte))}
ALGORITHM = BEST
display(pd.DataFrame([[tn, fp], [fn, tp]], index=["actual benign", "actual malignant"], columns=["predicted benign", "predicted malignant"]))''',
        compare_md="All three models at their own validation derived thresholds on the same test rows.",
        compare='''rows = []
for n, f in fitted.items():
    p = f.predict_proba(Xva)[:, 1]
    okt = [t for t in np.unique(p) if ((p >= t) & (yva == 1)).sum() / (yva == 1).sum() >= 0.95]
    t = max(okt) if okt else 0.5
    q = f.predict_proba(Xte)[:, 1] >= t
    rows.append({"model": n, "threshold": t, "test_sensitivity": ((q == 1) & (yte == 1)).sum() / (yte == 1).sum(), "test_specificity": ((q == 0) & (yte == 0)).sum() / (yte == 0).sum()})
display(pd.DataFrame(rows).set_index("model").round(3))''',
        interp_md="Which features drive the model, and what do the missed cases look like.",
        interp='''imp = permutation_importance(final, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=SEED)
display(pd.Series(imp.importances_mean, index=Xte.columns).sort_values(ascending=False).head(6).round(4).to_frame("ROC-AUC drop"))
missed = Xte[(pred == 0) & (yte.values == 1)]
print("missed malignant rows:", len(missed)); display(missed[["radius_mean", "area_mean", "concavity_mean"]].round(3) if "radius_mean" in missed else missed.iloc[:, :3])''',
        limits="One institution, 569 rows, only 114 test rows. No external validation. The data are measurements from images, not images. The result must not be used for patient care.",
        recs="Treat this as a method demonstration. Real use needs external validation on another site, clinical review of error costs, calibration checks and regulatory review.")

# ------------------------------------------------------------------ 4 control
project(id="autonomous_control", title="Autonomous Control: Balancing a Cart-Pole with a Heuristic and PPO", dataset="CartPole-v1", task="control",
        tagline="Compare a hand written controller, a random policy and PPO, then test robustness to noise.",
        problem="Keep a pole balanced on a cart for 500 steps, and find out how robust each controller is to sensor and actuator noise.",
        context="Learned controllers are only useful if they beat a simple engineered baseline and survive realistic noise. We test three controllers on the same evaluation seeds. This is a simulation, not a real vehicle.",
        data_md="**CartPole-v1** (Gymnasium classic control), simulated. 4 state values, 2 actions, reward 1 per step, maximum return 500. There is no dataset and no licence issue for the data. Gymnasium is MIT licensed.",
        refs=["schulman2017", "towers2024", "henderson2018"], refs_note="PPO (Schulman et al.), the environment API (Towers et al.) and a warning about seeds and reporting (Henderson et al.).",
        load='''import gymnasium as gym
from ml_lab.rl import deep
from ml_lab.evaluation import evaluate_policy
make_env = lambda: gym.make("CartPole-v1")
print(make_env().observation_space, make_env().action_space)''',
        eda_md="Behaviour of a random policy and of the simple heuristic on 30 episodes.",
        eda='''rng = np.random.default_rng(0)
random_policy = lambda s: int(rng.integers(2))
heuristic = lambda s: int(s[2] + 0.5 * s[3] > 0)            # push toward the lean
for name, pol in [("random", random_policy), ("heuristic", heuristic)]:
    r = evaluate_policy(make_env, pol, episodes=30, seed=2000)
    print(name, {k: round(v, 1) for k, v in r.items()})''',
        prep_md="No data preparation. Training seeds are 0, 1 and 2 with 40 960 steps each. Evaluation uses 30 episodes with environment seeds from 5000 that training never used. Noise is added to observations (Gaussian) or to actions (random flip).",
        prep='''SEEDS = [0, 1, 2]
def noisy(policy, obs_std=0.0, flip=0.0, seed=0):
    r = np.random.default_rng(seed)
    def wrapped(s):
        s = np.asarray(s) + r.normal(0, obs_std, size=len(s)) if obs_std else s
        a = policy(s)
        return int(1 - a) if r.random() < flip else a
    return wrapped''',
        select_md="Controllers: random, hand written heuristic, PPO from `ml_lab.rl`. No search over controllers is done, so there is no selection on test data.",
        select='''print("PPO config: rollout 1024, 10 epochs, minibatch 128, lr 1e-3, clip 0.2, gamma 0.99, lambda 0.95")''',
        train_md="Train PPO with three seeds.",
        train='''runs = {s: deep.train_ppo(seed=s, total_steps=40960) for s in SEEDS}
for s in SEEDS:
    print("seed", s, "| last 10 training episodes:", round(float(np.mean(runs[s]["returns"][-10:])), 1))''',
        eval_md="Return over 30 evaluation episodes per condition. PPO is evaluated per seed and averaged over the three seeds.",
        eval='''conds = {"clean": dict(), "observation noise 0.05": dict(obs_std=0.05), "observation noise 0.1": dict(obs_std=0.1), "action flip 10%": dict(flip=0.1)}
table = {}
for cname, kw in conds.items():
    row = {}
    row["random"] = evaluate_policy(make_env, noisy(random_policy, seed=1, **kw), episodes=30, seed=5000)["mean_return"]
    row["heuristic"] = evaluate_policy(make_env, noisy(heuristic, seed=1, **kw), episodes=30, seed=5000)["mean_return"]
    ppo = [evaluate_policy(make_env, noisy(runs[s]["policy"], seed=1, **kw), episodes=30, seed=5000)["mean_return"] for s in SEEDS]
    row["ppo_mean"], row["ppo_std_across_seeds"] = float(np.mean(ppo)), float(np.std(ppo, ddof=1))
    table[cname] = row
res = pd.DataFrame(table).T
display(res.round(1))
METRICS = {f"{c.replace(' ', '_').replace('%', 'pct')}_{k}": float(v) for c, row in table.items() for k, v in row.items()}
CONFIG = {"algorithm": "PPO", "total_steps": 40960, "seeds": 3, "eval_episodes": 30}
EXTRA = {"table": res.round(2).reset_index().to_dict("records")}
ALGORITHM = "PPO"''',
        compare_md="The table above is the comparison. The plot shows returns per condition.",
        compare='''ax = res[["random", "heuristic", "ppo_mean"]].plot.bar(figsize=(8, 4), rot=15)
ax.set_ylabel("Mean evaluation return (30 episodes)"); ax.set_title("Controllers under noise"); save_figure(ax.figure, PROJECT_ID, "robustness"); plt.show()''',
        interp_md="Compare the drop in return from the clean condition to each noisy condition.",
        interp='''drop = res[["heuristic", "ppo_mean"]].rsub(res.loc["clean", ["heuristic", "ppo_mean"]], axis=1)
display(drop.round(1).rename(columns={"heuristic": "heuristic drop", "ppo_mean": "PPO drop"}))''',
        limits="Three seeds and a small training budget. A simple heuristic may beat PPO or not. The table shows which on this setup. The simulator is noise free by default and real systems have delay, friction and sensor faults that are not modelled.",
        recs="Never deploy a learned controller without an engineered baseline and a noise test. For a real system add safety limits, more seeds with intervals, and tests on a higher fidelity simulator.")

# ------------------------------------------------------------------ 5 fuel efficiency
project(id="fuel_efficiency_regression", title="Fuel Efficiency Regression", dataset="auto_mpg", task="regression",
        tagline="Predict miles per gallon from engine and weight, with a model comparison and residual review. Substitutes the energy forecasting case for which no open data could be downloaded.",
        problem="Predict a car's fuel economy from its design.",
        context="Vehicle design teams trade weight, power and efficiency. A transparent model shows the cost of each design choice. The dataset is old (1970 to 1982), so conclusions about modern cars do not follow.",
        data_md=C.DS_MPG, refs=["hoerl1970", "tibshirani1996", "friedman2001"],
        refs_note="Ridge, lasso and gradient boosting are compared, with a linear baseline.",
        load='''bundle = load_dataset("auto_mpg")
X, y = bundle.X, bundle.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="regression", seed=SEED)
print(len(Xtr), len(Xva), len(Xte))''',
        eda_md="Target distribution and the relation with weight.",
        eda='''display(pd.concat([X, y.rename("mpg")], axis=1).corr()["mpg"].drop("mpg").round(2).sort_values().to_frame("correlation with mpg"))
fig, ax = plt.subplots(figsize=(5, 3.5)); ax.scatter(X["weight"], y, s=10, alpha=0.6); ax.set_xlabel("Weight (lb)"); ax.set_ylabel("mpg"); plt.show()''',
        prep_md="Scaling inside the pipeline for linear models. Trees need none.",
        prep='''print(X.dtypes.to_dict())''',
        select_md="Linear regression, ridge (alpha chosen on validation), lasso and histogram gradient boosting. The metric is RMSE in miles per gallon.",
        select='''from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
rmse = lambda a, b: float(mean_squared_error(a, b) ** 0.5)  # noqa: E731
best_alpha = {}
for name, cls_, alphas in [("ridge", Ridge, [0.01, 0.1, 1, 10, 100]), ("lasso", Lasso, [0.001, 0.01, 0.1, 1])]:
    scores = {a: rmse(yva, make_model_pipeline(Xtr, cls_(alpha=a)).fit(Xtr, ytr).predict(Xva)) for a in alphas}
    best_alpha[name] = min(scores, key=scores.get)
cands = {"dummy": (DummyRegressor(), True), "linear": (LinearRegression(), True), "ridge": (Ridge(alpha=best_alpha["ridge"]), True), "lasso": (Lasso(alpha=best_alpha["lasso"]), True),
         "hist_gradient_boosting": (HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED), False)}
fitted = {n: make_model_pipeline(Xtr, m, scale=s).fit(Xtr, ytr) for n, (m, s) in cands.items()}
sel = pd.Series({n: rmse(yva, f.predict(Xva)) for n, f in fitted.items()}, name="validation RMSE")
display(sel.round(3).to_frame()); BEST = sel.drop("dummy").idxmin(); print("selected:", BEST, "| alphas:", best_alpha)''',
        train_md="Refit the selected model on training plus validation rows and run five fold cross validation.",
        train='''Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
m, s = cands[BEST]
final = make_model_pipeline(Xtv, clone(m), scale=s).fit(Xtv, ytv)
cv = cross_val_score(make_model_pipeline(Xtv, clone(m), scale=s), Xtv, ytv, cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_root_mean_squared_error")
print(f"CV RMSE {-cv.mean():.3f} +- {cv.std(ddof=1):.3f}")''',
        eval_md="Test set, scored once, with the dummy baseline.",
        eval='''m_test = evaluate_estimator(final, Xte, yte, "regression")
m_base = evaluate_estimator(fitted["dummy"], Xte, yte, "regression")
METRICS = {**{f"test_{k}": v for k, v in m_test.items()}, "baseline_RMSE": m_base["RMSE"], "cv_RMSE_mean": float(-cv.mean()), "cv_RMSE_std": float(cv.std(ddof=1))}
CONFIG = {"selected_model": BEST, "ridge_alpha": best_alpha["ridge"], "lasso_alpha": best_alpha["lasso"]}
EXTRA = {"validation_RMSE": sel.round(4).to_dict()}
ALGORITHM = BEST
display(pd.DataFrame({"model": m_test, "dummy": m_base}).round(3))''',
        compare_md="All models on the test rows, for comparison only.",
        compare='''display(pd.Series({n: rmse(yte, f.predict(Xte)) for n, f in fitted.items()}, name="test RMSE").round(3).to_frame())''',
        interp_md="Residuals by origin, and permutation importance.",
        interp='''res_ = yte - final.predict(Xte)
display(res_.groupby(Xte["origin"]).agg(["mean", "std", "count"]).round(2))
imp = permutation_importance(final, Xte, yte, scoring="neg_root_mean_squared_error", n_repeats=10, random_state=SEED)
display(pd.Series(imp.importances_mean, index=Xte.columns).sort_values(ascending=False).round(3).to_frame("RMSE increase"))''',
        limits="Cars from 1970 to 1982 only, 392 rows, 79 test rows. Weight, displacement and horsepower are strongly correlated, so single coefficients are unstable.",
        recs="Use the model to explain relative effects, not to predict modern cars. For a design tool collect current data and report prediction intervals.")

# ------------------------------------------------------------------ 6 diabetes interpretation
project(id="diabetes_interpretation", title="Interpreting a Small Medical Regression: Diabetes Progression", dataset="diabetes", task="regression",
        tagline="Which baseline measurements relate to progression, how stable is that answer, and what does a sparse model keep?",
        problem="Predict disease progression one year ahead and identify measurements that matter, with a stability check.",
        context="In a small clinical sample, an interpretable model is more useful than a black box. But feature rankings from 442 rows can change a lot between resamples, so we test that.",
        data_md=C.DS_DIAB, refs=["efron2004", "tibshirani1996", "zou2005"],
        refs_note="The data and the lasso path come from the LARS paper (Efron et al.). Lasso (Tibshirani) and elastic net (Zou and Hastie) are compared.",
        load='''bundle = load_dataset("diabetes")
X, y = bundle.X, bundle.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="regression", seed=SEED)
print(len(Xtr), len(Xva), len(Xte))''',
        eda_md="Correlation between the six serum measures.",
        eda='''display(X.corr().round(2))''',
        prep_md="Features are already centred and scaled by the data provider. No further processing.",
        prep='''print("feature means (should be about 0):", np.round(X.mean().values[:3], 3))''',
        select_md="Linear regression, lasso, elastic net and gradient boosting. Penalty strengths are chosen on the validation rows.",
        select='''from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
rmse = lambda a, b: float(mean_squared_error(a, b) ** 0.5)  # noqa: E731
la = min([0.01, 0.03, 0.1, 0.3, 1.0], key=lambda a: rmse(yva, Lasso(alpha=a).fit(Xtr, ytr).predict(Xva)))
ea = min([0.01, 0.03, 0.1, 0.3, 1.0], key=lambda a: rmse(yva, ElasticNet(alpha=a, l1_ratio=0.5).fit(Xtr, ytr).predict(Xva)))
cands = {"dummy": DummyRegressor(), "linear": LinearRegression(), "lasso": Lasso(alpha=la), "elastic_net": ElasticNet(alpha=ea, l1_ratio=0.5),
         "gradient_boosting": GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=SEED)}
fitted = {n: m.fit(Xtr, ytr) for n, m in cands.items()}
sel = pd.Series({n: rmse(yva, f.predict(Xva)) for n, f in fitted.items()}, name="validation RMSE")
display(sel.round(2).to_frame()); BEST = sel.drop("dummy").idxmin(); print("selected:", BEST)''',
        train_md="Refit on training plus validation rows.",
        train='''Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
final = clone(cands[BEST]).fit(Xtv, ytv)
cv = cross_val_score(clone(cands[BEST]), Xtv, ytv, cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_root_mean_squared_error")
print(f"CV RMSE {-cv.mean():.2f} +- {cv.std(ddof=1):.2f}")''',
        eval_md="Test set, scored once, with baseline.",
        eval='''m_test = evaluate_estimator(final, Xte, yte, "regression"); m_base = evaluate_estimator(fitted["dummy"], Xte, yte, "regression")
METRICS = {**{f"test_{k}": v for k, v in m_test.items()}, "baseline_RMSE": m_base["RMSE"], "cv_RMSE_mean": float(-cv.mean()), "cv_RMSE_std": float(cv.std(ddof=1))}
CONFIG = {"selected_model": BEST, "lasso_alpha": la, "elastic_net_alpha": ea}
EXTRA = {"validation_RMSE": sel.round(3).to_dict()}
ALGORITHM = BEST
display(pd.DataFrame({"model": m_test, "dummy": m_base}).round(3))''',
        compare_md="All models on the test rows, for comparison only.",
        compare='''display(pd.Series({n: rmse(yte, f.predict(Xte)) for n, f in fitted.items()}, name="test RMSE").round(2).to_frame())''',
        interp_md="Lasso coefficients and selection frequency across 200 bootstrap resamples of the training plus validation rows.",
        interp='''coef = pd.Series(Lasso(alpha=la).fit(Xtv, ytv).coef_, index=X.columns)
rng = np.random.default_rng(SEED); picked = np.zeros(X.shape[1])
for _ in range(200):
    i = rng.integers(0, len(Xtv), len(Xtv))
    picked += (Lasso(alpha=la).fit(Xtv.iloc[i], ytv.iloc[i]).coef_ != 0)
display(pd.DataFrame({"lasso_coef": coef, "selected_in_share_of_bootstraps": picked / 200}).round(2).sort_values("lasso_coef", key=abs, ascending=False))''',
        limits="442 patients, one study, features pre-scaled by the provider, and `sex` is a standardised number. Associations, not causes. Rankings of correlated serum measures are unstable, as the bootstrap table shows.",
        recs="Report sparse model results with selection frequencies, not a single ranking. Use these results to generate hypotheses and test them on new patient data.")

# ------------------------------------------------------------------ 7 digits
project(id="digit_recognition", title="Handwritten Digit Recognition with Error Analysis", dataset="digits", task="classification",
        tagline="Compare five classifiers, study the errors, and test robustness to noise.",
        problem="Recognise handwritten digits from 8 by 8 pixel images and find out which digits get confused and how robust the model is.",
        context="Form processing and cheque reading depend on reliable digit recognition. Overall accuracy hides which digits fail, and clean test images hide fragility to noise.",
        data_md=C.DS_DIGITS, refs=["cortes1995", "breiman2001", "cover1967"],
        refs_note="Support vector machines, random forests and nearest neighbours are compared with a neural network.",
        load='''bundle = load_dataset("digits")
X, y = bundle.X, bundle.y
Xtr, Xva, Xte, ytr, yva, yte = split_three_way(X, y, task="classification", seed=SEED)
print(len(Xtr), len(Xva), len(Xte))''',
        eda_md="Example images and class balance.",
        eda='''fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, i in zip(axes.ravel(), range(16)):
    ax.imshow(X.values[i].reshape(8, 8), cmap="gray_r"); ax.set_title(int(y.iloc[i]), fontsize=8); ax.axis("off")
plt.show(); print(y.value_counts().sort_index().to_dict())''',
        prep_md="Pixel values are divided by 16 inside the scaler step. No other preprocessing.",
        prep='''print("pixel range:", X.values.min(), "to", X.values.max())''',
        select_md="Logistic regression, k nearest neighbours, RBF SVM, random forest and a small neural network. Chosen by validation macro F1.",
        select='''from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
cands = {"logistic": LogisticRegression(max_iter=3000), "knn": KNeighborsClassifier(3), "svm_rbf": SVC(C=10, gamma="scale"), "random_forest": RandomForestClassifier(300, n_jobs=2, random_state=SEED),
         "mlp": MLPClassifier((64,), alpha=1e-3, max_iter=500, early_stopping=True, random_state=SEED)}
fitted = {n: make_model_pipeline(Xtr, m).fit(Xtr, ytr) for n, m in cands.items()}
sel = pd.Series({n: f1_score(yva, f.predict(Xva), average="macro") for n, f in fitted.items()}, name="validation macro F1")
display(sel.round(4).to_frame()); BEST = sel.idxmax(); print("selected:", BEST)''',
        train_md="Refit the selected model on training plus validation rows.",
        train='''Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
final = make_model_pipeline(Xtv, clone(cands[BEST])).fit(Xtv, ytv)
cv = cross_val_score(make_model_pipeline(Xtv, clone(cands[BEST])), Xtv, ytv, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro")
print(f"CV macro F1 {cv.mean():.4f} +- {cv.std(ddof=1):.4f}")''',
        eval_md="Test set, scored once, with the dummy baseline (most frequent class).",
        eval='''pred = final.predict(Xte)
base = DummyClassifier(strategy="prior").fit(Xtr, ytr)
METRICS = {"test_accuracy": float((pred == yte).mean()), "test_macro_F1": f1_score(yte, pred, average="macro"), "baseline_accuracy": float((base.predict(Xte) == yte).mean()),
           "cv_macro_F1_mean": float(cv.mean()), "cv_macro_F1_std": float(cv.std(ddof=1))}
CONFIG = {"selected_model": BEST}
EXTRA = {"validation_macro_F1": sel.round(4).to_dict(), "test_rows": int(len(yte))}
ALGORITHM = BEST''',
        compare_md="All five models on the test rows, for comparison only.",
        compare='''display(pd.Series({n: f1_score(yte, f.predict(Xte), average="macro") for n, f in fitted.items()}, name="test macro F1").round(4).to_frame())''',
        interp_md="Which digit pairs are confused, and a picture of the errors. Then robustness to Gaussian noise added to the test pixels.",
        interp='''from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yte, pred); np.fill_diagonal(cm, 0)
pairs = sorted([(cm[i, j], i, j) for i in range(10) for j in range(10) if cm[i, j] > 0], reverse=True)[:5]
print("most frequent confusions (count, true, predicted):", pairs)
wrong = np.where(pred != yte.values)[0][:8]
fig, axes = plt.subplots(1, max(1, len(wrong)), figsize=(10, 1.8))
for ax, i in zip(np.atleast_1d(axes), wrong):
    ax.imshow(Xte.values[i].reshape(8, 8), cmap="gray_r"); ax.set_title(f"{yte.values[i]}->{pred[i]}", fontsize=8); ax.axis("off")
plt.show()
rng = np.random.default_rng(SEED)
for sd in [0, 1, 2, 4]:
    noisy_X = np.clip(Xte.values + rng.normal(0, sd, Xte.shape), 0, 16)
    print(f"noise std {sd}: accuracy {(final.predict(pd.DataFrame(noisy_X, columns=Xte.columns)) == yte.values).mean():.3f}")''',
        limits="8 by 8 images from one collection, 1797 rows. High accuracy here does not transfer to different writers, scanners or larger images.",
        recs="Use SVM or a small network as a baseline for small digit images, always review the confusion pairs, and test on noisy and shifted images before any deployment.")

# ------------------------------------------------------------------ 8 wine
project(id="wine_authentication", title="Wine Cultivar Authentication with Abstention", dataset="wine", task="classification",
        tagline="Classify wine cultivar from chemistry, find the smallest useful feature set, and abstain when the model is unsure.",
        problem="Identify the cultivar of a wine from chemical analysis, and decide when a human should re-check.",
        context="Authentication checks are more useful when the system says 'I am not sure' on borderline samples. With 178 samples, estimates are noisy, so we use repeated cross validation.",
        data_md=C.DS_WINE, refs=["cover1967", "tibshirani1996", "breiman2001"],
        refs_note="Nearest neighbours, a lasso penalised logistic regression and a random forest are compared.",
        load='''bundle = load_dataset("wine")
X, y = bundle.X, bundle.y
print(X.shape, y.value_counts().sort_index().to_dict())''',
        eda_md="Class counts and the features with the largest class differences.",
        eda='''from sklearn.feature_selection import f_classif
f, _ = f_classif(X, y)
display(pd.Series(f, index=X.columns).sort_values(ascending=False).head(6).round(1).to_frame("ANOVA F"))''',
        prep_md="Scaling inside each pipeline. With 178 rows we use repeated stratified 5-fold cross validation (5 repeats) instead of a single split.",
        prep='''from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)''',
        select_md="Logistic regression with L1 penalty, k nearest neighbours and random forest. Selection by cross validated macro F1 on all rows is acceptable here because no test claim is made beyond the cross validated score, and nested scoring is added below.",
        select='''from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
cands = {"logistic_l1": LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=5000), "knn": KNeighborsClassifier(7), "random_forest": RandomForestClassifier(300, n_jobs=2, random_state=SEED)}
rows = []
for n, m in cands.items():
    s = cross_val_score(make_model_pipeline(X, m), X, y, cv=cv, scoring="f1_macro")
    rows.append({"model": n, "cv_macro_F1": s.mean(), "std": s.std(ddof=1)})
sel = pd.DataFrame(rows).set_index("model"); display(sel.round(4)); BEST = sel["cv_macro_F1"].idxmax(); print("best:", BEST)''',
        train_md="Fit the best model on all rows for the final description. Nested scoring: the model choice is repeated inside each outer fold so the reported score is not optimistic.",
        train='''from sklearn.model_selection import GridSearchCV
nested = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
    best_n, best_s = None, -1
    for n, m in cands.items():
        s = cross_val_score(make_model_pipeline(X.iloc[tr], m), X.iloc[tr], y.iloc[tr], cv=3, scoring="f1_macro").mean()
        if s > best_s: best_n, best_s = n, s
    f = make_model_pipeline(X.iloc[tr], clone(cands[best_n])).fit(X.iloc[tr], y.iloc[tr])
    from sklearn.metrics import f1_score
    nested.append(f1_score(y.iloc[te], f.predict(X.iloc[te]), average="macro"))
print("nested macro F1:", round(float(np.mean(nested)), 4), "+-", round(float(np.std(nested, ddof=1)), 4))
final = make_model_pipeline(X, clone(cands[BEST])).fit(X, y)''',
        eval_md="Nested score with a dummy baseline, and the accuracy and coverage trade-off when the model abstains below a confidence cut-off (cross validated probabilities).",
        eval='''from sklearn.model_selection import cross_val_predict
proba = cross_val_predict(make_model_pipeline(X, clone(cands["random_forest"])), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
conf, pred = proba.max(axis=1), proba.argmax(axis=1)
rows = []
for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
    keep = conf >= t
    rows.append({"confidence_cutoff": t, "coverage": keep.mean(), "accuracy_on_answered": (pred[keep] == y.values[keep]).mean() if keep.any() else np.nan})
abst = pd.DataFrame(rows).set_index("confidence_cutoff"); display(abst.round(3))
base = cross_val_score(make_model_pipeline(X, DummyClassifier(strategy="prior")), X, y, cv=cv, scoring="f1_macro").mean()
METRICS = {"nested_macro_F1_mean": float(np.mean(nested)), "nested_macro_F1_std": float(np.std(nested, ddof=1)), "baseline_macro_F1": float(base),
           "accuracy_at_cutoff_0.8": float(abst.loc[0.8, "accuracy_on_answered"]), "coverage_at_cutoff_0.8": float(abst.loc[0.8, "coverage"])}
CONFIG = {"selected_model": BEST, "cv": "repeated stratified 5x5", "abstention_model": "random_forest"}
EXTRA = {"cv_table": sel.round(4).reset_index().to_dict("records"), "abstention": abst.round(4).reset_index().to_dict("records")}
ALGORITHM = BEST''',
        compare_md="The cross validated table above compares the models. Below, the smallest feature set that keeps most of the score, using univariate ranking inside a pipeline.",
        compare='''from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
for k in [1, 2, 3, 5, 8, 13]:
    s = cross_val_score(make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(max_iter=2000)), X, y, cv=cv, scoring="f1_macro")
    print(f"top {k} features: macro F1 {s.mean():.3f} +- {s.std(ddof=1):.3f}")''',
        interp_md="L1 logistic regression keeps few features. Show which.",
        interp='''l1 = make_model_pipeline(X, cands["logistic_l1"]).fit(X, y)
coef = pd.DataFrame(l1.named_steps["model"].coef_, columns=X.columns)
print("features with a non-zero coefficient in any class:", list(coef.columns[(coef != 0).any()]))''',
        limits="178 samples from one region and three cultivars. The task is easy, so high scores say little about harder authentication problems. Confidence cut-offs come from cross validated probabilities on the same data.",
        recs="Use abstention to route uncertain samples to a lab check, and collect more varied samples before relying on the result.")

# ------------------------------------------------------------------ 9 anomaly
project(id="anomaly_screening", title="Anomaly Screening: Comparing Detectors on Rare Positives", dataset="breast_cancer", task="anomaly_detection",
        tagline="Compare three anomaly detectors over many random draws of the rare class, instead of trusting one split. A proxy for industrial anomaly detection, for which no open data could be downloaded.",
        problem="Rank rows by how unusual they are when only a small share of rows are abnormal.",
        context="Industrial and screening systems have few examples of failures. A detector is judged by how it ranks the rare cases. With few positives one random draw is unreliable, so we repeat the draw 30 times.",
        data_md=C.DS_BC + "\n\n**Construction.** Benign rows are normal. For each repetition we draw malignant rows equal to 5 percent of the benign count as anomalies. This is a **constructed benchmark**, not a real anomaly study.",
        refs=["liu2008", "breunig2000"], refs_note="Isolation Forest and Local Outlier Factor are compared with a one class SVM.",
        load='''bundle = load_dataset("breast_cancer")
X, y = bundle.X.values, bundle.y.values
print(X.shape, "benign:", int((y == 0).sum()), "malignant:", int((y == 1).sum()))''',
        eda_md="How different are the two groups on a few features.",
        eda='''names = bundle.X.columns
d = (X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)) / X.std(axis=0)
display(pd.Series(d, index=names).abs().sort_values(ascending=False).head(5).round(2).to_frame("standardised mean difference"))''',
        prep_md="Each repetition: draw anomalies, split 70/30 stratified, fit the scaler on the training part, fit detectors on the training part without labels, score the test part.",
        prep='''from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def draw(seed):
    rng = np.random.default_rng(seed)
    benign = np.where(y == 0)[0]
    mal = rng.choice(np.where(y == 1)[0], int(0.05 * len(benign)), replace=False)
    rows = np.sort(np.r_[benign, mal])
    Xs, ys = X[rows], y[rows]
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte
print("anomalies per draw:", int(0.05 * (y == 0).sum()))''',
        select_md="Detectors are Isolation Forest, Local Outlier Factor (novelty mode) and One-Class SVM. No labels are used for fitting, and no selection on test rows.",
        select='''from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
detectors = {"isolation_forest": lambda: IsolationForest(n_estimators=200, random_state=SEED),
             "lof": lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
             "one_class_svm": lambda: OneClassSVM(nu=0.05, gamma="scale")}
print(list(detectors))''',
        train_md="30 repetitions, each fitting every detector on training rows.",
        train='''from sklearn.metrics import roc_auc_score, average_precision_score
results = {n: {"roc": [], "pr": []} for n in detectors}
for rep in range(30):
    Xa, Xb, ya, yb = draw(rep)
    for n, make in detectors.items():
        det = make().fit(Xa)
        score = -det.score_samples(Xb) if n != "one_class_svm" else -det.decision_function(Xb)
        results[n]["roc"].append(roc_auc_score(yb, score)); results[n]["pr"].append(average_precision_score(yb, score))
print("done")''',
        eval_md="Mean and standard deviation over 30 draws, with the random ranking as the PR-AUC floor (about the share of anomalies, 0.044).",
        eval='''summary = pd.DataFrame({n: {"ROC_AUC_mean": np.mean(r["roc"]), "ROC_AUC_std": np.std(r["roc"], ddof=1), "PR_AUC_mean": np.mean(r["pr"]), "PR_AUC_std": np.std(r["pr"], ddof=1)} for n, r in results.items()}).T
display(summary.round(3))
METRICS = {f"{n}_{k}": float(v) for n, row in summary.iterrows() for k, v in row.items()}
METRICS["random_PR_AUC"] = float(0.05 / 1.05)
CONFIG = {"repetitions": 30, "anomaly_share": 0.05, "test_size": 0.3}
EXTRA = {"summary": summary.round(4).reset_index().to_dict("records")}
ALGORITHM = "IsolationForest, LOF, OneClassSVM"''',
        compare_md="Paired differences between detectors across the same 30 draws.",
        compare='''diff = np.array(results["isolation_forest"]["roc"]) - np.array(results["lof"]["roc"])
print("Isolation Forest minus LOF, ROC-AUC: mean", round(float(diff.mean()), 4), "| 95% interval", np.round(np.percentile(diff, [2.5, 97.5]), 4), "| IF better in", int((diff > 0).sum()), "of 30 draws")
fig, ax = plt.subplots(figsize=(5.5, 3.5)); ax.boxplot([results[n]["roc"] for n in detectors], tick_labels=list(detectors)); ax.set_ylabel("ROC-AUC over 30 draws"); save_figure(fig, PROJECT_ID, "boxplot"); plt.show()''',
        interp_md="Which features make flagged rows unusual, on one draw.",
        interp='''Xa, Xb, ya, yb = draw(0)
det = IsolationForest(n_estimators=200, random_state=SEED).fit(Xa)
top = np.argsort(det.score_samples(Xb))[:3]
for i in top:
    order = np.argsort(-np.abs(Xb[i]))[:3]
    print("row label", int(yb[i]), "| most unusual:", dict(zip(names[order], np.round(Xb[i][order], 1))))''',
        limits="Constructed anomalies from a real class, one dataset, 30 repetitions of a 30 percent split. Real industrial anomalies differ in kind and frequency.",
        recs="Compare detectors over repeated draws, set the alert threshold from the cost of reviewing a case, and validate on real labelled failures before deployment.")

# ------------------------------------------------------------------ 10 benchmark
project(id="tabular_model_benchmark", title="Tabular Model Benchmark Across Four Datasets", dataset="multiple", task="classification",
        tagline="Compare six model families on four datasets with repeated cross validation and report average ranks, not a single winner.",
        problem="Which model families work best across several small tabular problems?",
        context="Claims such as 'trees beat neural networks on tabular data' depend on datasets and tuning. We run a small, transparent test on four datasets with equal, light tuning and report spread. It cannot settle the claim of Grinsztajn et al., which used many more datasets.",
        data_md="Four datasets from the repository registry: `breast_cancer`, `wine`, `digits` and `telco_churn` (see their notebooks for licences). Each dataset is evaluated with 5-fold stratified cross validation repeated 2 times. The metric is macro F1 (PR-AUC for Telco is in the churn project).",
        refs=["grinsztajn2022", "breiman2001", "friedman2001", "cortes1995"], refs_note="Grinsztajn et al. motivate the question. The other papers define the model families.",
        load='''names = ["breast_cancer", "wine", "digits", "telco_churn"]
data = {n: load_dataset(n) for n in names}
print({n: data[n].X.shape for n in names})''',
        eda_md="Size, number of classes and missing values per dataset.",
        eda='''display(pd.DataFrame({n: {"rows": len(b.X), "features": b.X.shape[1], "classes": b.y.nunique(), "minority_share": float(b.y.value_counts(normalize=True).min()), "missing": int(b.X.isna().sum().sum())} for n, b in data.items()}).T)''',
        prep_md="Each model sits in a pipeline with imputation, encoding and (for models that need it) scaling, fitted inside each fold. Telco is limited to the first 4000 rows to keep the run short, which is stated here and in the metadata.",
        prep='''from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
from types import SimpleNamespace
data["telco_churn"] = SimpleNamespace(X=data["telco_churn"].X.iloc[:4000], y=data["telco_churn"].y.iloc[:4000])''',
        select_md="Six families with fixed, light settings (no per-dataset search): logistic regression, k nearest neighbours, RBF SVM, random forest, histogram gradient boosting and MLP.",
        select='''from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
models = {"logistic": (LogisticRegression(max_iter=2000), True), "knn": (KNeighborsClassifier(7), True), "svm_rbf": (SVC(C=3), True),
          "random_forest": (RandomForestClassifier(200, min_samples_leaf=2, n_jobs=2, random_state=SEED), False),
          "hist_gb": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=150, random_state=SEED), False),
          "mlp": (MLPClassifier((64,), alpha=1e-3, max_iter=400, early_stopping=True, random_state=SEED), True)}
print(list(models))''',
        train_md="Score every model on every dataset.",
        train='''import time
scores, spread = {}, {}
for dn, b in data.items():
    for mn, (m, sc) in models.items():
        t = time.time()
        s = cross_val_score(make_model_pipeline(b.X, clone(m), scale=sc), b.X, b.y, cv=cv, scoring="f1_macro")
        scores[(dn, mn)], spread[(dn, mn)] = s.mean(), s.std(ddof=1)
    print(dn, "done")''',
        eval_md="Macro F1 table, with the standard deviation over folds, and average rank per model (1 is best).",
        eval='''tab = pd.Series(scores).unstack().loc[names, list(models)]
sd = pd.Series(spread).unstack().loc[names, list(models)]
display(tab.round(3)); display(sd.round(3))
ranks = tab.rank(axis=1, ascending=False)
avg_rank = ranks.mean().sort_values()
display(avg_rank.round(2).to_frame("average rank"))
METRICS = {f"avg_rank_{k}": float(v) for k, v in avg_rank.items()}
METRICS["best_model_by_avg_rank_index"] = float(list(models).index(avg_rank.index[0]))
CONFIG = {"cv": "repeated stratified 5x2", "datasets": names, "telco_rows": 4000}
EXTRA = {"macro_F1": tab.round(4).reset_index().to_dict("records"), "best_model_by_avg_rank": avg_rank.index[0]}
ALGORITHM = "benchmark of six model families"''',
        compare_md="Plot the scores per dataset and mark differences smaller than one fold standard deviation.",
        compare='''fig, ax = plt.subplots(figsize=(9, 4)); tab.T.plot.bar(ax=ax, rot=20); ax.set_ylabel("Macro F1 (mean of 10 folds)"); ax.set_title("Model families across four datasets"); save_figure(fig, PROJECT_ID, "benchmark"); plt.show()
within = (tab.max(axis=1) - tab.sub(tab.max(axis=1), axis=0).abs().min(axis=1))
close = ((tab.max(axis=1).values[:, None] - tab.values) < sd.values).sum(axis=1)
print("models within one fold std of the best model, per dataset:", dict(zip(names, close)))''',
        interp_md="With four datasets the ranking is fragile. The count of models within one fold standard deviation of the best shows how often the winner is clear.",
        interp='''print("rank table (1 is best):"); display(ranks.astype(int))''',
        limits="Four small datasets, fixed light settings, one metric, Telco shortened to 4000 rows. Differences of a few thousandths are noise. This benchmark cannot confirm or refute large scale benchmark claims.",
        recs="Start with a regularised linear model and a tree ensemble as baselines on any new tabular problem, tune the best two, and decide with repeated cross validation and spread, not a single run.")

if __name__ == "__main__":
    ids = sys.argv[1:]
    for p in PROJECTS:
        if not ids or p["id"] in ids:
            render(p)
