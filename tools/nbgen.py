"""Build notebooks from specs. Notebook text is written once here, results come from running them."""

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).parent))
from refs import REFS, bibtex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SECTIONS = [
    "Learning Objectives", "Real-World Problem", "Research Paper", "Mathematical Foundation",
    "Dataset Introduction", "Exploratory Data Analysis", "Data Preprocessing",
    "Algorithm Implementation", "Model Training", "Evaluation", "Visualization",
    "Interpretation", "Experiments and Challenges", "Research Discussion",
    "Summary and Further Reading",
]


def fill(template, **values):
    for key, value in values.items():
        template = template.replace(f"__{key}__", str(value))
    return template


def toc():
    lines = ["## Table of Contents", ""]
    for number, name in enumerate(SECTIONS, 1):
        anchor = f"{number}.-" + name.replace(" ", "-")
        lines.append(f"{number}. [{name}](#{anchor})")
    return "\n".join(lines)


def header(spec, family_label):
    meta = (f"**Family:** {family_label} | **Level:** {spec['level']} | **Dataset:** `{spec['dataset']}` "
            f"| **Runs on:** CPU, about {spec.get('minutes', 1)} minute(s)")
    return f"# {spec['title']}\n\n{meta}\n\n{spec['tagline']}\n\n{toc()}"


def objectives_cell(spec):
    pre = "\n".join(f"- {p}" for p in spec["prereq"])
    obj = "\n".join(f"- {o}" for o in spec["objectives"])
    skills = "\n".join(f"- {s}" for s in spec["skills"])
    return (f"## 1. Learning Objectives\n\n**After this notebook you will understand:**\n\n{obj}\n\n"
            f"**Prerequisites:**\n\n{pre}\n\n**Skills you will have afterwards:**\n\n{skills}")


def problem_cell(spec):
    p = spec["problem"]
    return (f"## 2. Real-World Problem\n\n**Context.** {p['context']}\n\n**The problem.** {p['problem']}\n\n"
            f"**Why it matters.** {p['matters']}\n\n**Why machine learning.** {p['why_ml']}\n\n"
            f"**Why this algorithm.** {p['why_algo']}")


def papers_cell(spec):
    out = ["## 3. Research Paper", "",
           "Entries are grouped by type. Bibliographic details marked *verified* were checked against the "
           "publisher record (Crossref) or the arXiv abstract page on 2026-11-17. Entries marked "
           "*not verified* come from the author's knowledge and must be checked before citing.", ""]
    for key in spec["papers"]:
        r = REFS[key]
        link = r["url"] or ""
        out += [f"### {r['title']}", "",
                f"- **Type:** {r['kind']} paper",
                f"- **Authors:** {r['authors']}",
                f"- **Year and venue:** {r['year']}, {r['venue']}",
                f"- **DOI or URL:** {link}",
                f"- **Verification:** {r['verified']}",
                f"- **Research objective:** {r['objective']}",
                f"- **Methodology:** {r['method']}",
                f"- **Dataset:** {r['data']}",
                f"- **Main findings:** {r['findings']}",
                f"- **Limitations:** {r['limits']}", ""]
    out.append(f"**Relationship to this notebook.** {spec['paper_relation']} "
               "This notebook does not reproduce the experimental protocol of any listed paper. "
               "It uses a different dataset and split, so numbers are not comparable with the paper.")
    return "\n".join(out)


def references_md(spec):
    lines = [f"# References: {spec['title']}", "",
             "Grouped by type. See `research/source-register.md` for the verification method.", ""]
    for kind in ["foundational", "modern", "application", "benchmark"]:
        keys = [k for k in spec["papers"] if REFS[k]["kind"] == kind]
        if not keys:
            continue
        lines += [f"## {kind.capitalize()} papers", ""]
        for key in keys:
            r = REFS[key]
            lines += [f"- {r['authors']} ({r['year']}). *{r['title']}*. {r['venue']}. {r['url'] or ''}  ",
                      f"  Verification: {r['verified']}", ""]
    lines += ["## BibTeX", "", "```bibtex"] + [bibtex(k) + "\n" for k in spec["papers"]] + ["```", ""]
    return "\n".join(lines)


def readme_md(spec, family_label, notebook_name):
    return (f"# {spec['title']}\n\n{spec['tagline']}\n\n"
            f"| Item | Value |\n|---|---|\n| Family | {family_label} |\n| Level | {spec['level']} |\n"
            f"| Dataset | `{spec['dataset']}` (see `datasets/catalog`) |\n"
            f"| Notebook | [`{notebook_name}`]({notebook_name}) |\n| Bibliography | [`references.md`](references.md) |\n\n"
            f"## What you learn\n\n" + "\n".join(f"- {o}" for o in spec["objectives"]) +
            f"\n\n## Run it\n\n```bash\npip install -e \".[{spec.get('extras', 'supervised')},development]\"\n"
            f"jupyter lab {notebook_name}\n```\n\nResults are saved to `experiments/results/{spec['id']}.json`.\n")


def write_notebook(spec, cells, family_label, extra_files=True):
    folder = ROOT / spec["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{spec['num']}_{spec['id']}.ipynb"
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    nbformat.validate(nb)
    nbformat.write(nb, folder / name)
    if extra_files:
        (folder / "README.md").write_text(readme_md(spec, family_label, name))
        (folder / "references.md").write_text(references_md(spec))
    return folder / name


# ---------------------------------------------------------------- supervised

SETUP = '''import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
__IMPORTS__
from ml_lab.data import load_dataset, missing_report
from ml_lab.preprocessing import split_three_way, build_preprocessor, make_model_pipeline, feature_names
from ml_lab.evaluation import evaluate_estimator, compare_models
from ml_lab.experiments import ExperimentRecord, save_figure
from ml_lab.utils import set_seed, print_environment
from ml_lab import visualization as viz

NOTEBOOK_ID = "__ID__"
SEED = 42
set_seed(SEED)
viz.apply_style()
print_environment()'''

LOAD = '''bundle = load_dataset("__DATASET__")
data = bundle.frame
TASK = bundle.task
TARGET = bundle.target
print("task:", TASK, "| target:", TARGET, "| shape:", data.shape)
display(pd.Series(bundle.describe_source(), name="source").to_frame())
display(data.head())'''

EDA = '''print("Rows and columns:", data.shape)
display(data.dtypes.value_counts().rename("columns by dtype").to_frame())
display(data.describe().T.round(3).head(20))
missing = missing_report(data)
print("Columns with missing values:", len(missing))
display(missing)
print("Duplicate rows:", int(data.duplicated().sum()))'''

EDA_PLOTS = '''fig = viz.plot_target(data[TARGET], TASK)
plt.show()
fig = viz.plot_distributions(data.drop(columns=[TARGET]))
plt.show()
fig = viz.plot_correlation(data)
plt.show()
display(viz.outlier_table(data.drop(columns=[TARGET])).head(8))'''

PREP = '''__PREP__
X, y = data.drop(columns=[TARGET]), data[TARGET]
X_train, X_val, X_test, y_train, y_val, y_test = split_three_way(X, y, task=TASK, seed=SEED)
print("train / validation / test rows:", len(X_train), len(X_val), len(X_test))
if TASK == "classification":
    split_rates = pd.DataFrame({"train": y_train.value_counts(normalize=True),
                                "validation": y_val.value_counts(normalize=True),
                                "test": y_test.value_counts(normalize=True)}).round(3)
    display(split_rates)
prep = build_preprocessor(X_train, scale=__SCALE__, encode="__ENCODE__")
Xtr = prep.fit_transform(X_train)   # fit on train only
Xva, Xte = prep.transform(X_val), prep.transform(X_test)
print("transformed train shape:", Xtr.shape)'''

TRAIN = '''SELECT = "__SELECT__"
LOWER_IS_BETTER = __LOWER__
SCORING = "__SCORING__"
SWEEP_PARAM = "__SWEEP__"
SWEEP_VALUES = __VALUES__

def build_pipeline():
    return make_model_pipeline(X_train, __MODEL__, scale=__SCALE__, encode="__ENCODE__")

pipe = build_pipeline()
print(pipe.named_steps["model"])
rows = []
for value in SWEEP_VALUES:
    candidate = clone(pipe).set_params(**{SWEEP_PARAM: value}).fit(X_train, y_train)
    rows.append({SWEEP_PARAM: value, **evaluate_estimator(candidate, X_val, y_val, TASK)})
sweep = pd.DataFrame(rows).set_index(SWEEP_PARAM)
display(sweep[[c for c in [SELECT, "R2", "MAE", "Accuracy", "F1", "ROC_AUC", "PR_AUC"] if c in sweep.columns]].round(4))
best_value = sweep[SELECT].idxmin() if LOWER_IS_BETTER else sweep[SELECT].idxmax()
print("best", SWEEP_PARAM, "on validation:", best_value)
final_model = clone(pipe).set_params(**{SWEEP_PARAM: best_value})
X_trainval, y_trainval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
final_model.fit(X_trainval, y_trainval)   # refit before test
print("refit on train plus validation rows")'''

EVAL = '''baseline = DummyRegressor(strategy="mean") if TASK == "regression" else DummyClassifier(strategy="prior")
baseline = make_model_pipeline(X_train, baseline, scale=False).fit(X_trainval, y_trainval)
test_metrics = evaluate_estimator(final_model, X_test, y_test, TASK)
baseline_metrics = evaluate_estimator(baseline, X_test, y_test, TASK)
print("Test set, scored once after model choice")
display(compare_models({"baseline": baseline_metrics, "__NAME__": test_metrics}))
cv = cross_validate(final_model, X_trainval, y_trainval, cv=5, scoring=SCORING, n_jobs=1)
cv_scores = cv["test_score"]
print(f"5-fold CV {SCORING}: mean {cv_scores.mean():.4f}, std {cv_scores.std(ddof=1):.4f}")
record = ExperimentRecord(notebook=NOTEBOOK_ID, family="supervised", algorithm="__NAME__", dataset="__DATASET__",
                          task=TASK, seed=SEED, config={"param": SWEEP_PARAM, "value": str(best_value)},
                          metrics={k: float(v) for k, v in test_metrics.items()},
                          extra={"baseline": {k: float(v) for k, v in baseline_metrics.items()},
                                 "cv_mean": float(cv_scores.mean()), "cv_std": float(cv_scores.std(ddof=1)),
                                 "cv_scoring": SCORING, "select_metric": SELECT})
print("saved", record.save().name)'''

VIZ_REG = '''pred = final_model.predict(X_test)
fig = viz.plot_pred_vs_actual(y_test, pred, unit="__UNIT__")
save_figure(fig, NOTEBOOK_ID, "pred_vs_actual"); plt.show()
fig = viz.plot_residuals(y_test, pred, unit="__UNIT__")
save_figure(fig, NOTEBOOK_ID, "residuals"); plt.show()
fig = viz.plot_learning_curve(final_model, X_trainval, y_trainval, scoring=SCORING)
save_figure(fig, NOTEBOOK_ID, "learning_curve"); plt.show()'''

VIZ_CLS = '''pred = final_model.predict(X_test)
labels = bundle.target_names or None
fig = viz.plot_confusion(y_test, pred, labels=labels)
save_figure(fig, NOTEBOOK_ID, "confusion"); plt.show()
if y_test.nunique() == 2:
    fig = viz.plot_roc_pr(final_model, X_test, y_test)
    save_figure(fig, NOTEBOOK_ID, "roc_pr"); plt.show()
fig = viz.plot_learning_curve(final_model, X_trainval, y_trainval, scoring=SCORING)
save_figure(fig, NOTEBOOK_ID, "learning_curve"); plt.show()
# decision regions in two PCA axes
pca_view = PCA(n_components=2, random_state=SEED).fit(Xtr)
model_for_plot = clone(final_model.named_steps["model"])
fig = viz.plot_decision_boundary(lambda: clone(model_for_plot), pca_view.transform(Xtr), y_train.values,
                                 names=["PCA 1", "PCA 2"])
save_figure(fig, NOTEBOOK_ID, "decision_regions"); plt.show()'''

INTERP = '''__PERM__
errors = (y_test.values - final_model.predict(X_test)) if TASK == "regression" else (y_test.values != final_model.predict(X_test))
if TASK == "regression":
    worst = pd.DataFrame({"actual": y_test.values, "predicted": final_model.predict(X_test)}, index=y_test.index)
    worst["abs_error"] = (worst["actual"] - worst["predicted"]).abs()
    print("Largest test errors:")
    display(worst.sort_values("abs_error", ascending=False).head(5).round(3))
else:
    print("Misclassified test rows:", int(errors.sum()), "of", len(errors))
print()
print(f"Test {SELECT}: {test_metrics[SELECT]:.4f} | baseline {SELECT}: {baseline_metrics[SELECT]:.4f}")
print(f"Cross validation spread (std): {cv_scores.std(ddof=1):.4f}. A test gap larger than this spread is meaningful.")'''

PERM = '''perm = permutation_importance(final_model, X_test, y_test, scoring=SCORING, n_repeats=10, random_state=SEED)
importance = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
fig = viz.plot_importance(importance.index, importance.values, title="Permutation importance on the test set",
                          top=12, xlabel="Drop in " + SCORING + " when shuffled")
save_figure(fig, NOTEBOOK_ID, "importance"); plt.show()
print("Top features:", ", ".join(importance.index[:5]))'''

NO_PERM = '''print("Permutation importance is skipped: the features are raw pixels, so single pixel importance is not meaningful.")'''


def supervised_notebook(spec):
    task = spec["task"]
    values = dict(ID=spec["id"], DATASET=spec["dataset"], NAME=spec["name"],
                  IMPORTS=spec["imports"], SCALE=spec.get("scale", True), ENCODE=spec.get("encode", "onehot"),
                  PREP=spec.get("prep", ""), SELECT=spec["select"], LOWER=spec["select"] in ("RMSE", "MAE"),
                  SCORING=spec["scoring"], SWEEP=spec["sweep"][0], VALUES=repr(spec["sweep"][1]),
                  MODEL=spec["model"], UNIT=spec.get("unit", ""))
    values["PERM"] = fill(PERM if spec.get("perm", True) else NO_PERM)
    cells = [new_markdown_cell(header(spec, "Supervised learning")),
             new_markdown_cell(objectives_cell(spec)),
             new_markdown_cell(problem_cell(spec)),
             new_markdown_cell(papers_cell(spec)),
             new_markdown_cell("## 4. Mathematical Foundation\n\n" + spec["math"]),
             new_markdown_cell("## 5. Dataset Introduction\n\n" + spec["dataset_md"] +
                               "\n\nThe table printed below the next cell comes from the dataset registry in "
                               "`src/ml_lab/data/registry.py`. Licence terms belong to the data provider, not to this repository."),
             new_code_cell(fill(SETUP, **values)),
             new_code_cell(fill(LOAD, **values)),
             new_markdown_cell("## 6. Exploratory Data Analysis\n\nWe look at size, types, missing values, the target, "
                               "feature distributions, correlation and outliers before any modelling. " + spec.get("eda_note", "")),
             new_code_cell(EDA), new_code_cell(EDA_PLOTS),
             new_markdown_cell("## 7. Data Preprocessing\n\nRules used here:\n\n"
                               "- Split first, then fit imputers, scalers and encoders on the training rows only. This prevents data leakage.\n"
                               "- The test set is touched once, at the end.\n"
                               "- Order inside the pipeline: impute, scale or encode, then model.\n" + spec.get("prep_note", "")),
             new_code_cell(fill(PREP, **values)),
             new_markdown_cell("## 8. Algorithm Implementation\n\n### 8.1 Concept\n\n" + spec["concept"] +
                               "\n\n### 8.2 Python implementation from scratch\n\n" + spec["scratch_md"]),
             ]
    if spec.get("scratch"):
        cells.append(new_code_cell(spec["scratch"]))
    cells += [new_markdown_cell("### 8.3 Library implementation\n\n" + spec["lib_md"] + "\n\n### 8.4 Important parameters\n\n" + spec["params"]),
              new_markdown_cell("## 9. Model Training\n\nProcedure: fit on the training rows, choose one hyperparameter on the validation rows, "
                                "refit on training plus validation, then evaluate once on the test rows. The random seed is fixed (`SEED = 42`). "
                                + spec.get("train_note", "")),
              new_code_cell(fill(TRAIN, **values)),
              new_markdown_cell("## 10. Evaluation\n\n" + spec["metrics_md"] + "\n\nA dummy model gives the floor every real model must beat."),
              new_code_cell(fill(EVAL, **values)),
              new_markdown_cell("## 11. Visualization\n\nAll plots use the test rows unless the title says otherwise."),
              new_code_cell(fill(VIZ_REG if task == "regression" else VIZ_CLS, **values)),
              new_markdown_cell("## 12. Interpretation\n\n" + spec["interpret_md"]),
              new_code_cell(fill(INTERP, **values)),
              new_markdown_cell("## 13. Experiments and Challenges")]
    for index, exp in enumerate(spec["experiments"], 1):
        cells.append(new_markdown_cell(f"### Experiment {index}: {exp['title']}\n\n{exp['md']}"))
        cells.append(new_code_cell(exp["code"]))
    ex = "\n".join(f"{i}. {t}" for i, t in enumerate(spec["exercises"], 1))
    cells.append(new_markdown_cell(f"### Exercises for learners\n\n{ex}"))
    cells.append(new_markdown_cell("## 14. Research Discussion\n\n" + spec["discussion"] +
                                   "\n\n**Dataset and method differences.** The papers above use other datasets, other splits and often other implementations. "
                                   "We do not claim that our numbers agree or disagree with them. What this notebook can show is whether the "
                                   "behaviour described in the paper (for example the effect of a parameter) appears on our data.\n\n"
                                   "**Possible improvements.** " + spec["improvements"] + "\n\n"
                                   "**Research limitations.** One dataset, one split seed for the headline number, and small test sets for the UCI data. "
                                   "Cross validation spread is reported to show how unstable a single split can be."))
    take = "\n".join(f"- {t}" for t in spec["takeaways"])
    cells.append(new_markdown_cell(f"## 15. Summary and Further Reading\n\n**Key takeaways**\n\n{take}\n\n"
                                   f"**Use cases.** {spec['use_cases']}\n\n**Limitations.** {spec['limitations']}\n\n"
                                   f"**Recommended next notebook.** {spec['next']}\n\n"
                                   "**Further reading.** See `references.md` in this folder and the library documentation for the classes used above."))
    return write_notebook(spec, cells, "Supervised learning")


def build(specs, builder):
    paths = []
    for spec in specs:
        paths.append(builder(spec))
        print("built", paths[-1].relative_to(ROOT))
    return paths
