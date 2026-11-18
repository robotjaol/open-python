"""Notebook builder for unsupervised and RL specs."""

from nbformat.v4 import new_code_cell, new_markdown_cell
from nbgen import (  # noqa: F401
    SECTIONS,
    header,
    objectives_cell,
    papers_cell,
    problem_cell,
    write_notebook,
)

UN_SETUP = '''import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_lab.data import load_dataset
from ml_lab.evaluation import internal_scores, external_scores
from ml_lab.experiments import ExperimentRecord, save_figure
from ml_lab.utils import set_seed, print_environment
from ml_lab import visualization as viz

NOTEBOOK_ID = "__ID__"
SEED = 42
set_seed(SEED)
viz.apply_style()
print_environment()

bundle = load_dataset("__DATASET__")
X_raw = bundle.X
y_true = bundle.y          # evaluation only
print("shape:", X_raw.shape)
display(pd.Series(bundle.describe_source(), name="source").to_frame())'''

UN_EDA = '''print("Rows and columns:", X_raw.shape)
print("Missing values:", int(X_raw.isna().sum().sum()))
display(X_raw.describe().T.head(15).round(3))
fig, ax = plt.subplots(figsize=(5, 3))
pd.Series(y_true).value_counts().sort_index().plot.bar(ax=ax)
ax.set_xlabel("Reference label (never used for fitting)"); ax.set_ylabel("Rows"); ax.set_title("Reference label counts")
plt.show()
corr = X_raw.corr().abs().to_numpy(copy=True)
np.fill_diagonal(corr, 0)
print("Mean absolute correlation between features:", round(float(corr.mean()), 3))'''

UN_SAVE = '''record = ExperimentRecord(notebook=NOTEBOOK_ID, family="__FAMILY__", algorithm="__NAME__", dataset="__DATASET__",
                          task="__TASK__", seed=SEED, config=CONFIG, metrics=METRICS, extra=EXTRA)
print("saved:", record.save())
display(pd.Series(METRICS, name="metric").round(4).to_frame())'''


def fill_keys(text, **kw):
    for k, v in kw.items():
        text = text.replace(f"__{k}__", str(v))
    return text


def generic_notebook(spec, family_label, family_key, task, setup=UN_SETUP, eda=UN_EDA, eda_intro=None):
    kw = dict(ID=spec["id"], DATASET=spec["dataset"], NAME=spec["name"], FAMILY=family_key, TASK=task)
    cells = [new_markdown_cell(header(spec, family_label)),
             new_markdown_cell(objectives_cell(spec)),
             new_markdown_cell(problem_cell(spec)),
             new_markdown_cell(papers_cell(spec)),
             new_markdown_cell("## 4. Mathematical Foundation\n\n" + spec["math"]),
             new_markdown_cell("## 5. Dataset Introduction\n\n" + spec["dataset_md"]),
             new_code_cell(fill_keys(spec.get("setup", setup), **kw)),
             new_markdown_cell("## 6. Exploratory Data Analysis\n\n" + (eda_intro or
                               "We check size, missing values, label balance and feature relations. The reference labels are used for "
                               "evaluation only, never for fitting.") + " " + spec.get("eda_note", "")),
             new_code_cell(spec.get("eda") or eda),
             new_markdown_cell("## 7. Data Preprocessing\n\n" + spec["prep_md"]),
             new_code_cell(spec["prep"]),
             new_markdown_cell("## 8. Algorithm Implementation\n\n### 8.1 Concept\n\n" + spec["concept"] +
                               "\n\n### 8.2 Python implementation from scratch\n\n" + spec["scratch_md"])]
    if spec.get("scratch"):
        cells.append(new_code_cell(spec["scratch"]))
    cells += [new_markdown_cell("### 8.3 Library implementation\n\n" + spec["lib_md"] + "\n\n### 8.4 Important parameters\n\n" + spec["params"]),
              new_markdown_cell("## 9. Model Training\n\n" + spec["train_md"]),
              new_code_cell(spec["train"]),
              new_markdown_cell("## 10. Evaluation\n\n" + spec["eval_md"]),
              new_code_cell(spec["eval"] + "\n\n" + fill_keys(UN_SAVE, **kw)),
              new_markdown_cell("## 11. Visualization"),
              new_code_cell(spec["viz"]),
              new_markdown_cell("## 12. Interpretation\n\n" + spec["interpret_md"]),
              new_code_cell(spec["interpret"]),
              new_markdown_cell("## 13. Experiments and Challenges")]
    for i, exp in enumerate(spec["experiments"], 1):
        cells.append(new_markdown_cell(f"### Experiment {i}: {exp['title']}\n\n{exp['md']}"))
        cells.append(new_code_cell(exp["code"]))
    ex = "\n".join(f"{i}. {t}" for i, t in enumerate(spec["exercises"], 1))
    cells.append(new_markdown_cell(f"### Exercises for learners\n\n{ex}"))
    cells.append(new_markdown_cell(
        "## 14. Research Discussion\n\n" + spec["discussion"] +
        "\n\n**Dataset and method differences.** The papers above use other data and protocols. We do not claim that our numbers agree "
        "or disagree with them. This notebook only shows whether the behaviour they describe appears on our data.\n\n"
        "**Possible improvements.** " + spec["improvements"] + "\n\n"
        "**Research limitations.** One dataset and one seed for the headline numbers. " + spec.get("limits_extra", "")))
    take = "\n".join(f"- {t}" for t in spec["takeaways"])
    cells.append(new_markdown_cell(
        f"## 15. Summary and Further Reading\n\n**Key takeaways**\n\n{take}\n\n**Use cases.** {spec['use_cases']}\n\n"
        f"**Limitations.** {spec['limitations']}\n\n**Recommended next notebook.** {spec['next']}\n\n"
        "**Further reading.** See `references.md` in this folder."))
    return write_notebook(spec, cells, family_label)
