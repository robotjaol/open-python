"""Generate research registers and dataset catalog from verified metadata."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from refs import REFS  # noqa: E402

from ml_lab.data.registry import DATASETS  # noqa: E402

FAMILY = {  # reference key -> algorithm families
    "supervised": ["breiman2001", "cortes1995", "chen2016", "friedman2001", "quinlan1986", "cover1967", "domingos1997", "tibshirani1996", "zou2005",
                   "efron2004", "freund1997", "hoerl1970", "wolberg1990", "smola2004", "grinsztajn2022", "ke2017", "prokhorenkova2018", "chen2016arxiv", "hastie2009"],
    "unsupervised": ["lloyd1982", "rousseeuw1987", "ward1963", "schubert2017", "campello2013", "dempster1977", "vonluxburg2007", "pearson1901", "scholkopf1998",
                     "liu2008", "breunig2000", "agrawal1993", "comaniciu2002", "mcinnes2018", "mcinnes2018arxiv", "vandermaaten2008"],
    "reinforcement_learning": ["sutton1988", "watkins1992", "mnih2015", "williams1992", "konda2003", "vanhasselt2016", "schulman2017", "mnih2016", "haarnoja2018",
                               "lillicrap2016", "fujimoto2018", "levine2020", "kumar2020", "towers2024", "henderson2018", "agarwal2021", "sutton2018"],
}
used = {k for v in FAMILY.values() for k in v}
missing = set(REFS) - used
assert not missing, missing


def cite(r):
    link = f"https://doi.org/{r['doi']}" if r.get("doi") else (r.get("url") or "no link")
    return f"{r['authors']} ({r['year']}). {r['title']}. {r['venue']}. {link}"


rows = []
for fam, keys in FAMILY.items():
    for k in keys:
        r = REFS[k]
        rows.append(f"| `{k}` | {fam} | {cite(r)} | {r['kind']} | {r['verified']} |")
(ROOT / "research" / "source-register.md").write_text(
    "# Source register\n\nEvery paper cited in the notebooks is listed here with how it was checked. "
    "Titles, authors, year and venue of entries marked Crossref or arXiv were compared with that service on 2026-11-17. "
    "Entries marked *not verified in build environment* were written from the maintainer's knowledge and must be checked before you cite them. "
    "Verification checks that a paper exists and that the bibliographic data match. It does not check that the notebook summary of a paper is complete.\n\n"
    "| Key | Family | Reference | Kind | Verification |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")

idx = ["# Algorithm paper index\n\nOne original paper per algorithm where one exists. Keys point to `source-register.md`.\n"]
ALG = {
    "supervised": [("Linear and ridge regression", "hoerl1970"), ("Lasso", "tibshirani1996"), ("Elastic net", "zou2005"), ("Least angle regression", "efron2004"),
                   ("Support vector machine", "cortes1995"), ("Support vector regression", "smola2004"), ("Decision tree", "quinlan1986"), ("Random forest", "breiman2001"),
                   ("Gradient boosting", "friedman2001"), ("XGBoost", "chen2016"), ("LightGBM", "ke2017"), ("CatBoost", "prokhorenkova2018"), ("AdaBoost", "freund1997"),
                   ("k nearest neighbours", "cover1967"), ("Naive Bayes", "domingos1997")],
    "unsupervised": [("k-means", "lloyd1982"), ("Hierarchical (Ward)", "ward1963"), ("DBSCAN", "schubert2017"), ("HDBSCAN", "campello2013"), ("Gaussian mixture (EM)", "dempster1977"),
                     ("Spectral clustering", "vonluxburg2007"), ("PCA", "pearson1901"), ("t-SNE", "vandermaaten2008"), ("UMAP", "mcinnes2018"), ("Isolation Forest", "liu2008")],
    "reinforcement_learning": [("Temporal difference", "sutton1988"), ("Q-learning", "watkins1992"), ("DQN", "mnih2015"), ("Double DQN", "vanhasselt2016"),
                               ("REINFORCE", "williams1992"), ("Actor-critic", "konda2003"), ("PPO", "schulman2017")],
}
for fam, items in ALG.items():
    idx.append(f"\n## {fam.replace('_', ' ').title()}\n\n| Algorithm | Key | Year | Verification |\n|---|---|---|---|")
    for name, k in items:
        idx.append(f"| {name} | `{k}` | {REFS[k]['year']} | {REFS[k]['verified']} |")
idx.append("\nMonte Carlo control, SARSA and dynamic programming are taught from the textbook `sutton2018`, which is not verified in the build environment. No original paper is claimed for them.\n")
(ROOT / "research" / "algorithm-paper-index.md").write_text("\n".join(idx) + "\n")

# dataset catalog
by_task = {"supervised": ["classification", "regression"]}
cat = {"supervised": [], "unsupervised": [], "reinforcement_learning": []}
for name, d in DATASETS.items():
    entry = {"name": name, **d}
    if d["task"] in by_task["supervised"]:
        cat["supervised"].append(entry)
cat["unsupervised"] = [dict(name=n, **DATASETS[n], role="unlabelled use; reference labels used only for external checks") for n in ["auto_mpg", "wine", "digits", "breast_cancer", "telco_churn"] if n in DATASETS]
cat["reinforcement_learning"] = [
    {"name": "FrozenLake-v1", "source": "Gymnasium toy_text", "license": "MIT (Gymnasium)", "type": "simulated environment", "access": "pip install gymnasium", "limits": "4x4 map, tabular only"},
    {"name": "Taxi-v3", "source": "Gymnasium toy_text", "license": "MIT (Gymnasium)", "type": "simulated environment", "access": "pip install gymnasium", "limits": "500 states, discrete"},
    {"name": "CliffWalking-v1", "source": "Gymnasium toy_text", "license": "MIT (Gymnasium)", "type": "simulated environment", "access": "pip install gymnasium", "limits": "tabular only"},
    {"name": "CartPole-v1", "source": "Gymnasium classic_control", "license": "MIT (Gymnasium)", "type": "simulated environment", "access": "pip install gymnasium", "limits": "4 state values, 2 actions, solved by simple rules"},
]
for fam, items in cat.items():
    (ROOT / "datasets" / "catalog" / f"{fam}.json").write_text(json.dumps(items, indent=2) + "\n")
print({k: len(v) for k, v in cat.items()})

# ---- comparisons generated from saved results
res = [json.loads(p.read_text()) for p in sorted((ROOT / "experiments" / "results").glob("*.json"))]
out = ROOT / "research" / "algorithm-comparisons"
out.mkdir(parents=True, exist_ok=True)
HEAD = "Generated by `tools/build_research.py` from `experiments/results/*.json`. Numbers are single runs with the seeds stored in each file. Different datasets are different rows, so do not compare across rows.\n\n"


def table(rows, cols):
    lines = ["| " + " | ".join(["algorithm", "dataset"] + cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for r in rows:
        lines.append("| " + " | ".join([r["algorithm"], r["dataset"]] + [f"{r['metrics'][c]:.3f}" if c in r["metrics"] else "n/a" for c in cols]) + " |")
    return "\n".join(lines)


fam = lambda f, t: sorted([r for r in res if r["family"] == f and r["task"] == t], key=lambda r: (r["dataset"], r["algorithm"]))  # noqa: E731
(out / "supervised-classification.md").write_text("# Supervised classification, test results\n\n" + HEAD + table(fam("supervised", "classification"), ["Accuracy", "F1", "ROC_AUC", "PR_AUC"]) + "\n")
(out / "supervised-regression.md").write_text("# Supervised regression, test results\n\n" + HEAD + table(fam("supervised", "regression"), ["RMSE", "MAE", "R2"]) + "\n")
unsup = sorted([r for r in res if r["family"] == "unsupervised"], key=lambda r: (r["dataset"], r["algorithm"]))
(out / "unsupervised.md").write_text("# Unsupervised results\n\n" + HEAD + "ARI and NMI use reference labels only as an outside check. They show n/a where the notebook did not save them.\n\n" + table(unsup, ["Silhouette", "ARI", "NMI"]) + "\n")
rl = sorted([r for r in res if r["family"] == "reinforcement_learning"], key=lambda r: r["algorithm"])
(out / "reinforcement-learning.md").write_text("# Reinforcement learning results\n\n" + HEAD + "Metric names differ per notebook, so open the JSON for details. Shown here: the first four metrics saved.\n\n" +
    "\n".join(f"- **{r['algorithm']}** on `{r['dataset']}`: " + ", ".join(f"{k} = {v:.3g}" for k, v in list(r["metrics"].items())[:4] if isinstance(v, (int, float))) for r in rl) + "\n")

(ROOT / "research" / "dataset-catalog" / "README.md").write_text(
    "# Dataset catalog\n\nMachine readable entries live in `datasets/catalog/*.json`. This page explains what is and is not in the repository.\n\n"
    "## Substitutions\n\nThe build environment could not reach UCI, OpenML, Kaggle, Hugging Face or DOI hosts. Datasets that ship inside installed Python packages (scikit-learn, mlxtend) were used, "
    "and the Telco churn file was downloaded from a GitHub raw URL and checked with SHA-256. Their original sources and licences are recorded in each catalog entry.\n\n"
    "## Not available\n\nThe directive lists predictive maintenance, credit risk, fraud detection, energy forecasting and recommendation projects. Their usual open datasets sit on blocked hosts, "
    "so those projects are **Blocked**, not simulated. See `05_end_to_end_projects/BLOCKED.md`.\n")
(ROOT / "research" / "literature-review").mkdir(exist_ok=True)
(ROOT / "research" / "literature-review" / "README.md").write_text(
    "# Literature review (scope and status)\n\nThis is a reading map, not a systematic review. No search protocol, database query log or inclusion criteria were run, so do not cite it as a systematic review.\n\n"
    "## Themes covered by verified sources\n\n"
    "- **Tabular supervised learning.** Tree ensembles and boosting (`breiman2001`, `friedman2001`, `chen2016`, `prokhorenkova2018`), kernel methods (`cortes1995`), penalised linear models (`hoerl1970`, `tibshirani1996`, `zou2005`, `efron2004`). `grinsztajn2022` reports that tree based models stayed ahead of deep networks on a benchmark of medium sized tabular datasets. The repository benchmark has four small datasets, so it neither confirms nor refutes that result.\n"
    "- **Unsupervised learning.** Partitioning, density and model based clustering (`lloyd1982`, `schubert2017`, `campello2013`, `dempster1977`, `vonluxburg2007`), projection (`pearson1901`, `mcinnes2018`) and anomaly detection (`liu2008`, `breunig2000`).\n"
    "- **Reinforcement learning.** Temporal difference and Q-learning (`sutton1988`, `watkins1992`), deep value methods (`mnih2015`, `vanhasselt2016`), policy gradient (`williams1992`, `konda2003`, `schulman2017`), and reporting practice (`henderson2018`, `agarwal2021`).\n\n"
    "## Not covered\n\nSoft actor-critic, DDPG, TD3 and offline RL have verified sources in the register but **no notebooks**. Time series, causal inference, fairness and large language model topics are not covered.\n")
(ROOT / "research" / "research-gaps").mkdir(exist_ok=True)
(ROOT / "research" / "research-gaps" / "README.md").write_text(
    "# Research gaps in this repository\n\nThese are gaps in the repository, not claims about the field.\n\n"
    "| Gap | Why it matters | How to close it |\n|---|---|---|\n"
    "| Required real world datasets (maintenance, credit, fraud, energy, recommendation) are missing | Projects cannot show domain specific issues such as time based leakage or extreme imbalance | Run `scripts/download_datasets.py` on a machine with access, add loaders, build the project notebooks |\n"
    "| RL runs use 3 seeds and small budgets | `agarwal2021` and `henderson2018` show that few seeds give wide uncertainty | Use 10 or more seeds, report confidence intervals, longer training |\n"
    "| Benchmark has four datasets | Cannot test claims from `grinsztajn2022` | Extend to 20 or more datasets with nested tuning |\n"
    "| Anomaly detection data are constructed from a labelled class | Real faults differ in type and rate | Use an open industrial anomaly dataset |\n"
    "| Unverified references (`ke2017`, `vandermaaten2008`, `sutton2018`, `hastie2009`) | Bibliographic details may contain errors | Check against the publisher page |\n"
    "| Advanced ML and deployment folders hold small examples | They are not production patterns | Add authentication, load tests, monitoring in production |\n")
print("research files written")
