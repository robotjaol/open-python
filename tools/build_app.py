"""Build the static preview app into app/dist from saved results and notebooks."""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DIST = ROOT / "app" / "src", ROOT / "app" / "dist"
FAMILIES = {"00_fundamentals": "Fundamentals", "01_supervised_learning": "Supervised", "02_unsupervised_learning": "Unsupervised",
            "03_reinforcement_learning": "Reinforcement learning", "04_advanced_machine_learning": "Advanced", "05_end_to_end_projects": "Projects"}
HEADLINE = ["test_PR_AUC", "ROC_AUC", "Accuracy", "test_macro_F1", "test_accuracy", "RMSE", "test_RMSE", "Silhouette", "eval_mean_return", "clean_ppo_mean", "nested_macro_F1_mean",
            "isolation_forest_ROC_AUC_mean", "avg_rank_logistic", "test_R2", "R2"]


BY_ID = {"pca": "variance_explained", "temporal_difference": "best_final_rmse", "sarsa": "online_mean_sarsa", "double_dqn": "eval_mean_return_double_dqn",
         "umap": "trustworthiness_heldout", "tsne": "trustworthiness", "dbscan": "Silhouette", "hdbscan": "Silhouette", "dynamic_programming": "exact_success_optimal_4x4"}


def headline(metrics, nid=None):
    if nid in BY_ID and BY_ID[nid] in metrics:
        return {"name": BY_ID[nid], "value": round(float(metrics[BY_ID[nid]]), 4)}
    for key in HEADLINE:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return {"name": key, "value": round(float(metrics[key]), 4)}
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            return {"name": key, "value": round(float(val), 4)}
    return None


def main(with_notebooks=True):
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    for f in SRC.iterdir():
        shutil.copy(f, DIST / f.name)
    results = {p.stem: json.loads(p.read_text()) for p in (ROOT / "experiments" / "results").glob("*.json")}
    exporter = None
    if with_notebooks:
        import nbformat
        from nbconvert import HTMLExporter
        exporter = HTMLExporter(template_name="lab")
        (DIST / "notebooks").mkdir()
    items = []
    for folder, family in FAMILIES.items():
        for path in sorted((ROOT / folder).rglob("*.ipynb")):
            if ".ipynb_checkpoints" in str(path):
                continue
            nid = path.parent.name
            rec = results.get(nid)
            item = {"id": nid, "family": family, "path": str(path.relative_to(ROOT)), "title": nid.replace("_", " ").title().replace(" Ml", " ML").replace("Pca", "PCA").replace("Tsne", "t-SNE").replace("Umap", "UMAP").replace("Dbscan", "DBSCAN").replace("Hdbscan", "HDBSCAN").replace("Dqn", "DQN").replace("Ppo", "PPO").replace("Xgboost", "XGBoost").replace("Lightgbm", "LightGBM").replace("Catboost", "CatBoost").replace("Mlp", "MLP")}
            if rec:
                item.update(algorithm=rec["algorithm"], dataset=rec["dataset"], task=rec["task"], seed=rec["seed"], created=rec["created"][:10],
                            headline=headline(rec["metrics"], nid), metrics={k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec["metrics"].items() if isinstance(v, (int, float))})
            else:
                item.update(dataset="synthetic or none (concept demo)", task="concept")
            figs = ROOT / "experiments" / "figures" / nid
            if figs.is_dir():
                (DIST / "figures" / nid).mkdir(parents=True, exist_ok=True)
                item["figures"] = []
                for fig in sorted(figs.glob("*.png")):
                    shutil.copy(fig, DIST / "figures" / nid / fig.name)
                    item["figures"].append(f"figures/{nid}/{fig.name}")
            if exporter:
                import nbformat
                body, _ = exporter.from_notebook_node(nbformat.read(path, as_version=4))
                rel = f"notebooks/{folder}__{nid}.html"
                (DIST / rel).write_text(body)
                item["preview"] = rel
            items.append(item)
    status = json.loads((ROOT / "app" / "src" / "status.json").read_text())
    (DIST / "data.json").write_text(json.dumps({"items": items, "status": status}, indent=1))
    (DIST / ".nojekyll").write_text("")
    print("items", len(items), "with results", sum(1 for i in items if "metrics" in i))


if __name__ == "__main__":
    main(with_notebooks="--no-notebooks" not in sys.argv)
