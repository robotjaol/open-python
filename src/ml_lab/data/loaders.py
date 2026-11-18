"""Load real datasets into one bundle type."""

from dataclasses import dataclass, field

import pandas as pd

from ml_lab.utils.reproducibility import repo_root

from .download import download_file
from .registry import DATASETS


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    target: str
    task: str
    meta: dict = field(default_factory=dict)
    target_names: list = field(default_factory=list)

    @property
    def X(self):
        return self.frame.drop(columns=[self.target])

    @property
    def y(self):
        return self.frame[self.target]

    def describe_source(self):
        keys = ["title", "source", "url", "license", "doi", "rows", "features", "limits"]
        return {key: self.meta.get(key) for key in keys}


def _cache_dir():
    folder = repo_root() / "datasets" / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _load_auto_mpg():
    from mlxtend.data import autompg_data

    features, mpg = autompg_data()
    names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    frame = pd.DataFrame(features[:, :7], columns=names)
    frame["mpg"] = mpg
    return frame, []


def _load_sklearn(loader, target_name, label_names=None):
    bunch = loader(as_frame=True)
    frame = bunch.frame.copy()
    frame = frame.rename(columns={"target": target_name})
    names = list(bunch.target_names) if hasattr(bunch, "target_names") else []
    return frame, names


def _load_breast_cancer():
    from sklearn.datasets import load_breast_cancer

    frame, _ = _load_sklearn(load_breast_cancer, "malignant")
    frame["malignant"] = 1 - frame["malignant"]  # sklearn codes benign as 1
    return frame, ["benign", "malignant"]


def _load_mnist():
    from mlxtend.data import mnist_data

    pixels, digits = mnist_data()
    frame = pd.DataFrame(pixels, columns=[f"px{i}" for i in range(pixels.shape[1])])
    frame["digit"] = digits.astype(int)
    return frame, [str(i) for i in range(10)]


def _load_telco():
    meta = DATASETS["telco_churn"]
    path = download_file(meta["url"], _cache_dir() / "telco_churn.csv", meta["sha256"])
    frame = pd.read_csv(path)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame["Churn"] = (frame["Churn"] == "Yes").astype(int)
    return frame.drop(columns=["customerID"]), ["stayed", "churned"]


def load_dataset(name):
    """Return a DatasetBundle by name."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name}, choose from {sorted(DATASETS)}")
    meta = DATASETS[name]
    if name == "auto_mpg":
        frame, names = _load_auto_mpg()
    elif name == "breast_cancer":
        frame, names = _load_breast_cancer()
    elif name == "wine":
        from sklearn.datasets import load_wine

        frame, names = _load_sklearn(load_wine, "cultivar")
    elif name == "digits":
        from sklearn.datasets import load_digits

        frame, names = _load_sklearn(load_digits, "digit")
    elif name == "mnist_5000":
        frame, names = _load_mnist()
    elif name == "diabetes":
        from sklearn.datasets import load_diabetes

        frame, names = _load_sklearn(load_diabetes, "progression")
    elif name == "telco_churn":
        frame, names = _load_telco()
    frame = frame.reset_index(drop=True)
    return DatasetBundle(name, frame, meta["target"], meta["task"], dict(meta), names)


def list_datasets():
    return pd.DataFrame(
        [{"name": k, "task": v["task"], "rows": v["rows"], "license": v["license"]} for k, v in DATASETS.items()]
    )
