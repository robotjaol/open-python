"""Seeds, paths, and environment info."""

import json
import os
import platform
import random
from pathlib import Path

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    """Seed python and numpy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass
    return seed


def repo_root(start=None):
    """Find folder with pyproject."""
    folder = Path(start or Path.cwd()).resolve()
    for candidate in [folder, *folder.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "ml_lab").exists():
            return candidate
    return Path.cwd()


def package_versions(names=None):
    """Read installed versions."""
    from importlib import metadata

    names = names or [
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "xgboost", "lightgbm", "catboost", "umap-learn", "torch", "gymnasium",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info():
    """Describe the run environment."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }


def print_environment():
    info = environment_info()
    print("python", info["python"], "|", info["platform"])
    for name, version in info["packages"].items():
        if version:
            print(f"  {name}=={version}")


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path
