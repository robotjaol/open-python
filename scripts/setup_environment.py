"""Check Python, packages and folders. Print a short report."""

import importlib
import platform
import sys

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "joblib", "ml_lab"]
OPTIONAL = ["xgboost", "lightgbm", "catboost", "umap", "gymnasium", "torch", "fastapi", "nbclient"]


def version_of(name):
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def main():
    print("python", platform.python_version(), "on", platform.system())
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or newer is required")
        return 1
    bad = 0
    for name in REQUIRED:
        found = version_of(name)
        print(f"required {name:12s}", found or "MISSING")
        bad += found is None
    for name in OPTIONAL:
        print(f"optional {name:12s}", version_of(name) or "not installed")
    if bad:
        print("Install missing packages with: pip install -e '.[all,development]'")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
