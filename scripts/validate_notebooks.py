"""Run notebooks from a clean kernel. Exit 1 if any fails.

Usage: python scripts/validate_notebooks.py [path ...]
With no path, every notebook in folders 00 to 05 runs. This can take about an hour.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_notebooks  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_notebooks.main(sys.argv[1:]))
