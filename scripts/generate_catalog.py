"""Rebuild research registers and dataset catalog from metadata."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "build_research.py"), run_name="__main__")
