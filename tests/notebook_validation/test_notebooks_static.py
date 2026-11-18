"""Static checks on every notebook. They do not execute the notebooks."""

from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = sorted(p for p in ROOT.glob("0*/**/*.ipynb") if ".ipynb_checkpoints" not in str(p))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_is_valid_and_executed_without_errors(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code, "no code cells"
    for cell in code:
        assert cell.execution_count is not None, "notebook was not executed"
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_has_no_absolute_paths_or_secrets(path):
    text = path.read_text()
    for bad in ["/home/claude", "/mnt/user-data", "ghp_", "AKIA"]:
        assert bad not in text, bad
