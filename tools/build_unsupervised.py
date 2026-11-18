"""Build unsupervised notebooks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import specs_unsup_a  # noqa: E402,F401
import specs_unsup_b  # noqa: E402,F401
from nbgen import build  # noqa: E402
from nbgen_generic import generic_notebook  # noqa: E402
from specs_unsup_a import UN  # noqa: E402


def builder(spec):
    return generic_notebook(spec, "Unsupervised learning", "unsupervised", spec.get("task", "clustering"))

ids = sys.argv[1:]
build([s for s in UN if not ids or s["id"] in ids], builder)
