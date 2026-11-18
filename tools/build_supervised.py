"""Build supervised notebooks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import specs_regression  # noqa: E402,F401
from nbgen import build, supervised_notebook  # noqa: E402
from specs_supervised import SPECS  # noqa: E402

try:
    import specs_classification  # noqa: E402,F401
except ImportError:
    pass

only = sys.argv[1:]
specs = [s for s in SPECS if not only or s["id"] in only]
build(specs, supervised_notebook)
