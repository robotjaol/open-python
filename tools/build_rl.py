"""Build RL notebooks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import specs_rl_a  # noqa: E402,F401
import specs_rl_b  # noqa: E402,F401
from nbgen import build  # noqa: E402
from nbgen_generic import generic_notebook  # noqa: E402
from specs_rl_a import RL  # noqa: E402

RL_EDA_NOTE = "Run the cell below to inspect the environment."

def builder(spec):
    return generic_notebook(spec, "Reinforcement learning", "reinforcement_learning", "control",
                            eda_intro="There is no dataset. We inspect the environment: observation and action spaces, rewards and a random policy baseline.")

ids = sys.argv[1:]
build([s for s in RL if not ids or s["id"] in ids], builder)
