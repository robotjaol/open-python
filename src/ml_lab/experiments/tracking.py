"""Save experiment records."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from ml_lab.utils.reproducibility import environment_info, repo_root


@dataclass
class ExperimentRecord:
    notebook: str
    family: str
    algorithm: str
    dataset: str
    task: str
    seed: int
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    created: str = ""

    def save(self):
        self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        folder = repo_root() / "experiments" / "results"
        folder.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["environment"] = environment_info()
        path = folder / f"{self.notebook}.json"
        path.write_text(json.dumps(payload, indent=2, default=float))
        return path


def save_figure(fig, notebook, name):
    folder = repo_root() / "experiments" / "figures" / notebook
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def load_records():
    folder = Path(repo_root()) / "experiments" / "results"
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
