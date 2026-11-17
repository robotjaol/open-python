"""Execute notebooks from a clean kernel and log results."""

import json
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "experiments" / "metadata" / "execution_log.json"


def run_one(path, timeout=1500):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    start = time.time()
    status, error = "ok", ""
    try:
        client.execute()
    except Exception as exc:  # keep going, log the failure
        status, error = "failed", str(exc)[-1500:]
    nbformat.write(notebook, path)
    return {"notebook": str(path.relative_to(ROOT)), "status": status,
            "seconds": round(time.time() - start, 1), "error": error}


def main(targets):
    paths = []
    for target in targets:
        target = Path(target)
        paths += sorted(target.rglob("*.ipynb")) if target.is_dir() else [target]
    paths = [p.resolve() for p in paths if ".ipynb_checkpoints" not in str(p)]
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    for path in paths:
        result = run_one(path)
        log[result["notebook"]] = result
        print(result["status"], result["seconds"], result["notebook"], flush=True)
        if result["error"]:
            print(result["error"][-600:], flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        LOG.write_text(json.dumps(log, indent=2))
    return 0 if all(r["status"] == "ok" for r in log.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["."]))
