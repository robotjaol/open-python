# Contributing

Thanks for helping. Small, checkable changes are easiest to review.

## Setup

```bash
pip install -e ".[all,development]"
make test && make lint
```

## Rules

1. **No invented sources.** Every paper has a DOI or arXiv id that you checked. If you cannot check it, mark it "not verified" in `tools/refs.py`.
2. **No invented results.** Numbers in text come from `experiments/results/*.json` or a table in the notebook.
3. **No leakage.** Fit preprocessing on training rows only. Choose settings on validation rows. Score test rows once.
4. **Reproducible notebooks.** Restart the kernel and Run All before committing. No absolute paths.
5. **Datasets.** Add a catalog entry with licence and source. Do not commit data you are not allowed to redistribute.
6. **Tests.** Logic in `src/ml_lab` needs a test in `tests/unit`.

## Notebooks are generated

Most notebooks come from specs in `tools/specs_*.py` and `tools/build_*.py`. Change the spec, rebuild, and re-execute, so the text and the outputs stay in sync. Do not edit generated notebooks by hand unless you also change the spec.

## Pull requests

Describe what changed, why, and how you checked it. Link the issue. Keep one topic per pull request.
