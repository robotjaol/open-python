"""Basic dataset checks."""

import pandas as pd


def validate_bundle(bundle):
    """Return list of problems."""
    problems = []
    frame = bundle.frame
    if bundle.target not in frame.columns:
        problems.append("target column missing")
    expected_rows = bundle.meta.get("rows")
    if expected_rows and len(frame) != expected_rows:
        problems.append(f"row count {len(frame)} differs from catalog {expected_rows}")
    if frame.duplicated().sum() > 0:
        problems.append(f"{int(frame.duplicated().sum())} duplicate rows")
    return problems


def missing_report(frame):
    counts = frame.isna().sum()
    report = pd.DataFrame({"missing": counts, "percent": (counts / len(frame) * 100).round(2)})
    return report[report["missing"] > 0]
