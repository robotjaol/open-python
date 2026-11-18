# Security policy

## Reporting

Please report vulnerabilities privately through GitHub: Security tab, then "Report a vulnerability" (private vulnerability reporting must be enabled in repository settings). Do not open a public issue for a vulnerability.

## Scope

- Code in `src/ml_lab` and `06_model_deployment`.
- Supply chain: the `dependency-audit` workflow runs `pip-audit` weekly.

## Known limits

- Model files use `joblib`, which can execute code on load. Load only files you created. The checksum in the model card only detects changes after saving.
- The example API has no authentication. It is a teaching example.

## Supported versions

Only the latest commit on `main`.
