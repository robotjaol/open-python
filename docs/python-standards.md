# Python standards

- Python 3.10 or newer. Format and lint with `ruff` (`make lint`). Line length 100.
- Names: `snake_case` for functions and variables, `CapWords` for classes.
- Reusable code lives in `src/ml_lab`, not copied between notebooks. A notebook calls the library and shows the result.
- Every random step takes a seed. Use `ml_lab.utils.set_seed`.
- Fit every preprocessing step on training rows only. Use a scikit-learn `Pipeline`.
- Notebooks must run top to bottom from a fresh kernel with no hidden state and no absolute paths.
- Do not commit data files larger than 5 MB, secrets, or model files you did not create.
- Public functions have a one line docstring that states what goes in and what comes out.
- New code gets a test in `tests/unit` when it contains logic.
