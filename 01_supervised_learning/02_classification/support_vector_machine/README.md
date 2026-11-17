# Support Vector Machine: Reading Handwritten Digits

Find the maximum margin boundary and bend it with the RBF kernel.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`05_support_vector_machine.ipynb`](05_support_vector_machine.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- What the margin is and why maximising it helps
- How the kernel trick gives nonlinear boundaries
- How C and gamma control flexibility
- How support vectors define the model

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 05_support_vector_machine.ipynb
```

Results are saved to `experiments/results/support_vector_machine.json`.
