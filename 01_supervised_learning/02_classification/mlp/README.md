# Multi-Layer Perceptron: A Small Neural Network for Digits

Stack linear layers with nonlinearities and train them with backpropagation.

| Item | Value |
|---|---|
| Family | Supervised learning |
| Level | Intermediate |
| Dataset | `digits` (see `datasets/catalog`) |
| Notebook | [`12_mlp.ipynb`](12_mlp.ipynb) |
| Bibliography | [`references.md`](references.md) |

## What you learn

- How a hidden layer creates features
- How backpropagation computes gradients
- Why neural networks vary between random seeds
- How early stopping and L2 regularisation help

## Run it

```bash
pip install -e ".[supervised,development]"
jupyter lab 12_mlp.ipynb
```

Results are saved to `experiments/results/mlp.json`.
