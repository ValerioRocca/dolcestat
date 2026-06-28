# dolcestat

*Sweet statistics* — a Python library implementing core machine learning algorithms from scratch.

This repository is designed for **educational use**. Every algorithm is implemented from first principles using only NumPy, with the goal of making the underlying mathematics transparent. It is a good companion to a course or textbook on statistical learning, optimization, or machine learning: read the source alongside the theory to see how the equations translate into code.

## Installation

```bash
git clone <repo-url>
cd dolcestat
pip install -e .
```

**Dependencies:** `numpy`, `polars`, `matplotlib`, `seaborn` (Python >= 3.10)

## Modules & Notebooks

| Module | Notebooks |
|---|---|
| `dolcestat.preprocessing` | — |
| `dolcestat.optimization` | [gradient_descent_sandbox](notebooks/gradient_descent_sandbox.ipynb) · [gradient_descent_sklearn](notebooks/gradient_descent_sklearn.ipynb) |
| `dolcestat.neighbors` | [knn_sklearn](notebooks/knn_sklearn.ipynb) |
