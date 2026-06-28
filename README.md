# dolcestat

*Sweet statistics* — a Python library implementing core machine learning algorithms from scratch.

Educational purpose: every algorithm is built from first principles using NumPy, so the mathematics stays visible. Read the source alongside a textbook to see how equations become code.

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
