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

New here? Start with the [quickstart](notebooks/00_quickstart.ipynb), then follow
the notebooks in order — they build a full **preprocess → fit → evaluate** pipeline.

| Module | Notebooks |
|---|---|
| _start here_ | [00 · quickstart](notebooks/00_quickstart.ipynb) |
| `dolcestat.preprocessing` | [01 · preprocessing](notebooks/01_preprocessing.ipynb) |
| `dolcestat.linear_models` | [02 · linear regression](notebooks/02_linear_regression.ipynb) · [03 · logistic regression](notebooks/03_logistic_regression.ipynb) |
| `dolcestat.optimization` | [04 · optimization](notebooks/04_optimization.ipynb) |
| `dolcestat.neighbors` | [05 · k-nearest neighbours](notebooks/05_knn.ipynb) |
| `dolcestat.metrics` | [06 · metrics](notebooks/06_metrics.ipynb) |
| `dolcestat.trees` | [07 · decision trees & random forests](notebooks/07_trees.ipynb) |
