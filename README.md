# dolcestat

*Sweet statistics* — a Python library implementing core machine learning algorithms from scratch.

This repository is designed for **educational use**. Every algorithm is implemented from first principles using only NumPy, with the goal of making the underlying mathematics transparent. It is a good companion to a course or textbook on statistical learning, optimization, or machine learning: read the source alongside the theory to see how the equations translate into code.

## Installation

```bash
pip install git+https://github.com/ValerioRocca/dolcestat.git
```

To run the notebooks, clone the repository and install it with Jupyter:

```bash
git clone https://github.com/ValerioRocca/dolcestat.git
cd dolcestat
pip install -e ".[notebooks]"
```

**Dependencies:** `numpy`, `polars`, `matplotlib`, `seaborn` (Python >= 3.10)

**Tests:** `pip install -e ".[test]"`, then `pytest`.

## Modules & Notebooks

New here? Start with the [quickstart](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/00_quickstart.ipynb), then follow
the notebooks in order — they build a full **preprocess → fit → evaluate** pipeline.

| Module | Notebooks |
|---|---|
| _start here_ | [00 · quickstart](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/00_quickstart.ipynb) |
| `dolcestat.preprocessing` | [01 · preprocessing](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/01_preprocessing.ipynb) |
| `dolcestat.linear_models` | [02 · linear regression](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/02_linear_regression.ipynb) · [03 · logistic regression](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/03_logistic_regression.ipynb) |
| `dolcestat.optimization` | [04 · optimization](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/04_optimization.ipynb) |
| `dolcestat.neighbors` | [05 · k-nearest neighbours](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/05_knn.ipynb) |
| `dolcestat.metrics` | [06 · metrics](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/06_metrics.ipynb) |
| `dolcestat.trees` | [07 · decision trees & random forests](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/07_trees.ipynb) · [11 · gradient boosting](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/11_gradient_boosting.ipynb) |
| `dolcestat.neural_networks` | [08 · rosenblatt perceptron](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/08_perceptron.ipynb) · [09 · feed-forward networks](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/09_FFNN.ipynb) |
| `dolcestat.clustering` | [10 · clustering](https://github.com/ValerioRocca/dolcestat/blob/main/notebooks/10_clustering.ipynb) |

## License

[MIT](LICENSE)
