# dolcestat

*Sweet statistics* — a Python library implementing core machine learning algorithms from scratch.

## Educational Purpose

This repository is designed for **educational use**. Every algorithm is implemented from first principles using only NumPy, with the goal of making the underlying mathematics transparent. It is a good companion to a course or textbook on statistical learning, optimization, or machine learning: read the source alongside the theory to see how the equations translate into code.

## Installation

Clone the repository and install locally in editable mode:

```bash
git clone <repo-url>
cd dolcestat
pip install -e .
```

**Dependencies:** `numpy`, `polars`, `matplotlib`, `seaborn` (Python >= 3.10 required)

## Modules

### `dolcestat.preprocessing`

`DolceSet` is a lightweight data container that wraps a Polars DataFrame and separates features from the target column. `scalers` provides standardization (z-score) and min-max scaling, each returning both the scaled data and the fitted statistics.

### `dolcestat.optimization`

`GradientDescent` supports three training flavors (`batch`, `sgd`, `mini_batch`) and two momentum variants (`polyak`, `nesterov`), working with both MSE (linear regression) and BCE (logistic regression) losses. `NewtonMethod` is a second-order alternative that uses the explicit Hessian for faster convergence on small datasets.

### `dolcestat.neighbors`

`KNNClassifier` and `KNNRegressor` implement k-nearest neighbors with a configurable Minkowski distance metric and multiple weighting schemes (`uniform`, `distance`, `squared_distance`, `gaussian`).

## Usage

### Preprocessing

```python
import polars as pl
from dolcestat.preprocessing import DolceSet
from dolcestat.preprocessing.scalers import standardize

df = pl.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [0, 1, 0]})
ds = DolceSet(df, target="y")

scaled_X, stats = standardize(ds.X)
```

### Optimization

```python
from dolcestat.optimization.gradient_descent import GradientDescent

gd = GradientDescent(ds, alpha=0.01, n_iters=500, flavor="batch", loss="mse")
gd.fit()

weights = gd.get_weights()   # final weight vector
losses  = gd.get_loss()      # loss at each iteration
```

Visualize convergence:

```python
from dolcestat.optimization.visualizers import OptimizerVisualizer

viz = OptimizerVisualizer([gd])
viz.plot()
```

### Neighbors

```python
from dolcestat.neighbors import KNNClassifier

clf = KNNClassifier(k=5, weight="distance")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
```

## Notebooks

Interactive experiments and comparisons against scikit-learn are in the [`notebooks/`](notebooks/) directory:

- `gradient_descent_sandbox.ipynb` — algorithm development and exploration
- `gradient_descent_sklearn.ipynb` — validation against scikit-learn baselines
