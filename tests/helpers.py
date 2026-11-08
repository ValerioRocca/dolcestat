"""Small data generators shared by the tests."""

import numpy as np

from dolcestat.preprocessing import DolceSet


def make_regression(n=150, seed=0, n_features=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 + rng.normal(0, 0.3, n)
    return X, y


def make_binary(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    prob = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.4)))
    return X, (rng.uniform(size=n) < prob).astype(int)


def make_blobs(n_per=40, seed=0, centres=((0, 0), (2.5, 0), (1.2, 2.2)), scale=1.0):
    """Gaussian blobs, one class per centre; classes are labelled 0..K-1."""
    rng = np.random.default_rng(seed)
    X = np.vstack([np.array(c) + rng.normal(0, scale, (n_per, 2)) for c in centres])
    y = np.repeat(np.arange(len(centres)), n_per)
    order = rng.permutation(len(y))
    return X[order], y[order]


def to_sets(X, y, n_train):
    """A train and a test DolceSet from the first n_train rows and the rest."""
    return (
        DolceSet.from_arrays(X[:n_train], y[:n_train]),
        DolceSet.from_arrays(X[n_train:], y[n_train:]),
    )
