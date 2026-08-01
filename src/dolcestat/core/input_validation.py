"""Stores input validation shared by the training loop."""

import numpy as np


def validate_X(X):
    """Checks the feature matrix."""
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values.")


def validate_y(y):
    """Checks the target vector, as it arrives at the DolceSet boundary (1-D)."""
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array (n_samples,).")
    if np.any(np.isinf(y)):
        raise ValueError("y contains infinite values.")


def validate_X_y(X, y):
    """Checks that features and targets describe the same rows."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be the same.")


def validate_tol(tol):
    """Checks the convergence tolerance, which may be None to disable it."""
    if tol is None:
        return
    if not isinstance(tol, (float, int)):
        raise ValueError("tol must be a float, an int, or None.")
    if tol <= 0:
        raise ValueError("tol must be positive.")


def validate_n_epochs(n_epochs):
    """Checks the epoch budget."""
    if not isinstance(n_epochs, int):
        raise ValueError("n_epochs must be an integer.")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")
