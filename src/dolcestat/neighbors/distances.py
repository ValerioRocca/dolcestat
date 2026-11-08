"""Pairwise distances between the rows of two data matrices."""

import numpy as np

from .input_validation import validate_metric

# Upper bound on the number of values in the temporary difference array built
# for one block of rows (about 32 MB of float64)
_BLOCK_VALUES = 4_000_000


def compute_distances_matrix(X_a, X_b, metric="euclidean", minkowski_p=None):
    """Returns the matrix of pairwise distances between rows of X_a and X_b."""
    validate_metric(metric, minkowski_p)

    # 1. Set correct Minkowski p
    #    (Minkowski is generalization of Manhattan and Euclidean)
    match metric:
        case "euclidean":
            minkowski_p = 2
        case "manhattan":
            minkowski_p = 1

    # 2. Distance from every row of X_a to every row of X_b, a block of rows at
    #    a time so the temporary (rows, n_b, n_features) array stays small.
    X_a = np.asarray(X_a, dtype=float)
    X_b = np.asarray(X_b, dtype=float)
    dist_matrix = np.empty((X_a.shape[0], X_b.shape[0]))
    block = max(1, _BLOCK_VALUES // max(1, X_b.shape[0] * X_a.shape[1]))
    for start in range(0, X_a.shape[0], block):
        abs_dist = np.abs(X_a[start : start + block, None, :] - X_b[None, :, :])
        dist_matrix[start : start + block] = (abs_dist**minkowski_p).sum(axis=2) ** (
            1 / minkowski_p
        )
    return dist_matrix
