import numpy as np

from .input_validation import validate_metric


def minkowski(a, b, p):
    abs_dist = np.abs(a - b)
    return np.sum(abs_dist**p) ** (1 / p)


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

    # 2. Iterate over each combination of rows
    dist_matrix = []
    for x_a in X_a:
        row = []
        for x_b in X_b:
            distance = minkowski(x_a, x_b, minkowski_p)
            row.append(distance)
        dist_matrix.append(row)
    return np.array(dist_matrix)
