"""Stores input validation for the nearest-neighbour estimators."""


def validate_k(k):
    if not isinstance(k, int):
        raise ValueError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be positive.")


def validate_metric(metric, minkowski_p=None):
    valid = {"euclidean", "manhattan", "minkowski"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {sorted(valid)}, got '{metric}'.")
    if metric == "minkowski":
        if minkowski_p is None:
            raise ValueError("minkowski_p must be provided when metric is 'minkowski'.")
        if not isinstance(minkowski_p, (float, int)):
            raise ValueError("minkowski_p must be a float or int.")
        if minkowski_p <= 0:
            raise ValueError("minkowski_p must be positive.")


def validate_weights(weights):
    valid = {"uniform", "distance", "squared_distance", "gaussian"}
    if weights not in valid:
        raise ValueError(f"weights must be one of {sorted(valid)}, got '{weights}'.")
