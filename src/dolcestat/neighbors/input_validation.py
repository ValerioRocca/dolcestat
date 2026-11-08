"""Checks on the settings of the k-NN models."""

from dolcestat._validation import (  # noqa: F401  (shared data checks, re-exported)
    validate_class_labels,
    validate_is_fitted,
    validate_prediction_data,
    validate_training_data,
)


def validate_k(k):
    """Checks the number of neighbours (or clusters): a positive integer."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be positive.")


def validate_metric(metric, minkowski_p=None):
    """Checks the distance metric, and the exponent when it is 'minkowski'."""
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
    """Checks the neighbour weighting: uniform, distance, squared_distance, gaussian."""
    valid = {"uniform", "distance", "squared_distance", "gaussian"}
    if weights not in valid:
        raise ValueError(f"weights must be one of {sorted(valid)}, got '{weights}'.")
