"""Settings, fitting and neighbour search shared by the k-NN models."""

from abc import ABC

import numpy as np

from .distances import compute_distances_matrix
from .input_validation import (
    validate_is_fitted,
    validate_k,
    validate_metric,
    validate_prediction_data,
    validate_training_data,
    validate_weights,
)


class BaseNeighbors(ABC):
    """Shared base for the nearest-neighbours estimators.

    Built from its settings alone. ``fit(data)`` just stores the labelled data
    (there is nothing to learn) and returns the estimator; the work happens at
    ``predict`` time.

    Parameters
    ----------
    k : int, default 5
        Number of neighbours that vote / are averaged.
    metric : {"euclidean", "manhattan", "minkowski"}, default "euclidean"
    minkowski_p : float, optional
        Exponent of the Minkowski distance; required when metric is "minkowski".
    weights : {"uniform", "distance", "squared_distance", "gaussian"}
        How much each neighbour counts: equally, or the closer the more.
    """

    def __init__(self, k=5, *, metric="euclidean", minkowski_p=None, weights="uniform"):
        validate_k(k)
        validate_metric(metric, minkowski_p)
        validate_weights(weights)
        self.k = k
        self.metric = metric
        self.minkowski_p = minkowski_p
        self.weights = weights
        self.lab_X = None
        self.lab_y = None
        self.is_fitted = False

    def fit(self, data):
        """Stores the labelled data (a DolceSet with a target); returns self."""
        validate_training_data(data)
        self.lab_X = data.X
        self.lab_y = data.y
        self.n_features = data.X.shape[1]
        self.is_fitted = True
        return self

    def _check_ready_to_predict(self, data):
        validate_is_fitted(self.is_fitted, type(self).__name__)
        validate_prediction_data(data, self.n_features)

    def _compute_distances_matrix(self, X_lab, X_unlab):
        return compute_distances_matrix(X_unlab, X_lab, self.metric, self.minkowski_p)

    def _get_knn_idxs(self, distances_matrix):
        """Returns the indexes of the K nearest neighbours for each unit to predict"""
        if self.k > distances_matrix.shape[1]:
            raise ValueError(
                f"k ({self.k}) cannot be greater than the number of labeled samples ({distances_matrix.shape[1]})."
            )
        return np.argsort(distances_matrix, axis=1, kind="stable")[:, : self.k]

    def _get_knn_y(self, y_lab, knn_idxs):
        """Returns the Y values of the K nearest neighbours for each unit to predict"""
        return y_lab[knn_idxs]

    def _get_knn_weights(self, distances_matrix, knn_idxs):
        """Returns a weight matrix for the K nearest neighbours"""
        knn_distances = np.take_along_axis(distances_matrix, knn_idxs, axis=1)
        match self.weights:
            case "uniform":
                return np.ones_like(knn_distances)
            case "distance" | "squared_distance":
                power = 1 if self.weights == "distance" else 2
                with np.errstate(divide="ignore"):
                    weights = 1.0 / knn_distances**power
                # A query that coincides with training points has infinite
                # weight on them: those neighbours decide alone.
                exact = knn_distances == 0
                has_exact = exact.any(axis=1)
                weights[has_exact] = exact[has_exact].astype(float)
                return weights
            case "gaussian":
                # Bandwidth per query, so a row's prediction never depends on
                # which other rows were predicted with it.
                sigma = knn_distances.std(axis=1, keepdims=True)
                sigma = np.where(sigma > 0, sigma, 1.0)
                return np.exp(-(knn_distances**2) / (2 * sigma**2))
