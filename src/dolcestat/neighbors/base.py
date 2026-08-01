"""Stores the shared base for nearest-neighbour estimators."""

from abc import ABC, abstractmethod

import numpy as np

from .distances import minkowski
from .input_validation import validate_k, validate_metric, validate_weights


class BaseNeighbors(ABC):

    def __init__(self, labeled_data):
        self.lab_X = labeled_data.X
        self.lab_y = labeled_data.y

    def _compute_distances_matrix(self, X_lab, X_unlab, metric, minkowski_p):
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
        for x_unlab in X_unlab:
            row = []
            for x_lab in X_lab:
                distance = minkowski(x_unlab, x_lab, minkowski_p)
                row.append(distance)
            dist_matrix.append(row)
        return np.array(dist_matrix)

    def _get_knn_idxs(self, distances_matrix, k):
        """Returns the indexes of the K nearest neighbours for each unit to predict"""
        validate_k(k)
        if k > distances_matrix.shape[1]:
            raise ValueError(
                f"k ({k}) cannot be greater than the number of labeled samples ({distances_matrix.shape[1]})."
            )
        return np.argsort(distances_matrix, axis=1)[:, :k]

    def _get_knn_y(self, y_lab, knn_idxs):
        """Returns the Y values of the K nearest neighbours for each unit to predict"""
        return y_lab[knn_idxs]

    def _get_knn_weights(self, distances_matrix, knn_idxs, weights):
        """Returns a weight matrix for the K nearest neighbours"""
        validate_weights(weights)
        knn_distances = distances_matrix[
            np.arange(distances_matrix.shape[0])[:, None], knn_idxs
        ]
        match weights:
            case "uniform":
                return np.ones_like(knn_distances)
            case "distance":
                # avoid division by zero for exact matches
                return np.where(knn_distances == 0, np.inf, 1.0 / knn_distances)
            case "squared_distance":
                return np.where(knn_distances == 0, np.inf, 1.0 / (knn_distances**2))
            case "gaussian":
                sigma = np.std(knn_distances) if np.std(knn_distances) > 0 else 1.0
                return np.exp(-(knn_distances**2) / (2 * sigma**2))
