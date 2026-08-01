"""Stores the k-nearest-neighbours estimators."""

import numpy as np

from dolcestat.metrics import KNNClassificationAnalyzer, KNNRegressionAnalyzer

from .base import BaseNeighbors


class KNNClassifier(BaseNeighbors):

    def __init__(self, labeled_data):
        super().__init__(labeled_data)

    def predict(
        self,
        unlabeled_data,
        k,
        metric="euclidean",
        minkowski_p=None,
        weights="uniform",
    ):

        # 1. Compute distances matrix
        distances_matrix = self._compute_distances_matrix(
            self.lab_X, unlabeled_data.X, metric, minkowski_p
        )

        # 2. Extract indexes of K Nearest Neighbours
        knn_idxs = self._get_knn_idxs(distances_matrix, k)

        # 3. Extract K Nearest Neighbours Y values
        knn_y = self._get_knn_y(self.lab_y, knn_idxs)

        # 4. Compute K Nearest Neighbours weights
        weight_matrix = self._get_knn_weights(distances_matrix, knn_idxs, weights)

        # 5. For each row, sum weights per class and pick the class with the highest total weight
        classes = np.unique(self.lab_y)
        weighted_votes = np.array(
            [
                [weight_matrix[i][knn_y[i] == c].sum() for c in classes]
                for i in range(len(knn_y))
            ]
        )
        pred_y = classes[np.argmax(weighted_votes, axis=1)]

        # 6. Compute class probabilities
        row_sums = weighted_votes.sum(axis=1, keepdims=True)
        pred_proba = weighted_votes / row_sums

        # 7. Return an analyzer carrying the predictions and class probabilities
        return KNNClassificationAnalyzer(unlabeled_data.y, pred_y, proba=pred_proba)


class KNNRegressor(BaseNeighbors):

    def __init__(self, labeled_data):
        super().__init__(labeled_data)

    def predict(
        self,
        unlabeled_data,
        k,
        metric="euclidean",
        minkowski_p=None,
        weights="uniform",
    ):

        # 1. Compute distances matrix
        distances_matrix = self._compute_distances_matrix(
            self.lab_X, unlabeled_data.X, metric, minkowski_p
        )

        # 2. Extract indexes of K Nearest Neighbours
        knn_idxs = self._get_knn_idxs(distances_matrix, k)

        # 3. Extract K Nearest Neighbours Y values
        knn_y = self._get_knn_y(self.lab_y, knn_idxs)

        # 4. Compute K Nearest Neighbours weights
        weight_matrix = self._get_knn_weights(distances_matrix, knn_idxs, weights)

        # 5. Compute weighted average: sum(y * w) / sum(w)
        pred_y = (knn_y * weight_matrix).sum(axis=1) / weight_matrix.sum(axis=1)

        # 6. Return an analyzer carrying the predictions and neighbour weights
        return KNNRegressionAnalyzer(
            unlabeled_data.y, pred_y, weight_matrix=weight_matrix
        )
