"""k-nearest-neighbours classifier and regressor."""

import numpy as np

from dolcestat.metrics import (
    KNNClassificationAnalyzer,
    KNNRegressionAnalyzer,
    MulticlassAnalyzer,
)

from .base import BaseNeighbors
from .input_validation import validate_class_labels


class KNNClassifier(BaseNeighbors):
    """k-nearest-neighbours classifier: the neighbours vote for a class.

    The target must hold the class labels 0..K-1 (see DolceSet.encode_target).
    ``predict`` returns a KNNClassificationAnalyzer for two classes and a
    MulticlassAnalyzer for more; both carry the per-class probabilities in
    ``proba``. Settings are described in :class:`BaseNeighbors`.
    """

    def fit(self, data):
        """Stores the labeled rows of ``data`` and its class labels; returns self.

        Raises a ValueError if the labels are not the integers 0..K-1 with every
        class present (see ``DolceSet.encode_target``).
        """
        super().fit(data)
        n_classes = validate_class_labels(self.lab_y)
        self.classes_ = np.arange(n_classes)
        self.lab_y = self.lab_y.astype(int)
        return self

    def predict(self, data):
        """Classifies the rows of ``data`` by weighted vote of their neighbours.

        ``data`` needs no target. Returns a KNNClassificationAnalyzer (two
        classes) or a MulticlassAnalyzer (more), with the class vote shares in
        ``proba``.
        """

        self._check_ready_to_predict(data)

        # 1. Compute distances matrix
        distances_matrix = self._compute_distances_matrix(self.lab_X, data.X)

        # 2. Extract indexes of K Nearest Neighbours
        knn_idxs = self._get_knn_idxs(distances_matrix)

        # 3. Extract K Nearest Neighbours Y values
        knn_y = self._get_knn_y(self.lab_y, knn_idxs)

        # 4. Compute K Nearest Neighbours weights
        weight_matrix = self._get_knn_weights(distances_matrix, knn_idxs)

        # 5. For each row, sum weights per class and pick the class with the highest total weight
        weighted_votes = np.stack(
            [(weight_matrix * (knn_y == c)).sum(axis=1) for c in self.classes_], axis=1
        )
        pred_y = self.classes_[np.argmax(weighted_votes, axis=1)]

        # 6. Compute class probabilities
        row_sums = weighted_votes.sum(axis=1, keepdims=True)
        pred_proba = weighted_votes / row_sums

        # 7. Return an analyzer carrying the predictions and class probabilities
        if len(self.classes_) > 2:
            return MulticlassAnalyzer(
                data.y, pred_y, proba=pred_proba, classes=self.classes_
            )
        return KNNClassificationAnalyzer(data.y, pred_y, proba=pred_proba)


class KNNRegressor(BaseNeighbors):
    """k-nearest-neighbours regressor: the neighbours' targets are averaged.

    Settings are described in :class:`BaseNeighbors`. ``predict`` returns a
    KNNRegressionAnalyzer carrying the neighbour weights in ``weight_matrix``.
    """

    def predict(self, data):
        """Predicts the rows of ``data`` as the weighted mean of their neighbours.

        ``data`` needs no target. Returns a KNNRegressionAnalyzer.
        """

        self._check_ready_to_predict(data)

        # 1. Compute distances matrix
        distances_matrix = self._compute_distances_matrix(self.lab_X, data.X)

        # 2. Extract indexes of K Nearest Neighbours
        knn_idxs = self._get_knn_idxs(distances_matrix)

        # 3. Extract K Nearest Neighbours Y values
        knn_y = self._get_knn_y(self.lab_y, knn_idxs)

        # 4. Compute K Nearest Neighbours weights
        weight_matrix = self._get_knn_weights(distances_matrix, knn_idxs)

        # 5. Compute weighted average: sum(y * w) / sum(w)
        pred_y = (knn_y * weight_matrix).sum(axis=1) / weight_matrix.sum(axis=1)

        # 6. Return an analyzer carrying the predictions and neighbour weights
        return KNNRegressionAnalyzer(data.y, pred_y, weight_matrix=weight_matrix)
