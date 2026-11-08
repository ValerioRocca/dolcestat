"""CART decision tree for classification and regression."""

import numpy as np

from dolcestat.metrics import RegressionAnalyzer, classification_analyzer
from dolcestat.trees.tree import Node

from .input_validation import (
    validate_categorical_features,
    validate_class_labels,
    validate_fit_algorithm,
    validate_impurity_measure,
    validate_is_fitted,
    validate_max_depth,
    validate_min_samples_to_split,
    validate_prediction_data,
    validate_training_data,
)


class DecisionTree:
    """A CART decision tree, for classification or regression.

    Built from its settings alone; ``fit(data)`` grows the tree and returns it,
    and ``predict(data)`` returns an analyzer.

    Parameters
    ----------
    max_depth : int or None, default None
        Maximum depth of the tree; None grows until the leaves are pure.
    min_samples_to_split : int, default 2
        Nodes with fewer rows than this become leaves.
    fit_algorithm : {"cart"}, default "cart"
    impurity_measure : {"gini", "entropy", "mse"}, default "gini"
        Selects the task: "gini" / "entropy" for classification (the target
        must hold the class labels 0..K-1, see DolceSet.encode_target), "mse"
        for regression.
    categorical_features : iterable of int, optional
        Column indices of X to split on by category membership (``==``) instead
        of by numeric threshold (``<=``). Values are still expected to be
        numeric (e.g. label-encoded).
    """

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_to_split=2,
        fit_algorithm="cart",
        impurity_measure="gini",
        categorical_features=None,
    ):

        # 1. Validate input parameters
        validate_max_depth(max_depth)
        validate_min_samples_to_split(min_samples_to_split)
        validate_fit_algorithm(fit_algorithm)
        validate_impurity_measure(impurity_measure)
        validate_categorical_features(categorical_features)

        # 2. Assign parameters
        self.max_depth = max_depth
        self.min_samples_to_split = min_samples_to_split
        self.fit_algorithm = fit_algorithm
        self.impurity_measure = impurity_measure
        self.categorical_features = (
            set(categorical_features) if categorical_features is not None else set()
        )

        # 3. Initialize the fitted state
        self.tree = None
        self.is_fitted = False
        self.n_features = None
        # For classification: the class labels 0..K-1 seen at fit time
        self.classes_ = None

    @property
    def _is_classification(self):
        return self.impurity_measure in ("gini", "entropy")

    def _compute_gini_impurity(self, y):
        # 1. Compute prob of each output class
        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.size
        # 2. Compute Gini
        return 1 - np.sum(probs**2)

    def _compute_entropy_impurity(self, y):
        # 1. Compute prob of each output class
        _, counts = np.unique(y, return_counts=True)
        probs = counts / y.size
        # 2. Compute entropy
        return -np.sum(probs * np.log(probs))

    def _compute_mse_impurity(self, y):
        # Variance of y in the node.
        return np.mean((y - np.mean(y)) ** 2)

    def _compute_impurity(self, y):
        if self.impurity_measure == "gini":
            return self._compute_gini_impurity(y)
        elif self.impurity_measure == "entropy":
            return self._compute_entropy_impurity(y)
        elif self.impurity_measure == "mse":
            return self._compute_mse_impurity(y)
        else:
            raise ValueError(f"Unknown impurity measure: {self.impurity_measure}")

    def _leaf_prediction(self, y, impurity_measure):
        # Majority class for classification, mean target for regression
        if impurity_measure in ["gini", "entropy"]:
            return np.bincount(y).argmax()
        return np.mean(y)

    def _cart_iterator(
        self,
        X,
        y,
        depth,
        max_depth,
        min_samples_to_split,
        impurity_measure,
        categorical_features,
    ):

        # 1. Initialize a node
        node = Node(depth, n_samples=len(y))

        # 2. First check stopping criteria
        if (
            (max_depth is not None and depth >= max_depth)
            or len(y) < min_samples_to_split
            or len(set(y)) == 1
        ):
            node.is_leaf = True
            node.prediction = self._leaf_prediction(y, impurity_measure)
            return node

        # 3. Find best split
        n = y.size
        best_feature, best_threshold, best_gain = None, None, 0.0
        best_is_categorical = False

        # 3a. Compute impurity of parent node
        parent_impurity = self._compute_impurity(y)

        # 3b. For each feature, find the candidate split points
        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]
            is_categorical = feature_idx in categorical_features

            # Categorical: try each category vs the rest (equality split).
            # Numeric: try each observed value as a <= threshold.
            candidates = np.unique(values)

            # 3c. For each candidate, compute impurity reduction
            for candidate in candidates:
                if is_categorical:
                    left_mask = values == candidate
                else:
                    left_mask = values <= candidate
                right_mask = ~left_mask

                if not left_mask.any() or not right_mask.any():
                    continue

                left_impurity = self._compute_impurity(y[left_mask])
                right_impurity = self._compute_impurity(y[right_mask])

                weighted_impurity = (
                    left_mask.sum() / n * left_impurity
                    + right_mask.sum() / n * right_impurity
                )
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_feature, best_threshold, best_gain = (
                        feature_idx,
                        candidate,
                        gain,
                    )
                    best_is_categorical = is_categorical

        # 4. Second check stopping criteria
        if best_feature is None:  # no useful split found
            node.is_leaf = True
            node.prediction = self._leaf_prediction(y, impurity_measure)
            return node

        # 5. Else, iterate on next node
        node.is_leaf = False
        node.split_feature = best_feature
        node.split_threshold = best_threshold
        node.is_categorical_split = best_is_categorical

        if best_is_categorical:
            left_mask = X[:, best_feature] == best_threshold
        else:
            left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        node.left = self._cart_iterator(
            X[left_mask],
            y[left_mask],
            depth + 1,
            max_depth,
            min_samples_to_split,
            impurity_measure,
            categorical_features,
        )
        node.right = self._cart_iterator(
            X[right_mask],
            y[right_mask],
            depth + 1,
            max_depth,
            min_samples_to_split,
            impurity_measure,
            categorical_features,
        )

        return node

    def _fit_xy(self, X, y):
        # Grows the tree on raw arrays. Shared by fit() and by the ensembles,
        # which build many trees on arrays they already validated.
        self.n_features = X.shape[1]
        match self.fit_algorithm:
            case "cart":
                self.tree = self._cart_iterator(
                    X,
                    y,
                    depth=0,
                    max_depth=self.max_depth,
                    min_samples_to_split=self.min_samples_to_split,
                    impurity_measure=self.impurity_measure,
                    categorical_features=self.categorical_features,
                )
        self.is_fitted = True
        return self

    def fit(self, data):
        """Grows the tree on ``data`` (a DolceSet with a target); returns self."""
        validate_training_data(data)
        validate_categorical_features(self.categorical_features, data.X.shape[1])

        y = data.y
        if self._is_classification:
            self.classes_ = np.arange(validate_class_labels(y))
            y = y.astype(int)
        return self._fit_xy(data.X, y)

    def _traverse(self, row, node):
        if node.is_leaf:
            return node.prediction
        if node.is_categorical_split:
            goes_left = row[node.split_feature] == node.split_threshold
        else:
            goes_left = row[node.split_feature] <= node.split_threshold
        if goes_left:
            return self._traverse(row, node.left)
        return self._traverse(row, node.right)

    def _predict_values(self, X):
        return np.array([self._traverse(row, self.tree) for row in X])

    def predict(self, data):
        """Predictions for ``data`` (no target needed), wrapped in an analyzer.

        Regression returns a RegressionAnalyzer; classification returns the
        binary or the multiclass analyzer according to the number of classes
        seen at fit time.
        """
        validate_is_fitted(self.is_fitted, "DecisionTree")
        validate_prediction_data(data, self.n_features)
        preds = self._predict_values(data.X)
        if self._is_classification:
            return classification_analyzer(data.y, preds, len(self.classes_))
        return RegressionAnalyzer(data.y, preds)
