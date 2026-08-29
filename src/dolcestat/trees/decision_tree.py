import numpy as np

from dolcestat.metrics import ClassificationAnalyzer, RegressionAnalyzer
from dolcestat.trees.tree import Node

from .input_validation import (
    validate_categorical_features,
    validate_fit_algorithm,
    validate_if_fitting_without_target,
    validate_if_training_not_loaded,
    validate_impurity_measure,
    validate_max_depth,
    validate_min_samples_to_split,
    validate_X,
    validate_X_y,
    validate_y,
)


class DecisionTree:

    def __init__(
        self,
        data=None,
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

        # 2. Assign parameters
        self.max_depth = max_depth
        self.min_samples_to_split = min_samples_to_split
        self.fit_algorithm = fit_algorithm
        self.impurity_measure = impurity_measure
        # Column indices of X to split on by category membership (==) instead
        # of by numeric threshold (<=). Values are still expected to be
        # numeric (e.g. label-encoded); range is validated once X is known.
        self.categorical_features = (
            set(categorical_features) if categorical_features is not None else set()
        )

        # 3. Initialize tree
        self.tree = None

        # 4. Training data is optional at construction. When omitted, the tree
        #    must be populated later via load_training before fit.
        self.can_train = False
        self.is_training_loaded = False
        if data is not None:
            self.load_training(data)

    def load_training(self, data):
        # 1. Validate input data
        validate_X(data.X)
        if data.can_train:
            validate_y(data.y)
            validate_X_y(data.X, data.y)

        # 2. Assign data-dependent attributes
        self.can_train = data.can_train
        self.X = data.X
        if self.can_train:
            self.y = data.y
        self.n_samples = data.X.shape[0]
        self.n_features = data.X.shape[1]
        validate_categorical_features(self.categorical_features, self.n_features)

        self.is_training_loaded = True

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
        node = Node(X, y, depth)

        # 2. First check stopping criteria
        if (
            (max_depth is not None and depth >= max_depth)
            or len(y) < min_samples_to_split
            or len(set(y)) == 1
        ):
            node.is_leaf = True
            if impurity_measure in ["gini", "entropy"]:
                node.prediction = np.bincount(y).argmax()
            else:
                node.prediction = np.mean(y)
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
            if impurity_measure in ["gini", "entropy"]:
                node.prediction = np.bincount(y).argmax()
            else:
                node.prediction = np.mean(y)
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

    def fit(self):

        validate_if_training_not_loaded(self.is_training_loaded)
        validate_if_fitting_without_target(self.can_train)

        match self.fit_algorithm:
            case "cart":
                self.tree = self._cart_iterator(
                    self.X,
                    self.y,
                    depth=0,
                    max_depth=self.max_depth,
                    min_samples_to_split=self.min_samples_to_split,
                    impurity_measure=self.impurity_measure,
                    categorical_features=self.categorical_features,
                )

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

    def predict(self, data):
        preds = np.array([self._traverse(row, self.tree) for row in data.X])
        if self.impurity_measure == "mse":
            return RegressionAnalyzer(data.y, preds)
        return ClassificationAnalyzer(data.y, preds)
