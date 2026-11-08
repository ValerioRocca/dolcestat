"""Gradient boosting with small regression trees."""

import numpy as np

from dolcestat.metrics import (
    ClassificationAnalyzer,
    MulticlassAnalyzer,
    RegressionAnalyzer,
)

from .decision_tree import DecisionTree
from .input_validation import (
    validate_alpha,
    validate_categorical_features,
    validate_class_labels,
    validate_fit_algorithm,
    validate_impurity_measure,
    validate_is_fitted,
    validate_max_depth,
    validate_max_forest_size,
    validate_min_samples_to_split,
    validate_prediction_data,
    validate_training_data,
)


class GradientBoostingTree:
    """Gradient boosting with CART trees, for regression or classification.

    Trees are grown one after the other, each fitted to the residual error of
    the ensemble so far; its prediction, shrunk by ``alpha``, is added on top.

    Built from its settings alone; ``fit(data)`` boosts and returns the model,
    and ``predict(data)`` returns an analyzer.

    Parameters
    ----------
    alpha : float, default 0.1
        Learning rate: how much of each tree's prediction is added.
    max_depth : int or None, default 3
        Depth of each tree. Kept small on purpose: boosting adds up many weak,
        shallow trees.
    min_samples_to_split : int, default 2
    max_forest_size : int, default 100
        Number of boosting rounds (trees per class).
    fit_algorithm : {"cart"}, default "cart"
    impurity_measure : {"gini", "entropy", "mse"}, default "gini"
        Selects the task: "mse" is regression, "gini" / "entropy" is
        classification (labels 0..K-1, see DolceSet.encode_target). The trees
        themselves always split on MSE, because they are fitted to continuous
        residuals.
    categorical_features : iterable of int, optional
        As in :class:`DecisionTree`.

    Classification is multinomial: one forest per class boosts that class's
    log-odds, and a shared softmax turns the scores into probabilities. Two
    classes is just K = 2 of the same code.
    """

    def __init__(
        self,
        *,
        alpha=0.1,
        max_depth=3,
        min_samples_to_split=2,
        max_forest_size=100,
        fit_algorithm="cart",
        impurity_measure="gini",
        categorical_features=None,
    ):

        # 1. Validate input parameters
        validate_alpha(alpha)
        validate_max_depth(max_depth)
        validate_min_samples_to_split(min_samples_to_split)
        validate_max_forest_size(max_forest_size)
        validate_fit_algorithm(fit_algorithm)
        validate_impurity_measure(impurity_measure)
        validate_categorical_features(categorical_features)

        # 2. Assign parameters
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_samples_to_split = min_samples_to_split
        self.max_forest_size = max_forest_size
        self.fit_algorithm = fit_algorithm
        self.impurity_measure = impurity_measure
        self.categorical_features = (
            set(categorical_features) if categorical_features is not None else set()
        )

        # 3. Initialize the fitted state
        self.forest = []
        self.is_fitted = False
        self.n_features = None
        self.init_value = None
        self.classes_ = None

    @property
    def _is_regression(self):
        return self.impurity_measure == "mse"

    def _new_tree(self):
        # Boosted trees are fitted to continuous residuals, so they split on
        # MSE whatever this model's own impurity_measure is (gini/entropy
        # would treat the residuals as discrete class labels).
        return DecisionTree(
            max_depth=self.max_depth,
            min_samples_to_split=self.min_samples_to_split,
            fit_algorithm=self.fit_algorithm,
            impurity_measure="mse",
            categorical_features=self.categorical_features,
        )

    def _fit_regression(self, X, y):

        # 1. Compute the baseline
        self.init_value = np.mean(y)
        y_hat = np.full(X.shape[0], self.init_value)

        for i in range(self.max_forest_size):

            # 2. Compute the residuals
            res = y - y_hat

            # 3. Fit a tree on the residuals
            tree = self._new_tree()._fit_xy(X, res)
            self.forest.append(tree)

            # 4. Update Y hat
            y_hat = y_hat + self.alpha * tree._predict_values(X)

    def _compute_classification_baseline(self, y):
        # Log-odds baseline: log of each class's frequency, one value per
        # class. Softmax-ing these later reproduces the class priors, the
        # standard init value for (multinomial) log-odds gradient boosting.
        # Binary is just the 2-class case of the same formula.
        _, counts = np.unique(y, return_counts=True)
        return np.log(counts / len(y))

    def _softmax(self, y_hat):
        # y_hat: (n_samples, n_classes) log-odds. Row-wise softmax.
        shifted = y_hat - np.max(y_hat, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _fit_classification(self, X, y, n_classes):
        n_samples = X.shape[0]

        # 0. One-hot-encode the target: column k is the indicator y == class_k.
        #    self.forest[k] is the list of trees boosting class k's log-odds;
        #    binary classification is just this with n_classes == 2.
        y_one_hot = (y[:, np.newaxis] == self.classes_[np.newaxis, :]).astype(float)

        # 1. Compute the baseline: one log-odds value per class
        self.init_value = self._compute_classification_baseline(y)
        y_hat = np.tile(self.init_value, (n_samples, 1))

        # 2. One forest per class
        self.forest = [[] for _ in range(n_classes)]

        for i in range(self.max_forest_size):

            # 3. Current class probabilities from the shared softmax
            p_hat = self._softmax(y_hat)

            # Compared with regression, there is one forest per class, each
            # fitted on that class's own residual against the shared softmax.
            for k in range(n_classes):

                # 4. Compute the residuals for class k
                res = y_one_hot[:, k] - p_hat[:, k]

                # 5. Fit a regression tree on the residuals
                tree = self._new_tree()._fit_xy(X, res)
                self.forest[k].append(tree)

                # 6. Update column k of y_hat only
                y_hat[:, k] = y_hat[:, k] + self.alpha * tree._predict_values(X)

    def fit(self, data):
        """Boosts on ``data`` (a DolceSet with a target); returns self.

        Fitting again starts from scratch: the model never accumulates trees
        across calls.
        """
        validate_training_data(data)
        validate_categorical_features(self.categorical_features, data.X.shape[1])

        self.forest = []
        self.n_features = data.X.shape[1]
        if self._is_regression:
            self._fit_regression(data.X, data.y)
        else:
            n_classes = validate_class_labels(data.y)
            self.classes_ = np.arange(n_classes)
            self._fit_classification(data.X, data.y.astype(int), n_classes)
        self.is_fitted = True
        return self

    def _class_scores(self, X):
        # Log-odds per class: baseline plus every tree's shrunk contribution
        y_hat = np.tile(self.init_value, (X.shape[0], 1))
        for k, class_forest in enumerate(self.forest):
            for tree in class_forest:
                y_hat[:, k] = y_hat[:, k] + self.alpha * tree._predict_values(X)
        return y_hat

    def predict(self, data):
        """Predictions for ``data`` (no target needed), wrapped in an analyzer.

        Regression returns a RegressionAnalyzer. Binary classification returns
        a ClassificationAnalyzer whose ``y_fit`` holds the class-1 probability;
        three or more classes return a MulticlassAnalyzer with the predicted
        labels and the full probability matrix.
        """
        validate_is_fitted(self.is_fitted, "GradientBoostingTree")
        validate_prediction_data(data, self.n_features)

        if self._is_regression:

            # 1. Compute baseline contribution
            y_hat = np.full(data.X.shape[0], self.init_value)

            # 2. Add each tree's contribution
            for tree in self.forest:
                y_hat = y_hat + self.alpha * tree._predict_values(data.X)

            # 3. Wrap final prediction around Analyzer object
            return RegressionAnalyzer(data.y, y_hat)

        # Classification: self.forest is one list of trees per class, and
        # init_value / the scores are (n_classes,) / (n_samples, n_classes).
        proba = self._softmax(self._class_scores(data.X))
        if len(self.classes_) > 2:
            return MulticlassAnalyzer(
                data.y, proba.argmax(axis=1), proba=proba, classes=self.classes_
            )
        # Binary: ClassificationAnalyzer expects a single class-1 probability
        # column, i.e. softmax's column 1.
        return ClassificationAnalyzer(data.y, proba[:, 1])
