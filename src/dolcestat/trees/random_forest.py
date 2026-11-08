"""Random forest: bagged decision trees on bootstrap samples and random columns."""

import numpy as np

from dolcestat.metrics import RegressionAnalyzer, classification_analyzer
from dolcestat.trees.decision_tree import DecisionTree

from .input_validation import (
    validate_categorical_features,
    validate_class_labels,
    validate_fit_algorithm,
    validate_impurity_measure,
    validate_is_fitted,
    validate_max_depth,
    validate_min_samples_to_split,
    validate_n_feat,
    validate_n_trees,
    validate_prediction_data,
    validate_training_data,
)


class RandomForest:
    """A random forest of CART trees, for classification or regression.

    Each tree is grown on a bootstrap sample of the rows and a random subset of
    ``n_feat`` columns, drawn once per tree. (Textbook forests re-draw the
    column subset at every split; drawing it once per tree is a simpler variant
    that still decorrelates the trees.) Classification combines the trees by
    majority vote, regression by averaging.

    Built from its settings alone; ``fit(data)`` grows the forest and returns
    it, and ``predict(data)`` returns an analyzer.

    Parameters
    ----------
    n_trees : int, default 100
    n_feat : int or None, default None
        Columns each tree sees. None uses the square root of the number of
        features, rounded down (at least 1).
    max_depth, min_samples_to_split, fit_algorithm, impurity_measure,
    categorical_features
        As in :class:`DecisionTree`; ``categorical_features`` indexes the
        columns of the full X.
    """

    def __init__(
        self,
        *,
        n_trees=100,
        n_feat=None,
        max_depth=None,
        min_samples_to_split=2,
        fit_algorithm="cart",
        impurity_measure="gini",
        categorical_features=None,
    ):

        # 1. Validate input parameters
        validate_n_trees(n_trees)
        validate_n_feat(n_feat)
        validate_max_depth(max_depth)
        validate_min_samples_to_split(min_samples_to_split)
        validate_fit_algorithm(fit_algorithm)
        validate_impurity_measure(impurity_measure)
        validate_categorical_features(categorical_features)

        # 2. Assign parameters
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.n_feat = n_feat
        self.min_samples_to_split = min_samples_to_split
        self.fit_algorithm = fit_algorithm
        self.impurity_measure = impurity_measure
        self.categorical_features = (
            set(categorical_features) if categorical_features is not None else set()
        )

        # 3. Initialize the fitted state
        self.forest = []
        self.is_fitted = False
        self.n_features = None
        self.classes_ = None

    @property
    def _is_classification(self):
        return self.impurity_measure in ("gini", "entropy")

    # Assumes CART is being used
    def fit(self, data):
        """Grows the forest on ``data`` (a DolceSet with a target); returns self."""
        validate_training_data(data)
        n_samples, n_features = data.X.shape
        validate_categorical_features(self.categorical_features, n_features)
        n_feat = self.n_feat
        if n_feat is None:
            n_feat = max(1, int(np.sqrt(n_features)))
        validate_n_feat(n_feat, n_features)

        y = data.y
        if self._is_classification:
            self.classes_ = np.arange(validate_class_labels(y))
            y = y.astype(int)
        self.n_features = n_features

        self.forest = []
        for t in range(self.n_trees):

            # 1. Extract a subset of features from the original dataset.
            random_idxs = np.random.choice(n_features, n_feat, replace=False)
            sub_X = data.X[:, random_idxs]
            # The tree sees the subset as its own columns 0..n_feat-1, so the
            # categorical columns are renumbered to their new positions.
            sub_categorical_features = {
                position
                for position, original in enumerate(random_idxs.tolist())
                if original in self.categorical_features
            }

            # 2. Apply bootstrapping on the sampled dataset
            bootstrapped_idxs = np.random.randint(0, n_samples, size=n_samples)
            sub_X = sub_X[bootstrapped_idxs]
            sub_y = y[bootstrapped_idxs]

            # 3. Build a tree on it and add it to the forest
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_to_split=self.min_samples_to_split,
                fit_algorithm=self.fit_algorithm,
                impurity_measure=self.impurity_measure,
                categorical_features=sub_categorical_features,
            )
            tree._fit_xy(sub_X, sub_y)
            # Remember which original columns this tree was trained on, so
            # predict() can hand it the matching slice of the full feature
            # matrix (its split_feature indices are relative to sub_X).
            tree.feature_idxs = random_idxs

            self.forest.append(tree)

        self.is_fitted = True
        return self

    def predict(self, data):
        """Predictions for ``data`` (no target needed), wrapped in an analyzer.

        Classification returns the voted labels, with the share of votes each
        class received as probabilities; regression returns the mean.
        """
        validate_is_fitted(self.is_fitted, "RandomForest")
        validate_prediction_data(data, self.n_features)
        n = data.X.shape[0]

        # 1. For each row, make a prediction with each tree, restricting the
        #    input to the feature subset that tree was trained on
        prediction_by_tree = [
            tree._predict_values(data.X[:, tree.feature_idxs]) for tree in self.forest
        ]

        # 2. Combine them: majority vote for classification, mean for regression
        if self._is_classification:
            n_classes = len(self.classes_)
            votes = np.zeros((n, n_classes))
            for predictions in prediction_by_tree:
                votes[np.arange(n), predictions.astype(int)] += 1
            preds = votes.argmax(axis=1)
            return classification_analyzer(
                data.y, preds, n_classes, proba=votes / len(self.forest)
            )
        return RegressionAnalyzer(data.y, np.mean(prediction_by_tree, axis=0))
