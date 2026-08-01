"""Stores the random forest model."""

import numpy as np

from dolcestat.metrics import ClassificationAnalyzer, RegressionAnalyzer
from dolcestat.preprocessing import DolceSet
from dolcestat.trees.decision_tree import DecisionTree

from .input_validation import (
    validate_categorical_features,
    validate_fit_algorithm,
    validate_if_fitting_without_target,
    validate_if_training_not_loaded,
    validate_impurity_measure,
    validate_max_depth,
    validate_min_samples_to_split,
    validate_n_feat,
    validate_n_trees,
    validate_X,
    validate_X_y,
    validate_y,
)


class RandomForest:

    def __init__(
        self,
        data=None,
        n_trees=1000,
        n_feat=5,
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

        # 2. Assign parameters
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.n_feat = n_feat
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

    # Assumes CART is being used
    def fit(self):

        validate_if_training_not_loaded(self.is_training_loaded)
        validate_if_fitting_without_target(self.can_train)

        self.forest = []
        for t in range(self.n_trees):

            # 1. Extract a subset of features from the original dataset.
            random_idxs = np.random.choice(self.X.shape[1], self.n_feat, replace=False)
            sub_X = self.X[:, random_idxs]
            sub_categorical_features = self.categorical_features & set(
                random_idxs.tolist()
            )

            # 2. Apply bootstrapping on the sampled dataset
            bootstrapped_idxs = np.random.randint(
                0, sub_X.shape[0], size=sub_X.shape[0]
            )
            sub_X = sub_X[bootstrapped_idxs]
            sub_y = self.y[bootstrapped_idxs]

            # 3. Re-build a dolceset object
            data = DolceSet()
            data.X = sub_X
            data.y = sub_y
            data.can_train = True

            # 4. Build a tree and add it to the forest
            tree = DecisionTree(
                data=data,
                max_depth=self.max_depth,
                min_samples_to_split=self.min_samples_to_split,
                fit_algorithm=self.fit_algorithm,
                impurity_measure=self.impurity_measure,
                categorical_features=sub_categorical_features,
            )
            tree.fit()
            # Remember which original columns this tree was trained on, so
            # predict() can hand it the matching slice of the full feature
            # matrix (its split_feature indices are relative to sub_X).
            tree.feature_idxs = random_idxs

            self.forest.append(tree)

    def predict(self, data):

        prediction_by_tree = []

        # 1. For each row, make a prediction with each tree, restricting the
        #    input to the feature subset that tree was trained on
        for tree in self.forest:
            sub_data = DolceSet()
            sub_data.X = data.X[:, tree.feature_idxs]
            sub_data.y = data.y
            sub_data.can_train = data.can_train
            prediction_by_tree.append(tree.predict(sub_data).y_fit)

        # 2. For each row, apply majority vote (mode for classification,
        #    mean for regression)
        predictions = np.array(prediction_by_tree)
        if self.impurity_measure in ["gini", "entropy"]:
            preds = np.array(
                [
                    np.bincount(predictions[:, i].astype(int)).argmax()
                    for i in range(predictions.shape[1])
                ]
            )
            return ClassificationAnalyzer(data.y, preds)
        else:
            preds = predictions.mean(axis=0)
            return RegressionAnalyzer(data.y, preds)
