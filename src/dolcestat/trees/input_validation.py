"""Stores input validation for the tree models."""

import numpy as np


def validate_X(X):
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values.")


def validate_y(y):
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array (n_samples,).")
    if np.any(np.isinf(y)):
        raise ValueError("y contains infinite values.")


def validate_X_y(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be the same.")


def validate_categorical_features(categorical_features, n_features):
    if categorical_features is None:
        return
    for idx in categorical_features:
        if not isinstance(idx, int):
            raise ValueError("categorical_features must contain integer indices.")
        if idx < 0 or idx >= n_features:
            raise ValueError(
                f"categorical_features index {idx} is out of bounds for "
                f"X with {n_features} features."
            )


def validate_max_depth(max_depth):
    if max_depth is None:
        return
    if not isinstance(max_depth, int):
        raise ValueError("max_depth must be an integer or None.")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")


def validate_n_trees(n_trees):
    if not isinstance(n_trees, int):
        raise ValueError("n_trees must be an integer.")
    if n_trees < 1:
        raise ValueError("n_trees must be at least 1.")


def validate_n_feat(n_feat):
    if not isinstance(n_feat, int):
        raise ValueError("n_feat must be an integer.")
    if n_feat < 1:
        raise ValueError("n_feat must be at least 1.")


def validate_min_samples_to_split(min_samples_to_split):
    if not isinstance(min_samples_to_split, int):
        raise ValueError("min_samples_to_split must be an integer.")
    if min_samples_to_split < 2:
        raise ValueError("min_samples_to_split must be at least 2.")


def validate_fit_algorithm(fit_algorithm):
    valid = {"cart"}
    if fit_algorithm not in valid:
        raise ValueError(
            f"fit_algorithm must be one of {sorted(valid)}, got '{fit_algorithm}'."
        )


def validate_impurity_measure(impurity_measure):
    valid = {"gini", "entropy", "mse"}
    if impurity_measure not in valid:
        raise ValueError(
            f"impurity_measure must be one of {sorted(valid)}, got '{impurity_measure}'."
        )


def validate_if_fitting_without_target(can_train):
    if not can_train:
        raise ValueError(
            "Cannot fit: the DolceSet has no target and can only be used "
            "for testing/prediction, not training."
        )


def validate_if_training_not_loaded(is_training_loaded):
    if not is_training_loaded:
        raise ValueError(
            "Cannot fit: no training data has been loaded. Pass data at "
            "construction or call load_training(data) before fit()."
        )
