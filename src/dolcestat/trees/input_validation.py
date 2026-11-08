"""Checks on the settings of the tree models."""

import numbers

from dolcestat._validation import (  # noqa: F401  (shared data checks, re-exported)
    validate_class_labels,
    validate_is_fitted,
    validate_prediction_data,
    validate_training_data,
    validate_X,
    validate_X_y,
    validate_y,
)


def validate_categorical_features(categorical_features, n_features=None):
    """Checks the column indices; the range is only checked once ``n_features``
    (the width of X) is known."""
    if categorical_features is None:
        return
    for idx in categorical_features:
        if isinstance(idx, bool) or not isinstance(idx, numbers.Integral):
            raise ValueError("categorical_features must contain integer indices.")
        if idx < 0 or (n_features is not None and idx >= n_features):
            raise ValueError(
                f"categorical_features index {idx} is out of bounds for "
                f"X with {n_features} features."
            )


def validate_max_depth(max_depth):
    """Checks the depth limit: None (no limit) or an integer of at least 1."""
    if max_depth is None:
        return
    if not isinstance(max_depth, int):
        raise ValueError("max_depth must be an integer or None.")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")


def validate_n_trees(n_trees):
    """Checks the number of trees in a forest: an integer of at least 1."""
    if not isinstance(n_trees, int):
        raise ValueError("n_trees must be an integer.")
    if n_trees < 1:
        raise ValueError("n_trees must be at least 1.")


def validate_max_forest_size(max_forest_size):
    """Checks the number of boosting rounds: an integer of at least 1."""
    if not isinstance(max_forest_size, int):
        raise ValueError("max_forest_size must be an integer.")
    if max_forest_size < 1:
        raise ValueError("max_forest_size must be at least 1.")


def validate_alpha(alpha):
    """Checks the boosting learning rate: a positive number."""
    if not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a number.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")


def validate_n_feat(n_feat, n_features=None):
    """Checks the number of columns each forest tree sees. None means "pick
    for me" (the square root of the number of features)."""
    if n_feat is None:
        return
    if not isinstance(n_feat, int):
        raise ValueError("n_feat must be an integer or None.")
    if n_feat < 1:
        raise ValueError("n_feat must be at least 1.")
    if n_features is not None and n_feat > n_features:
        raise ValueError(
            f"n_feat ({n_feat}) cannot be larger than the number of features "
            f"({n_features})."
        )


def validate_min_samples_to_split(min_samples_to_split):
    """Checks the smallest node that may be split: an integer of at least 2."""
    if not isinstance(min_samples_to_split, int):
        raise ValueError("min_samples_to_split must be an integer.")
    if min_samples_to_split < 2:
        raise ValueError("min_samples_to_split must be at least 2.")


def validate_fit_algorithm(fit_algorithm):
    """Checks the tree-growing algorithm (only 'cart' exists)."""
    valid = {"cart"}
    if fit_algorithm not in valid:
        raise ValueError(
            f"fit_algorithm must be one of {sorted(valid)}, got '{fit_algorithm}'."
        )


def validate_impurity_measure(impurity_measure):
    """Checks the split criterion: 'gini', 'entropy' or 'mse'."""
    valid = {"gini", "entropy", "mse"}
    if impurity_measure not in valid:
        raise ValueError(
            f"impurity_measure must be one of {sorted(valid)}, got '{impurity_measure}'."
        )
