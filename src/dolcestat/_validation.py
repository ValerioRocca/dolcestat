"""Input checks shared by every model package.

Each package keeps its own ``input_validation`` module for its hyperparameters;
the checks on the *data* are the same everywhere, so they live here.
"""

import numpy as np


def _is_numeric(array):
    return array.dtype == bool or np.issubdtype(array.dtype, np.number)


def validate_X(X):
    """Checks a feature matrix: a non-empty, finite, numeric 2D NumPy array."""
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    if not _is_numeric(X):
        raise ValueError(
            "X must be numeric. Encode text columns first, for example with "
            "DolceSet.one_hot_encode."
        )
    if X.shape[0] == 0:
        raise ValueError("X has no rows.")
    if np.any(np.isnan(X)):
        raise ValueError(
            "X contains missing values (NaN). Models cannot handle them: drop "
            "or fill those rows before fitting."
        )
    if np.any(np.isinf(X)):
        raise ValueError("X contains infinite values.")


def validate_y(y):
    """Checks a target vector: a finite, numeric 1D NumPy array."""
    if not isinstance(y, np.ndarray):
        raise ValueError("y must be a numpy array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array (n_samples,).")
    if not _is_numeric(y):
        raise ValueError(
            "y must be numeric. For text class labels, convert them to "
            "integers with DolceSet.encode_target()."
        )
    if np.any(np.isnan(y)):
        raise ValueError("y contains missing values (NaN).")
    if np.any(np.isinf(y)):
        raise ValueError("y contains infinite values.")


def validate_X_y(X, y):
    """Checks that ``X`` and ``y`` have the same number of rows."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("Number of samples in X and y must be the same.")


def validate_has_target(data):
    """Checks that a DolceSet carries a target, so it can be used to fit."""
    if not getattr(data, "can_train", False) or data.y is None:
        raise ValueError(
            "Cannot fit: the DolceSet has no target and can only be used "
            "for testing/prediction, not training."
        )


def validate_training_data(data):
    """Checks a DolceSet handed to ``fit``: it must carry a valid X and y."""
    validate_has_target(data)
    validate_X(data.X)
    validate_y(data.y)
    validate_X_y(data.X, data.y)


def validate_n_features(X, expected):
    """Checks that ``X`` has the number of columns the model was fitted on."""
    if X.shape[1] != expected:
        raise ValueError(
            f"X has {X.shape[1]} features, but the model was fitted on "
            f"{expected} features."
        )


def validate_prediction_data(data, n_features):
    """Checks a DolceSet handed to ``predict``. A target is not required."""
    validate_X(data.X)
    validate_n_features(data.X, n_features)


def validate_is_fitted(is_fitted, model_name):
    """Raises a ValueError naming the model if it has not been fitted yet."""
    if not is_fitted:
        raise ValueError(
            f"Cannot predict: this {model_name} is not fitted yet. Call fit(data) first."
        )


def validate_class_labels(y):
    """Checks that ``y`` holds the class labels 0, 1, ..., K-1 and returns K.

    Every classifier in the library works with integer labels starting at 0,
    with every class present. DolceSet.encode_target() converts any other
    labelling (text, labels starting at 1, ...) into this one.
    """
    validate_y(y)
    if np.any(y != np.floor(y)) or np.any(y < 0):
        raise ValueError(
            "Class labels must be the integers 0, 1, ..., K-1, but y holds "
            f"{np.unique(y)[:8].tolist()}. Convert them with "
            "DolceSet.encode_target()."
        )
    classes = np.unique(y)
    n_classes = int(classes.max()) + 1
    if not np.array_equal(classes, np.arange(n_classes)):
        raise ValueError(
            "Class labels must be the integers 0, 1, ..., K-1 with every "
            f"class present, but y holds {classes[:8].tolist()}. Convert them "
            "with DolceSet.encode_target()."
        )
    return n_classes


def validate_binary_target(y, model_name):
    """Checks that ``y`` only holds 0 and 1, for models that are binary-only."""
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError(
            f"{model_name} needs a binary target with values 0 and 1, but y "
            f"holds {np.unique(y)[:8].tolist()}. For more than two classes "
            "use a tree, forest, gradient boosting or KNN; to convert other "
            "labels use DolceSet.encode_target()."
        )
