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


def validate_alpha(alpha):
    if not isinstance(alpha, (float, int)):
        raise ValueError("alpha must be a float or int.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in the interval (0, 1).")


def validate_n_iters(n_iters):
    if not isinstance(n_iters, int):
        raise ValueError("n_iters must be an integer.")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive.")


def validate_tol(tol):
    if not isinstance(tol, (float, int)):
        raise ValueError("tol must be a float or int.")
    if tol <= 0:
        raise ValueError("tol must be positive.")


def validate_loss_function(loss_function):
    if loss_function is not None and not isinstance(loss_function, str):
        raise ValueError("loss_function must be a string or None.")
    if loss_function is not None and loss_function not in ["mse", "bce"]:
        raise ValueError("loss_function must be 'mse', 'bce', or None.")


def validate_flavor(flavor):
    valid = {"batch", "sgd", "mini_batch"}
    if flavor not in valid:
        raise ValueError(f"flavor must be one of {sorted(valid)}, got '{flavor}'.")


def validate_batch_fraction(batch_fraction):
    if not isinstance(batch_fraction, (float, int)):
        raise ValueError("batch_fraction must be a float or int.")
    if not (0 < batch_fraction <= 1):
        raise ValueError("batch_fraction must be in the interval (0, 1].")


def validate_get_weights_iteration(iteration, n_iters):
    if iteration is not None:
        if not isinstance(iteration, int):
            raise ValueError("iteration must be an integer.")
        if iteration < -n_iters or iteration >= n_iters:
            raise ValueError(f"iteration must be between {-n_iters} and {n_iters - 1}.")


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


def validate_momentum_type(momentum_type):
    valid = {None, "polyak", "nesterov"}
    if momentum_type not in valid:
        raise ValueError(
            f"momentum_type must be one of [None, 'nesterov', 'polyak'], "
            f"got '{momentum_type}'."
        )


def validate_momentum_rate(momentum_rate):
    if not isinstance(momentum_rate, (float, int)):
        raise ValueError("momentum_rate must be a float or int.")
    if not (0 <= momentum_rate < 1):
        raise ValueError("momentum_rate must be in the interval [0, 1).")
