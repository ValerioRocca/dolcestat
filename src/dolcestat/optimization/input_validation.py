"""Checks on the settings of the optimizers."""

from dolcestat._validation import (  # noqa: F401  (shared data checks, re-exported)
    validate_X,
    validate_X_y,
    validate_y,
)


def validate_alpha(alpha):
    """Checks the learning rate: a number in the open interval (0, 1)."""
    if not isinstance(alpha, (float, int)):
        raise ValueError("alpha must be a float or int.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in the interval (0, 1).")


def validate_n_iters(n_iters):
    """Checks the iteration budget: a positive integer."""
    if not isinstance(n_iters, int):
        raise ValueError("n_iters must be an integer.")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive.")


def validate_tol(tol):
    """Checks the convergence tolerance: a positive number."""
    if not isinstance(tol, (float, int)):
        raise ValueError("tol must be a float or int.")
    if tol <= 0:
        raise ValueError("tol must be positive.")


def validate_loss_function(loss_function):
    """Checks the loss name: 'mse', 'bce' or None (inferred from y at fit)."""
    if loss_function is not None and not isinstance(loss_function, str):
        raise ValueError("loss_function must be a string or None.")
    if loss_function is not None and loss_function not in ["mse", "bce"]:
        raise ValueError("loss_function must be 'mse', 'bce', or None.")


def validate_flavor(flavor):
    """Checks the gradient descent flavor: 'batch', 'sgd' or 'mini_batch'."""
    valid = {"batch", "sgd", "mini_batch"}
    if flavor not in valid:
        raise ValueError(f"flavor must be one of {sorted(valid)}, got '{flavor}'.")


def validate_batch_fraction(batch_fraction):
    """Checks the mini-batch share of the rows: a number in (0, 1]."""
    if not isinstance(batch_fraction, (float, int)):
        raise ValueError("batch_fraction must be a float or int.")
    if not (0 < batch_fraction <= 1):
        raise ValueError("batch_fraction must be in the interval (0, 1].")


def validate_get_weights_iteration(iteration, n_iters):
    """Checks an iteration index: None, or an integer within the iteration budget."""
    if iteration is not None:
        if not isinstance(iteration, int):
            raise ValueError("iteration must be an integer.")
        if iteration < -n_iters or iteration >= n_iters:
            raise ValueError(f"iteration must be between {-n_iters} and {n_iters - 1}.")


def validate_momentum_type(momentum_type):
    """Checks the momentum variant: None, 'polyak' or 'nesterov'."""
    valid = {None, "polyak", "nesterov"}
    if momentum_type not in valid:
        raise ValueError(
            f"momentum_type must be one of [None, 'nesterov', 'polyak'], "
            f"got '{momentum_type}'."
        )


def validate_momentum_rate(momentum_rate):
    """Checks the momentum coefficient: a number in [0, 1)."""
    if not isinstance(momentum_rate, (float, int)):
        raise ValueError("momentum_rate must be a float or int.")
    if not (0 <= momentum_rate < 1):
        raise ValueError("momentum_rate must be in the interval [0, 1).")


def validate_l1(l1):
    """Checks the L1 penalty strength: a non-negative number."""
    if not isinstance(l1, (float, int)):
        raise ValueError("l1 must be a float or int.")
    if l1 < 0:
        raise ValueError("l1 must be non-negative.")


def validate_l2(l2):
    """Checks the L2 penalty strength: a non-negative number."""
    if not isinstance(l2, (float, int)):
        raise ValueError("l2 must be a float or int.")
    if l2 < 0:
        raise ValueError("l2 must be non-negative.")
