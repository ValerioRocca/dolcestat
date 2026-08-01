"""Stores input validation for the network training loop and its optimizer.

These duplicate a few checks from ``dolcestat.optimization.input_validation``
on purpose: ``neural_networks`` is self-contained and imports nothing from the
other model packages, so the handful of predicates it needs live here.
"""


def validate_alpha(alpha):
    """Checks the learning rate."""
    if not isinstance(alpha, (float, int)):
        raise ValueError("alpha must be a float or int.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in the interval (0, 1).")


def validate_n_epochs(n_epochs):
    """Checks the epoch budget."""
    if not isinstance(n_epochs, int):
        raise ValueError("n_epochs must be an integer.")
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")


def validate_tol(tol):
    """Checks the convergence tolerance, which may be None to disable it."""
    if tol is None:
        return
    if not isinstance(tol, (float, int)):
        raise ValueError("tol must be a float, an int, or None.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
