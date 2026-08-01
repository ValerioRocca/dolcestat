"""Stores input validation for the optimizers.

Data validation lives in ``dolcestat.core.input_validation``, where the training
loop that receives the data can apply it.
"""


def validate_alpha(alpha):
    if not isinstance(alpha, (float, int)):
        raise ValueError("alpha must be a float or int.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in the interval (0, 1).")


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
