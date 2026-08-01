"""Stores input validation for the linear models."""

from dolcestat.core.optimizer import Optimizer


def validate_optimizer(optimizer, allowed_optimizers, model_name):
    """Checks that ``optimizer`` is a strategy this model supports.

    Note. There is no longer a "the optimizer must be data-less" rule: an
    Optimizer holds no data at all, and training data is supplied to ``fit``.
    The footgun that rule guarded against no longer exists.
    """
    if isinstance(optimizer, str):
        if optimizer not in allowed_optimizers:
            raise ValueError(
                f"optimizer '{optimizer}' is not supported by {model_name}. "
                f"Allowed: {sorted(allowed_optimizers)}."
            )
    elif not isinstance(optimizer, Optimizer):
        raise ValueError(
            f"optimizer must be a string ({sorted(allowed_optimizers)}) or an "
            f"Optimizer instance, got {type(optimizer).__name__}."
        )
