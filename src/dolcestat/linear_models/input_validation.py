"""Checks on the settings of the linear models."""

from dolcestat.optimization.gradient_descent import GradientDescent
from dolcestat.optimization.newton import NewtonMethod


def validate_optimizer(optimizer, allowed_optimizers, model_name):
    """Checks the ``optimizer`` setting: an allowed name or an optimizer instance."""
    if isinstance(optimizer, str):
        if optimizer not in allowed_optimizers:
            raise ValueError(
                f"optimizer '{optimizer}' is not supported by {model_name}. "
                f"Allowed: {sorted(allowed_optimizers)}."
            )
    elif not isinstance(optimizer, (GradientDescent, NewtonMethod)):
        raise ValueError(
            f"optimizer must be a string ({sorted(allowed_optimizers)}) or a "
            "GradientDescent/NewtonMethod instance, got "
            f"{type(optimizer).__name__}."
        )
