from dolcestat.optimization.gradient_descent import GradientDescent
from dolcestat.optimization.newton import NewtonMethod


def validate_optimizer(optimizer, allowed_optimizers, model_name):
    if isinstance(optimizer, str):
        if optimizer not in allowed_optimizers:
            raise ValueError(
                f"optimizer '{optimizer}' is not supported by {model_name}. "
                f"Allowed: {sorted(allowed_optimizers)}."
            )
    elif isinstance(optimizer, (GradientDescent, NewtonMethod)):
        if optimizer.is_training_loaded:
            raise ValueError(
                "Pass a data-less optimizer: the given "
                f"{type(optimizer).__name__} already has training data loaded. "
                "Construct it without `data` and let the model load it."
            )
    else:
        raise ValueError(
            f"optimizer must be a string ({sorted(allowed_optimizers)}) or a "
            "data-less GradientDescent/NewtonMethod instance, got "
            f"{type(optimizer).__name__}."
        )
