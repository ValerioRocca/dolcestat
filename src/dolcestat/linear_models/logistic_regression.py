"""Binary logistic regression."""

from dolcestat._validation import validate_binary_target
from dolcestat.metrics import ClassificationAnalyzer
from dolcestat.optimization.loss_and_activation import sigmoid

from .abstract import GLMAbstract


class LogisticRegression(GLMAbstract):
    """Binary logistic regression (Bernoulli GLM with logit link).

    The target must be 0/1. ``predict`` returns an analyzer whose ``y_fit``
    holds the predicted class-1 probabilities.

    Parameters
    ----------
    optimizer : str or GradientDescent/NewtonMethod, default "gradient descent"
        How to fit. One of "gradient descent" (mini-batch Nesterov GD) or
        "newton"; or an optimizer instance holding the settings you want (it is
        copied, never modified). There is no closed-form solution for logistic
        regression.
    """

    _loss_function = "bce"
    _activation = staticmethod(sigmoid)
    _analyzer_class = ClassificationAnalyzer
    _allowed_optimizers = {"gradient descent", "newton"}

    def __init__(self, optimizer="gradient descent"):
        super().__init__(optimizer)

    def _validate_target(self, y):
        validate_binary_target(y, "LogisticRegression")
