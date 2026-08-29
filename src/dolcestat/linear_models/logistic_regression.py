from dolcestat.metrics import ClassificationAnalyzer
from dolcestat.optimization.loss_and_activation import sigmoid

from .abstract import GLMAbstract


class LogisticRegression(GLMAbstract):
    """Binary logistic regression (Bernoulli GLM with logit link).

    Parameters
    ----------
    optimizer : str or GradientDescent/NewtonMethod, default "gradient descent"
        How to fit. One of "gradient descent" (mini-batch Nesterov GD) or
        "newton"; or a data-less optimizer instance. There is no closed-form
        solution for logistic regression.
    """

    _loss_function = "bce"
    _activation = staticmethod(sigmoid)
    _analyzer_class = ClassificationAnalyzer
    _allowed_optimizers = {"gradient descent", "newton"}

    def __init__(self, optimizer="gradient descent"):
        super().__init__(optimizer)
