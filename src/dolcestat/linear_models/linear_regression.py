"""Ordinary least squares linear regression."""

import numpy as np

from dolcestat.metrics import LinearRegressionAnalyzer
from dolcestat.optimization.loss_and_activation import identity

from .abstract import GLMAbstract


class LinearRegression(GLMAbstract):
    """Ordinary linear regression (Gaussian GLM with identity link).

    Parameters
    ----------
    optimizer : str or GradientDescent/NewtonMethod, default "closed form"
        How to fit. One of "closed form" (normal equation), "gradient descent"
        (mini-batch Nesterov GD) or "newton"; or an optimizer instance holding
        the settings you want (it is copied, never modified).
    """

    _loss_function = "mse"
    _activation = staticmethod(identity)
    _analyzer_class = LinearRegressionAnalyzer
    _allowed_optimizers = {"closed form", "gradient descent", "newton"}

    def __init__(self, optimizer="closed form"):
        super().__init__(optimizer)

    def _build_optimizer(self):
        # Closed form is linear-only and skips the optimizer entirely.
        if self.optimizer == "closed form":
            return None
        return super()._build_optimizer()

    def _fit_closed_form(self, data):
        # Solve the normal equation (XᵀX) w = Xᵀy on the bias-augmented design
        # matrix and store the weights so predict() works without an optimizer.
        X = np.column_stack((data.X, np.ones(data.X.shape[0])))
        try:
            self.weights = np.linalg.solve(X.T @ X, X.T @ data.y)
        except np.linalg.LinAlgError:
            raise ValueError(
                "The normal equation has no unique solution: some features are "
                "perfectly collinear (or constant). Drop the redundant ones, or "
                "fit with 'gradient descent'."
            ) from None

    def predict(self, data):
        """A LinearRegressionAnalyzer over the model's predictions for ``data``.

        ``data`` needs no target. The analyzer also receives the predictor count
        (excluding the intercept the model appends), which adjusted R² needs.
        """
        return LinearRegressionAnalyzer(
            data.y, self._predict_values(data), n_features=data.X.shape[1]
        )
