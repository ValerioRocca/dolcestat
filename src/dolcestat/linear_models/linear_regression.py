"""Stores the linear regression model."""

import numpy as np

from dolcestat.core.activations import LU
from dolcestat.core.losses import MeanSquaredError
from dolcestat.metrics import LinearRegressionAnalyzer

from .abstract import GLMAbstract


class LinearRegression(GLMAbstract):
    """Ordinary linear regression (Gaussian GLM with identity link).

    As a one-layer network: ``Sequential(DenseLayer(LU(), n_features, 1))``
    trained against ``MeanSquaredError``.

    Parameters
    ----------
    optimizer : str or Optimizer, default "closed form"
        How to fit. One of "closed form" (normal equation), "gradient descent"
        or "newton"; or an Optimizer instance to drive the training loop with.
    sampler : Sampler, optional
        Batching strategy for the gradient-descent path.
    """

    _loss_class = MeanSquaredError
    _activation_class = LU
    _analyzer_class = LinearRegressionAnalyzer
    _allowed_optimizers = {"closed form", "gradient descent", "newton"}

    def __init__(self, optimizer="closed form", sampler=None):
        super().__init__(optimizer, sampler)

    def _solve_closed_form(self, X_augmented, y):
        """
        Solves the normal equation w = (XtX)^-1 Xt y.

        Args:
            X_augmented: design matrix with a trailing column of ones
            y: targets

        Returns:
            Flat weight vector, intercept last
        """
        return np.matmul(
            np.linalg.inv(np.matmul(X_augmented.T, X_augmented)),
            np.matmul(X_augmented.T, y),
        )

    def _newton_gradient(self, X_augmented, y, predictions):
        """
        Returns:
            dL/dw = (2/n) Xt (p - y) for the squared-error loss
        """
        n = len(y)
        return (2 / n) * np.matmul(X_augmented.T, predictions - y)

    def _newton_hessian(self, X_augmented, y, predictions):
        """
        Returns:
            d2L/dw2 = (2/n) Xt X. Constant: the squared-error loss is quadratic
            in w, which is why Newton solves it in a single step.
        """
        n = len(y)
        return (2 / n) * np.matmul(X_augmented.T, X_augmented)

    def predict(self, data):
        """
        A LinearRegressionAnalyzer over the model's predictions for ``data``.

        Args:
            data: data to predict on (DolceSet)

        Returns:
            LinearRegressionAnalyzer, which also needs the predictor count
            (excluding the intercept) for adjusted R².
        """
        return LinearRegressionAnalyzer(
            data.y, self._predict_values(data), n_features=data.X.shape[1]
        )
