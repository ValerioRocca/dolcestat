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
        (mini-batch Nesterov GD) or "newton"; or a data-less optimizer instance.
    """

    _loss_function = "mse"
    _activation = staticmethod(identity)
    _analyzer_class = LinearRegressionAnalyzer
    _allowed_optimizers = {"closed form", "gradient descent", "newton"}

    def __init__(self, optimizer="closed form"):
        super().__init__(optimizer)

    def _resolve_optimizer(self, data):
        # Closed form is linear-only and skips the optimizer entirely: solve the
        # normal equation w = (XᵀX)⁻¹ Xᵀy on the bias-augmented design matrix and
        # store the weights so predict() works without an optimizer object.
        if self.optimizer == "closed form":
            X = np.column_stack((data.X, np.ones(data.X.shape[0])))
            y = data.y
            self.weights = np.matmul(
                np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y)
            )
            return None

        # Gradient descent / Newton / instance: defer to the shared dispatch.
        return super()._resolve_optimizer(data)

    def predict(self, data):
        # LinearRegressionAnalyzer also needs the predictor count (excluding the
        # intercept the model appends) to compute adjusted R² and the like.
        return LinearRegressionAnalyzer(
            data.y, self._predict_values(data), n_features=data.X.shape[1]
        )
