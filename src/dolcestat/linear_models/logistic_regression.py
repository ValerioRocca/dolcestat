"""Stores the logistic regression model."""

import numpy as np

from dolcestat.core.activations import Sigmoid
from dolcestat.core.losses import BinaryCrossEntropy
from dolcestat.metrics import ClassificationAnalyzer

from .abstract import GLMAbstract


class LogisticRegression(GLMAbstract):
    """Binary logistic regression (Bernoulli GLM with logit link).

    As a one-layer network: ``Sequential(DenseLayer(Sigmoid(), n_features, 1))``
    trained against ``BinaryCrossEntropy``.

    Parameters
    ----------
    optimizer : str or Optimizer, default "gradient descent"
        How to fit. Either "gradient descent" or "newton"; or an Optimizer
        instance to drive the training loop with. There is no closed-form
        solution for logistic regression.
    sampler : Sampler, optional
        Batching strategy for the gradient-descent path.
    """

    _loss_class = BinaryCrossEntropy
    _activation_class = Sigmoid
    _analyzer_class = ClassificationAnalyzer
    _allowed_optimizers = {"gradient descent", "newton"}

    def __init__(self, optimizer="gradient descent", sampler=None):
        super().__init__(optimizer, sampler)

    def _newton_gradient(self, X_augmented, y, predictions):
        """
        Returns:
            dL/dw = (1/n) Xt (p - y) for the cross-entropy loss
        """
        n = len(y)
        return (1 / n) * np.matmul(X_augmented.T, predictions - y)

    def _newton_hessian(self, X_augmented, y, predictions):
        """
        Returns:
            d2L/dw2 = (1/n) Xt diag(p(1-p)) X — each row weighted by the
            variance of its predicted probability.
        """
        n = len(y)
        weights = predictions * (1 - predictions)
        return (1 / n) * np.matmul(X_augmented.T * weights, X_augmented)
