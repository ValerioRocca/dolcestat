"""Stores loss function classes"""

from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the loss.

        Args:
            y_pred: predicted values (numpy array)
            y_true: true values (numpy array), same shape as y_pred

        Returns:
            Scalar loss value
        """

    @abstractmethod
    def backward(self) -> np.ndarray:
        """
        Returns:
            dL/d(y_pred), same shape as forward's y_pred
        """


class BinaryCrossEntropy(Loss):
    """y_pred are probabilities in (0, 1), e.g. output of a Sigmoid activation."""

    def __init__(self, eps: float = 1e-12):
        """
        Args:
            eps: clipping margin used to keep y_pred away from 0 and 1
        """
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss.

        Args:
            y_pred: predicted probabilities in (0, 1) (numpy array)
            y_true: true labels in {0, 1} (numpy array), same shape as y_pred

        Returns:
            Scalar loss value
        """

        # 1. Clip 0 and 1 to avoid log(0)
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)

        # 1b. Align y_true to y_pred's shape (e.g. DolceSet.y is 1-D while a
        #     single-output layer's predictions are a 2-D column) so the two
        #     combine elementwise instead of silently broadcasting to (n, n).
        y_true = np.reshape(y_true, y_pred.shape)

        # 2. Compute Loss value
        #    (also caches y_pred and y_true for backward step)
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self) -> np.ndarray:
        """
        Returns:
            dL/d(y_pred), same shape as forward's y_pred
        """
        n = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * n)
