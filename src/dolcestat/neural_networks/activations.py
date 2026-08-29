"""Stores activation function classes"""

from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    """Abstract base class for activation functions."""

    @abstractmethod
    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies the activation function.

        Args:
            Z: pre-activation values (numpy array)
            training: whether to cache values needed for the backward pass

        Returns:
            Activated values A
        """
        pass

    @abstractmethod
    def backward(self, dL_dA: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the activation function.

        Args:
            dL_dA: gradient of the loss with respect to A

        Returns:
            Gradient of the loss with respect to Z (dL/dZ)
        """
        pass


class LU(ActivationFunction):
    """Linear (identity) activation function."""

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies the identity function.

        Args:
            Z: pre-activation values (numpy array)
            training: whether to cache values needed for the backward pass

        Returns:
            Activated values A
        """
        return Z

    def backward(self, dL_dA: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the identity function.

        Args:
            dL_dA: gradient of the loss with respect to A

        Returns:
            Gradient of the loss with respect to Z (dL/dZ)
        """
        return dL_dA


class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies the sigmoid function.

        Args:
            Z: pre-activation values (numpy array)
            training: whether to cache the output needed for the backward pass

        Returns:
            Activated values A
        """
        A = 1 / (1 + np.exp(-Z))
        if training:
            self.A = A
        return A

    def backward(self, dL_dA: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the sigmoid function.

        Args:
            dL_dA: gradient of the loss with respect to A

        Returns:
            Gradient of the loss with respect to Z (dL/dZ)
        """
        return dL_dA * self.A * (1 - self.A)


class ReLU(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function."""

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies the ReLU function.

        Args:
            Z: pre-activation values (numpy array)
            training: whether to cache the input needed for the backward pass

        Returns:
            Activated values A
        """
        if training:
            self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dL_dA: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the ReLU function.

        Args:
            dL_dA: gradient of the loss with respect to A

        Returns:
            Gradient of the loss with respect to Z (dL/dZ)
        """
        return dL_dA * (self.Z > 0)
