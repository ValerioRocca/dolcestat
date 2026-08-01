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
    """
    Linear Unit: the identity activation, i.e. ReLU without the rectification.

    The name deliberately pairs with ``ReLU``. Used on the output layer of a
    regression model, where the prediction is the linear predictor itself.
    """

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
    """
    Sigmoid activation function.

    The output is clipped to ``[eps, 1 - eps]``, so this activation never
    returns exactly 0 or 1. That keeps ``A(1 - A)`` bounded below in
    ``backward``, and it is what lets a downstream BinaryCrossEntropy divide by
    the very same ``A(1 - A)`` this class multiplies back (see the contract on
    ``BinaryCrossEntropy``). For a probability model, never returning a
    degenerate 0/1 probability is arguably the right behaviour anyway.
    """

    def __init__(self, eps: float = 1e-12):
        """
        Args:
            eps: clipping margin used to keep the output away from 0 and 1
        """
        self.eps = eps

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies the sigmoid function.

        Args:
            Z: pre-activation values (numpy array)
            training: whether to cache the output needed for the backward pass

        Returns:
            Activated values A, clipped to [eps, 1 - eps]
        """
        A = 1 / (1 + np.exp(-Z))

        # Clip here, once, and cache exactly the array that is returned. Caching
        # a different array than the one handed downstream is what used to make
        # the sigmoid+BCE gradients cancel inexactly at saturation.
        A = np.clip(A, self.eps, 1 - self.eps)

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


class Step(ActivationFunction):
    """
    Heaviside step: 1 where Z > 0, else 0. The original threshold unit.

    Its true derivative is zero everywhere it exists, which would freeze every
    parameter behind it. ``backward`` therefore passes the incoming gradient
    through **unchanged** — the "straight-through" convention. That is not a
    chain-rule derivative and is not claimed to be one; it is the modelling
    choice that turns Rosenblatt's rule into an ordinary gradient step, and it
    is why the perceptron could be trained at all before backpropagation
    existed. Composed with ``PerceptronCriterion`` it reproduces the classic
    rule exactly.
    """

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies the step function.

        Args:
            Z: pre-activation values (numpy array)
            training: unused; the straight-through backward caches nothing

        Returns:
            Activated values A, in {0.0, 1.0}
        """
        return (Z > 0).astype(float)

    def backward(self, dL_dA: np.ndarray) -> np.ndarray:
        """
        Args:
            dL_dA: gradient of the loss with respect to A

        Returns:
            dL_dA unchanged (straight-through; see the class docstring)
        """
        return dL_dA


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
