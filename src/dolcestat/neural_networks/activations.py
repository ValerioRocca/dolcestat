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
        # a different array than the one handed downstream is what makes the
        # sigmoid+BCE gradients cancel inexactly at saturation: the loss would
        # divide by a clipped A while this class multiplied back an unclipped
        # A(1 - A) that had already collapsed to 0.
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


class Softplus(ActivationFunction):
    """
    Softplus: ``eps + log(1 + exp(Z))``, a smooth, strictly positive output.

    Where ``Sigmoid`` maps the real line onto a probability, this maps it onto a
    positive scale -- so it is the activation for a head that must emit a
    variance, a rate or any other non-negative quantity. Paired with
    ``GaussianNegativeLogLikelihood``'s variance head, it is what guarantees the
    loss never takes the log of, or divides by, a negative number.

    The ``eps`` floor follows the same reasoning as ``Sigmoid``'s clipping: the
    constraint is enforced **here**, once, so a downstream loss can divide by the
    very array this class returns instead of clipping a private copy and then
    multiplying a different number back through the chain rule.

    Choosing eps. It is not only a guard against exactly zero -- it is what bounds
    the gradient a variance head can receive. ``dL/d(sigma^2)`` under a Gaussian
    likelihood carries a ``1/(2 sigma^4)`` term, so a head that collapses toward
    the floor sees a gradient of order ``1/eps^2``. The default is deliberately
    much larger than ``Sigmoid``'s: for a variance head on a standardized target,
    ``eps=1e-3`` is a reasonable floor, and 1e-6 is already generous.
    """

    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: floor added to the output, keeping it strictly positive. Raise it
                (e.g. 1e-3) on a variance head to bound the 1/(2 sigma^4) gradient.
        """
        self.eps = eps

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Applies the softplus function.

        Args:
            Z: pre-activation values (numpy array)
            training: whether to cache the input needed for the backward pass

        Returns:
            Activated values A, strictly greater than eps
        """
        # logaddexp(0, Z) is log(1 + exp(Z)) computed without ever forming
        # exp(Z), which would overflow for Z of a few hundred.
        if training:
            self.Z = Z
        return self.eps + np.logaddexp(0.0, Z)

    def backward(self, dL_dA: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the softplus function.

        Args:
            dL_dA: gradient of the loss with respect to A

        Returns:
            Gradient of the loss with respect to Z (dL/dZ)
        """
        # d/dZ [eps + log(1 + exp(Z))] = sigmoid(Z); the eps is a constant and
        # drops out. exp(-logaddexp(0, -Z)) is sigmoid(Z) written so that a very
        # negative Z underflows to 0 rather than overflowing exp(-Z) to inf.
        return dL_dA * np.exp(-np.logaddexp(0.0, -self.Z))


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
