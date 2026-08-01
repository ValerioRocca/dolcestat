"""Stores layer classes"""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.core.activations import ActivationFunction
from dolcestat.core.parameters import HeInitializer, ParametersInitializer


class Layer(ABC):
    """Abstract base class for layers. Shape convention: (input_size, output_size)."""

    @abstractmethod
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Computes the layer's output."""

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the layer.

        Args:
            grad_out: dL/d(output), same shape as forward's output

        Returns:
            dL/d(input), same shape as forward's input. Side effect:
            accumulates dL/d(param) into each Parameter.grad.
        """

    def parameters(self) -> list:
        """
        Returns:
            List of Parameters owned by the layer (empty if none).
        """
        return []


class DenseLayer(Layer):
    """Fully-connected layer computing ``activation(X @ W + b)``."""

    def __init__(
        self,
        activation: ActivationFunction,
        input_size: int,
        output_size: int,
        parameters_initializer: ParametersInitializer = None,
    ):
        """
        Args:
            activation: activation function applied to the linear output
            input_size: number of input features
            output_size: number of output features
            parameters_initializer: strategy used to initialize weights and
                biases. Defaults to HeInitializer().
        """

        # 1. Weights initialization. The default initializer is built here, not
        #    in the signature: a default argument is evaluated once at import,
        #    so every layer would otherwise share a single initializer instance.
        if parameters_initializer is None:
            parameters_initializer = HeInitializer()
        self.W, self.b = parameters_initializer.initialize(input_size, output_size)

        # 2. Activation initialization
        self.activation = activation

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Computes ``activation(X @ W + b)``.

        Args:
            X: input features (numpy array)
            training: whether the forward pass is used for training

        Returns:
            Activated output
        """
        # 1. Cache the input for backward step (skipped for inference-only calls,
        #    e.g. predict(), so they don't clobber an in-progress training cache)
        if training:
            self.X = X

        # 2. Computes WX + b
        z = X @ self.W.value + self.b.value

        # 3. Applies the activation function (and if applicable caches the output
        #    for backward step)
        return self.activation.forward(z, training=training)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the activation and linear map.

        Args:
            grad_out: dL/d(output), same shape as forward's output

        Returns:
            dL/d(input), same shape as forward's input. Side effect:
            accumulates dL/dW into ``self.W.grad`` and dL/db into ``self.b.grad``.
        """
        grad_z = self.activation.backward(grad_out)
        self.W.grad += self.X.T @ grad_z
        self.b.grad += grad_z.sum(axis=0)
        return grad_z @ self.W.value.T

    def parameters(self) -> list:
        """
        Returns:
            List of Parameters: [weights, biases]
        """
        return [self.W, self.b]
