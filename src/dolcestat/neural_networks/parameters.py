"""Stores parameters class and parameters initializers"""

from abc import ABC, abstractmethod

import numpy as np


class Parameters:
    """
    Wraps a parameter value (weights or biases) together with its gradient.

    Args:
        value: initial parameter value (numpy array)
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

    def zero_grad(self):
        "Zeroes the stored gradient. Called at the start of each epoch."
        self.grad = np.zeros_like(self.value, dtype=float)


class ParametersInitializer(ABC):
    """Abstract base class for weights and biases initialization strategies."""

    @abstractmethod
    def initialize(self, input_size: int, output_size: int):
        """
        Initializes weights and biases.

        Args:
            input_size: number of input features
            output_size: number of output features

        Returns:
            Tuple of Parameters: (weights, biases)
        """
        pass


class ZeroParametersInitializer(ParametersInitializer):
    """Initializes weights and biases to zero."""

    def initialize(self, input_size: int, output_size: int):
        """
        Initializes weights and biases to zero.

        Args:
            input_size: number of input features
            output_size: number of output features

        Returns:
            Tuple of Parameters: (weights, biases)
        """
        W_value = np.zeros((input_size, output_size), dtype=float)
        b_value = np.zeros(output_size, dtype=float)
        return Parameters(W_value), Parameters(b_value)


class GaussianInitializer(ParametersInitializer):
    """Initializes weights and biases by sampling from a Gaussian distribution."""

    def __init__(self, mean=0.0, std=1.0):
        """
        Args:
            mean: mean of the Gaussian distribution
            std: standard deviation of the Gaussian distribution
        """
        self.mean = mean
        self.std = std

    def initialize(self, input_size: int, output_size: int):
        """
        Initializes weights and biases from N(mean, std).

        Args:
            input_size: number of input features
            output_size: number of output features

        Returns:
            Tuple of Parameters: (weights, biases)
        """
        W_value = np.random.normal(self.mean, self.std, (input_size, output_size))
        b_value = np.random.normal(self.mean, self.std, output_size)
        return Parameters(W_value), Parameters(b_value)


class XavierInitializer(ParametersInitializer):
    """
    Initializes weights and biases using Xavier/Glorot initialization,
    scaled by the sum of input and output sizes. Suited to layers with
    sigmoid or tanh activations.
    """

    def __init__(self, distribution="normal"):
        """
        Args:
            distribution: sampling distribution, either "normal" or "uniform"
        """
        self.distribution = distribution

    def initialize(self, input_size: int, output_size: int):
        """
        Initializes weights and biases using Xavier/Glorot initialization.

        Args:
            input_size: number of input features
            output_size: number of output features

        Returns:
            Tuple of Parameters: (weights, biases)
        """
        io_size = input_size + output_size
        if self.distribution == "normal":
            W_value = np.random.normal(
                0, np.sqrt(2.0 / io_size), (input_size, output_size)
            )
            b_value = np.random.normal(0, np.sqrt(2.0 / io_size), output_size)
        elif self.distribution == "uniform":
            limit = np.sqrt(6.0 / io_size)
            W_value = np.random.uniform(-limit, limit, (input_size, output_size))
            b_value = np.random.uniform(-limit, limit, output_size)
        else:
            raise ValueError("Invalid distribution type. Choose 'normal' or 'uniform'.")
        return Parameters(W_value), Parameters(b_value)


class HeInitializer(ParametersInitializer):
    """
    Initializes weights and biases using He initialization, scaled by the
    input size. Suited to layers with ReLU-family activations.
    """

    def __init__(self, distribution="normal"):
        """
        Args:
            distribution: sampling distribution, either "normal" or "uniform"
        """
        self.distribution = distribution

    def initialize(self, input_size: int, output_size: int):
        """
        Initializes weights and biases using He initialization.

        Args:
            input_size: number of input features
            output_size: number of output features

        Returns:
            Tuple of Parameters: (weights, biases)
        """
        if self.distribution == "normal":
            W_value = np.random.normal(
                0, np.sqrt(2.0 / input_size), (input_size, output_size)
            )
            b_value = np.random.normal(0, np.sqrt(2.0 / input_size), output_size)
        elif self.distribution == "uniform":
            limit = np.sqrt(6.0 / input_size)
            W_value = np.random.uniform(-limit, limit, (input_size, output_size))
            b_value = np.random.uniform(-limit, limit, output_size)
        else:
            raise ValueError("Invalid distribution type. Choose 'normal' or 'uniform'.")
        return Parameters(W_value), Parameters(b_value)
