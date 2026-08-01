"""Stores neural network container classes."""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.core.losses import Loss
from dolcestat.core.optimizer import Optimizer
from dolcestat.core.samplers import Sampler
from dolcestat.core.trainer import Trainer
from dolcestat.optimization.gradient_descent import GradientDescent
from dolcestat.preprocessing.core import DolceSet


class NeuralNetwork(ABC):
    """Abstract base class for neural network containers."""

    @abstractmethod
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Computes the Neural Network's output."""

    @abstractmethod
    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through the network.

        Args:
            grad_out: dL/d(output), same shape as forward's output

        Returns:
            dL/d(input), same shape as forward's input. Side effect:
            accumulates dL/d(param) into each Parameter.grad.
        """

    @abstractmethod
    def get_parameters(self) -> list:
        """
        Returns:
            List of Parameters held by the network.
        """

    @abstractmethod
    def fit(self, data: DolceSet, loss: Loss):
        """Fit the model to the data using the specified loss function."""


class Sequential(NeuralNetwork):
    """
    Feed-forward network that runs a fixed sequence of layers.

    Example of initialization.:
    ```python
    my_seq = Sequential(
        DenseLayer(input_size=32, activation=ReLU(), output_size=16),
        DenseLayer(input_size=16, activation=ReLU(), output_size=8),
        DenseLayer(input_size=8, activation=ReLU(), output_size=1)
    )
    ```
    """

    def __init__(self, *layers):
        """
        Args:
            *layers: Layer instances to run in sequence, input to output
        """
        self.layers = layers

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Runs ``X`` through each layer in sequence.

        Args:
            X: input features (numpy array)
            training: whether the forward pass is used for training

        Returns:
            Network output
        """
        for layer in self.layers:
            X = layer.forward(X, training=training)
        return X

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagates the gradient through each layer in reverse order.

        Args:
            grad_out: dL/d(output), same shape as forward's output

        Returns:
            dL/d(input), same shape as forward's input. Side effect:
            accumulates dL/d(param) into each layer's Parameter.grad.
        """
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out

    def get_parameters(self) -> list:
        """
        Returns:
            List of Parameters (weights and biases) across all layers.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def fit(
        self,
        data: DolceSet,
        loss: Loss,
        n_epochs: int = 100,
        sampler: Sampler = None,
        optimizer: Optimizer = None,
        **run_kwargs,
    ) -> "Sequential":
        """
        Trains the network on ``data``.

        This is a convenience wrapper: it assembles a Trainer and runs it. The
        loop itself lives in Trainer, so a network, a GLM and a perceptron all
        train through exactly one implementation. Build the Trainer yourself
        when you want to drive the loop directly.

        Args:
            data: training data (DolceSet)
            loss: loss function used to compare predictions and targets.
                Required: a general network has no sensible default, and
                defaulting to a classification loss would silently do the wrong
                thing for regression.
            n_epochs: number of passes over the (sampled) data
            sampler: strategy used to split data into (mini-)batches each epoch.
                Defaults to MiniBatchIteratingSampler().
            optimizer: optimizer used to update the layers' parameters.
                Defaults to GradientDescent().
            **run_kwargs: forwarded to Trainer.run (tol, max_steps,
                track_full_loss)

        Returns:
            self, with the run recorded in ``self.history``
        """

        # F1. Resolve the strategy defaults here rather than in the signature.
        #     Default arguments are evaluated once, at import, so every network
        #     trained with defaults would share one optimizer instance — and,
        #     once optimizers carry momentum buffers, leak momentum between
        #     unrelated networks.
        if optimizer is None:
            optimizer = GradientDescent()

        # F2. Hand the pieces to the loop and record the run.
        trainer = Trainer(model=self, loss=loss, optimizer=optimizer, sampler=sampler)
        self.history = trainer.run(data, n_epochs=n_epochs, **run_kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the model's predictions for ``X``.

        Args:
            X: input features (numpy array)

        Returns:
            Network output with ``training=False``
        """
        return self.forward(X, training=False)
