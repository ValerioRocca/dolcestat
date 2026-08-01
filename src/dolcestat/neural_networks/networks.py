"""Stores neural network container classes."""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.neural_networks.losses import BinaryCrossEntropy, Loss
from dolcestat.neural_networks.samplers import MiniBatchIteratingSampler, Sampler
from dolcestat.optimization.base import BaseOptimizer
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
        loss: Loss = BinaryCrossEntropy(),
        n_epochs: int = 100,
        sampler: Sampler = MiniBatchIteratingSampler(),
        optimizer: BaseOptimizer = GradientDescent(),
    ) -> None:
        """
        Trains the network on ``data`` via mini-batch gradient descent.

        Args:
            data: training data (DolceSet)
            loss: loss function used to compare predictions and targets
            n_epochs: number of passes over the (sampled) data
            sampler: strategy used to split data into (mini-)batches each epoch
            optimizer: optimizer used to update the layers' parameters
        """

        for epoch in range(n_epochs):
            # F1. Extract one or more (mini-)batches depending on the chosen strategy
            batched_data = sampler.sample(data)

            for batch in batched_data:
                X_batch = batch.X
                y_batch = batch.y

                # F2. Zero the gradients (from previous iterations). Gradients for this
                #     epoch will be computed during step F6.
                for p in self.get_parameters():
                    p.zero_grad()

                # F3. Forward pass: compute the model's predictions and caches, for
                #     each layer, its input (used to compute dL/dW) and (if applicable)
                #     sigmoid output (used to compute dz(l+1)/da(l) in the backward step).
                logits = self.forward(X_batch)

                # F4. Based on the predictions, compute the loss and caches true and
                #     predicted values for the backward step.
                loss_value = loss.forward(logits, y_batch)

                # F5. Loss backward pass: compute dL/da(l) - with l output layer -
                #     using the true/predicted values cached in F4.
                grad = loss.backward()

                # F6. Model backward pass: for each layer, this function does two things:
                #     1. Propagate the gradient back to the previous layer via chain rule:
                #        dL/dz(l) = dL/dz(l+1) dz(l+1)/da(l) da(l)/dz(l) where:
                #        - dL/dz(l+1) is just dL/dz passed from the layer above.
                #        - dz(l+1)/da(l) is the matrix [weights(l+1), bias(l+1)]
                #        - da(l)/dz(l) is the derivative of the activation function (this
                #          is why we cached the sigmoid output in F3).
                #        Note. The output layer of course does not have a next layer.
                #        For the output layer, dL/dz(l) = dL/da(l) da(l)/dz(l), where
                #        dL/da(l) is computed in F5.
                #     2. Compute the gradients of the layer's parameters (weights and bias)
                #        again via chain rule: dL/dW(l) = dL/dz(l) dz(l)/dW(l) where:
                #        - dL/dz(l) is computed in the previous step.
                #        - dz(l)/dW(l) is just the input to the layer (this is why we cached
                #          it in F3).
                #        The gradients are stored in layer's parameters object, enabling an
                #        the optimizer to simply retrieve themin the next step (F7).
                self.backward(grad)

                # F7. Update the models's parameters via chosen optimizer (default is
                #     mini-batch Nesterov gradient descent).
                optimizer.step(self.get_parameters())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the model's predictions for ``X``.

        Args:
            X: input features (numpy array)

        Returns:
            Network output with ``training=False``
        """
        return self.forward(X, training=False)
