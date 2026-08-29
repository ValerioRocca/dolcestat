"""Stores the parameter update rules used to train networks.

These are *not* the optimizers in ``dolcestat.optimization``. That package's
``GradientDescent`` owns the training data, the iteration loop, the batching and
the gradient computation, and is driven by calling ``fit()``. The optimizer here
owns exactly one thing — the update rule — and is driven by calling
``step(parameters)`` after backpropagation has already deposited the gradients.

The split is what lets a network train: the loop lives in ``NeuralNetwork.fit``,
which knows how to backpropagate, and the optimizer only decides how far to move
once ``.grad`` is filled in. Batching is likewise not the optimizer's business;
it belongs to the injected ``Sampler``.
"""

from abc import ABC, abstractmethod

from dolcestat.neural_networks.input_validation import validate_alpha


class Optimizer(ABC):
    """Abstract base class for parameter update rules."""

    @abstractmethod
    def step(self, parameters: list) -> None:
        """
        Updates every parameter from the gradient stored on it.

        Args:
            parameters: list of Parameters carrying the gradients accumulated
                by the backward pass. Each one's ``value`` is updated in place
                and its ``grad`` is left untouched (the training loop zeroes it).
        """


class GradientDescent(Optimizer):
    """
    Plain gradient descent: ``w = w - alpha * dL/dw``.

    Deliberately minimal — a learning rate and nothing else. There is no
    momentum and no batching strategy, so the class is stateless between calls
    and one instance can safely train several networks.

    Whether a step is "batch", "stochastic" or "mini-batch" gradient descent is
    decided by the ``Sampler`` handed to ``fit``, not here: this class only ever
    sees the gradients a batch produced.
    """

    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha: learning rate; must be in (0, 1)
        """
        validate_alpha(alpha)
        self.alpha = alpha

    def step(self, parameters: list) -> None:
        """
        Moves every parameter one step down its gradient.

        Args:
            parameters: list of Parameters carrying the gradients accumulated
                by the backward pass
        """
        for parameter in parameters:
            # Rebinding rather than ``-=`` keeps an integer-valued parameter
            # from silently truncating the update to an integer.
            parameter.value = parameter.value - self.alpha * parameter.grad
