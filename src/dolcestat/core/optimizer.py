"""Stores the optimizer protocol shared by every training loop."""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for first-order parameter update rules.

    An optimizer sees only a list of Parameters: it consumes each
    ``Parameters.grad`` and updates each ``Parameters.value``. It knows nothing
    about the model, the data, the loss or the batching — those belong to the
    Trainer. That separation is what lets a single optimizer serve a generalized
    linear model, a perceptron and a deep network without modification.

    Gradients are *not* zeroed here. The training loop owns that, which keeps
    the optimizer from having to remember the parameter list between calls.

    Note. Second-order methods do not fit this protocol: Newton needs the design
    matrix and a single joint Hessian over one flat parameter vector, and a list
    of independent per-parameter gradients can supply neither. Newton therefore
    stays a fitting strategy on the model rather than an Optimizer — see
    REFACTOR.md D10.
    """

    @abstractmethod
    def step(self, parameters: list) -> None:
        """
        Updates every parameter from the gradient stored on it.

        Args:
            parameters: list of Parameters, each carrying a ``.value`` and the
                ``.grad`` accumulated by the backward pass
        """
