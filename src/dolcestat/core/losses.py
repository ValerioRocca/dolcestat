"""Stores loss function classes"""

from abc import ABC, abstractmethod

import numpy as np


def validate_matching_shapes(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """
    Checks that predictions and targets line up elementwise.

    Args:
        y_pred: predicted values
        y_true: true values

    Raises:
        ValueError: if the shapes differ, which NumPy would otherwise broadcast
            into an (n, n) matrix and silently return a meaningless loss.

    Note. Losses used to reshape ``y_true`` to match instead. That papered over
    the 1-D/2-D target question rather than answering it; ``Trainer`` now owns
    that conversion at a single boundary, so a mismatch reaching this point is a
    real bug and is reported as one.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred has shape {y_pred.shape} but y_true has shape "
            f"{y_true.shape}; they must match elementwise."
        )


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True
    ) -> float:
        """
        Computes the loss.

        Args:
            y_pred: predicted values (numpy array)
            y_true: true values (numpy array), same shape as y_pred
            training: whether to cache the values needed for the backward pass.
                Pass False to evaluate the loss as a pure metric, with no
                backward-pass cache left behind.

        Returns:
            Scalar loss value
        """

    @abstractmethod
    def backward(self) -> np.ndarray:
        """
        Returns:
            dL/d(y_pred), evaluated at the y_pred that forward was handed, and
            of the same shape.
        """


class PerceptronCriterion(Loss):
    """
    The perceptron's 0/1 error, paired with Rosenblatt's update direction.

    ``forward`` reports the **misclassification rate**, which is what you want
    to watch converge. ``backward`` returns ``(y_pred - y_true)/n``: zero on
    every correctly classified example, ±1/n on a mistake.

    These are deliberately not a matched forward/backward pair in the autodiff
    sense — the 0/1 loss has zero gradient almost everywhere, so nothing could
    be learned from its true derivative. What ``backward`` returns is the
    gradient of the *perceptron criterion*, the quantity Rosenblatt's rule
    actually descends. Composed with a ``Step`` activation and plain gradient
    descent over one sample at a time, it reproduces::

        w = w + alpha * (y - y_pred) * x

    which is the 1958 learning rule, recovered rather than special-cased.
    """

    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True
    ) -> float:
        """
        Computes the misclassification rate.

        Args:
            y_pred: predicted labels in {0, 1} (numpy array)
            y_true: true labels in {0, 1} (numpy array), same shape as y_pred
            training: whether to cache y_pred/y_true for the backward pass

        Returns:
            Fraction of examples classified incorrectly
        """
        validate_matching_shapes(y_pred, y_true)

        if training:
            self.y_pred = y_pred
            self.y_true = y_true

        return float(np.mean(y_pred != y_true))

    def backward(self) -> np.ndarray:
        """
        Returns:
            (y_pred - y_true)/n — zero where the prediction was right
        """
        n = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / n


class MeanSquaredError(Loss):
    """
    Squared-error loss for regression: mean((y_pred - y_true)^2).

    Pairs with an ``LU`` output activation, exactly as BinaryCrossEntropy pairs
    with ``Sigmoid``. Composing this class with ``LU`` and a ``DenseLayer``
    reproduces the textbook dL/dw = (2/n) X^T (y_pred - y_true): the 2/n lives
    here, the X^T comes from the layer.
    """

    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True
    ) -> float:
        """
        Computes the mean squared error.

        Args:
            y_pred: predicted values (numpy array)
            y_true: true values (numpy array), same shape as y_pred
            training: whether to cache y_pred/y_true for the backward pass

        Returns:
            Scalar loss value
        """

        # 1. Predictions and targets must already line up (Trainer guarantees it).
        validate_matching_shapes(y_pred, y_true)

        # 2. Cache for the backward pass; skipped for pure-metric calls.
        if training:
            self.y_pred = y_pred
            self.y_true = y_true

        # 3. Compute the loss value.
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        """
        Returns:
            dL/d(y_pred) = 2(y_pred - y_true)/n, same shape as forward's y_pred
        """
        n = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / n


class BinaryCrossEntropy(Loss):
    """
    y_pred are probabilities in (0, 1), e.g. output of a Sigmoid activation.

    Gradient contract. ``backward`` returns dL/d(y_pred) evaluated at *exactly*
    the y_pred ``forward`` was handed — it divides by ``y_pred(1 - y_pred)``,
    unmodified. An activation's ``backward`` multiplies by dA/dZ at that same A.
    Composing this class with ``Sigmoid`` therefore cancels exactly, yielding
    dL/dZ = (A - y)/n for every input, saturated or not.

    This is why the clipping lives in ``Sigmoid`` and not here: clipping y_pred
    locally *and* caching the clipped copy would make this class divide by a
    different number than Sigmoid multiplies back, and a saturated unit would
    then get a gradient scaled toward zero instead of the correct ±1/n.

    Keeping the two classes separate rather than fusing them (PyTorch's
    BCEWithLogitsLoss) is deliberate: sigmoid and cross-entropy composing is the
    lesson. The cost is that the intermediate dL/dA can reach ~1/eps, which is
    far from float64 overflow and exact in the product, but is the reason
    production libraries fuse the pair.
    """

    def __init__(self, eps: float = 1e-12):
        """
        Args:
            eps: clipping margin applied to the log arguments only, to avoid
                log(0) in the returned scalar. It never touches the cached
                y_pred used by backward.
        """
        self.eps = eps

    def forward(
        self, y_pred: np.ndarray, y_true: np.ndarray, training: bool = True
    ) -> float:
        """
        Computes the binary cross-entropy loss.

        Args:
            y_pred: predicted probabilities in (0, 1) (numpy array)
            y_true: true labels in {0, 1} (numpy array), same shape as y_pred
            training: whether to cache y_pred/y_true for the backward pass

        Returns:
            Scalar loss value
        """

        # 1. Predictions and targets must already line up (Trainer guarantees it).
        validate_matching_shapes(y_pred, y_true)

        # 2. Cache the arrays exactly as handed in, so backward divides by the
        #    same y_pred the upstream activation will multiply back. Skipped for
        #    pure-metric calls, which leave no backward-pass cache behind.
        if training:
            self.y_pred = y_pred
            self.y_true = y_true

        # 3. Clip the log arguments only — locally, for the returned scalar — to
        #    avoid log(0). The cached y_pred above stays unmodified.
        y_pred_safe = np.clip(y_pred, self.eps, 1 - self.eps)
        return -np.mean(
            y_true * np.log(y_pred_safe) + (1 - y_true) * np.log(1 - y_pred_safe)
        )

    def backward(self) -> np.ndarray:
        """
        Returns:
            dL/d(y_pred), same shape as forward's y_pred. Assumes y_pred lies
            strictly inside (0, 1) — see the gradient contract on the class.
        """
        n = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * n)
