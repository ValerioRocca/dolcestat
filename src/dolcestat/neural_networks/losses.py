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

    Note. ``DolceSet.y`` is 1-D while a single-output layer predicts an (n, 1)
    column. ``NeuralNetwork.fit`` reshapes the targets once, at the top of the
    loop, so a mismatch reaching this point is a real bug and is reported as one
    rather than reshaped away here.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"y_pred has shape {y_pred.shape} but y_true has shape "
            f"{y_true.shape}; they must match elementwise."
        )


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def forward(self, y_pred, y_true: np.ndarray, training: bool = True) -> float:
        """
        Computes the loss.

        Args:
            y_pred: predicted values (numpy array), or a tuple of them for a
                loss that consumes several predictions at once
            y_true: true values (numpy array), same shape as y_pred
            training: whether to cache the values needed by the backward pass.
                Pass False to use the loss as a pure metric, leaving no stale
                cache behind.

        Returns:
            Scalar loss value
        """

    @abstractmethod
    def backward(self):
        """
        Returns:
            dL/d(y_pred), same shape as forward's y_pred. A loss that consumes
            several predictions returns a **tuple** instead, one gradient per
            prediction and in the same order — which is what a MultiHead
            network's backward expects.
        """


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

        # 1. Predictions and targets must already line up (fit guarantees it).
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


class GaussianNegativeLogLikelihood(Loss):
    """
    Negative log-likelihood of ``y ~ N(mu, sigma^2)`` with a per-sample variance.

    This is the loss for **heteroscedastic** regression: the model predicts not
    just where the target sits but how noisy it is there. It therefore consumes
    *two* predictions rather than one, and pairs with a ``MultiHead`` network
    whose heads carry ``LU`` (mu, the whole real line) and ``Softplus``
    (sigma^2, strictly positive), the same way ``MeanSquaredError`` pairs with
    ``LU`` and ``BinaryCrossEntropy`` pairs with ``Sigmoid``.

    Per sample, writing ``r = y - mu``::

        L_i        = 0.5 log(2 pi) + 0.5 log(sigma^2) + r^2 / (2 sigma^2)
        dL_i/dmu   = (mu - y) / sigma^2
        dL_i/dvar  = 1/(2 sigma^2) - r^2/(2 sigma^4) = (1 - r^2/sigma^2)/(2 sigma^2)

    Two things fall out of those derivatives, and they are the point of the class.

    Setting ``dL/dvar = 0`` gives ``sigma^2 = r^2``. The loss is minimized when the
    predicted variance equals the squared residual, which is exactly the maximum
    likelihood estimate -- that is *why* a second head, given this loss and nothing
    else, learns the noise scale.

    Freezing ``sigma^2 = 1`` reduces ``dL/dmu`` to ``(mu - y)/n``, precisely half
    of ``MeanSquaredError.backward``'s ``2(mu - y)/n``. Squared error **is** this
    loss under a homoscedastic unit variance, up to a factor of 2 and additive
    constants. The new content is the ``1/sigma^2`` factor: each residual is
    reweighted by the model's own confidence, so points the network believes are
    noisy pull on the mean head less. That is inverse-variance (generalized least
    squares) weighting, arrived at rather than imposed.

    Unlike the other losses here, the value is a log-density, not a distance: it
    is **not** bounded below by zero and a training curve that goes negative is
    working, not broken. For the same reason a ``tol`` tuned against a squared
    error does not carry over.

    Fitting notes. Two heads coupled through one likelihood optimize less kindly
    than a single regression head, and the failure is quiet -- a variance head that
    gives up predicts one constant, and the run merely converges to the
    homoscedastic answer instead of erroring. In practice:

    - **Standardize the target.** ``HeInitializer`` starts the variance head near
      ``softplus(0) = 0.7``, which is a sensible prior only if the noise scale is
      order 1.
    - **Use a small learning rate.** Empirically ``alpha=0.01`` recovers the noise
      curve from every initialization tried, while an effective step near 0.5
      diverges outright on a plain sine-with-growing-noise problem.
    - **Warm up the mean.** Fitting the trunk and mean head against
      ``MeanSquaredError`` first gives the variance head honest residuals to
      explain. Because a ``Sequential`` holds layer objects by reference, this
      needs no extra machinery::

          Sequential(*net.trunk.layers, *net.heads[0].layers).fit(
              data, loss=MeanSquaredError(), n_epochs=150
          )
          net.fit(data, loss=GaussianNegativeLogLikelihood(), n_epochs=650)

    - **Keep batches reasonably large.** The variance head estimates a second
      moment, far noisier per sample than a mean.

    Judge the result by calibration, not by RMSE: roughly 68% of held-out targets
    should fall within one predicted sigma of mu, in *every* region of the input
    space. A homoscedastic fit can only manage that on average.
    """

    def forward(self, y_pred, y_true: np.ndarray, training: bool = True) -> float:
        """
        Computes the mean negative log-likelihood.

        Args:
            y_pred: tuple ``(mu, sigma^2)`` of predictions, each shaped like
                y_true -- i.e. what a two-headed network's forward returns
            y_true: true values (numpy array), same shape as each prediction
            training: whether to cache the values for the backward pass

        Returns:
            Scalar loss value

        Raises:
            ValueError: if y_pred is not a 2-tuple, if the shapes disagree, or if
                any predicted variance is non-positive.
        """

        # 1. A single array here means the model was a Sequential, not a
        #    MultiHead. Say so, rather than failing on an opaque unpack.
        if not isinstance(y_pred, (tuple, list)) or len(y_pred) != 2:
            raise ValueError(
                "GaussianNegativeLogLikelihood expects y_pred to be a "
                "(mu, sigma_squared) pair, as returned by a two-headed MultiHead "
                f"network, but got {type(y_pred).__name__}. A single-output "
                "Sequential cannot supply a variance."
            )
        mu, var = y_pred

        # 2. Both predictions must line up with the targets elementwise.
        validate_matching_shapes(mu, y_true)
        validate_matching_shapes(var, y_true)

        # 3. The likelihood is undefined for a non-positive variance, and NumPy
        #    would return a silent nan rather than complaining. Point at the
        #    activation, since a variance head wired to LU is the way this happens.
        if not np.all(var > 0):
            raise ValueError(
                "Predicted variances must be strictly positive, but the smallest "
                f"is {float(np.min(var))}. Give the variance head a Softplus "
                "activation: LU leaves it free to go negative."
            )

        # 4. Cache for the backward pass; skipped for pure-metric calls.
        if training:
            self.mu = mu
            self.var = var
            self.y_true = y_true

        # 5. Compute the loss value.
        return float(
            np.mean(0.5 * np.log(2 * np.pi * var) + (y_true - mu) ** 2 / (2 * var))
        )

    def backward(self) -> tuple:
        """
        Returns:
            ``(dL/dmu, dL/dvar)``, one gradient per head and each shaped like
            forward's corresponding prediction -- the pair a two-headed
            MultiHead's backward expects.
        """
        n = self.y_true.shape[0]
        resid = self.mu - self.y_true

        # Each residual is scaled by 1/var: the noisier the model thinks a point
        # is, the less it pulls on the mean.
        d_mu = resid / (n * self.var)

        # Zero exactly where var == resid^2, the MLE variance.
        d_var = (1.0 - resid**2 / self.var) / (2.0 * n * self.var)

        return d_mu, d_var


class BinaryCrossEntropy(Loss):
    """
    y_pred are probabilities in (0, 1), e.g. output of a Sigmoid activation.

    Gradient contract. ``backward`` returns dL/d(y_pred) evaluated at *exactly*
    the y_pred ``forward`` was handed -- it divides by ``y_pred(1 - y_pred)``,
    unmodified. An activation's ``backward`` multiplies by dA/dZ at that same A.
    Composing this class with ``Sigmoid`` therefore cancels exactly, yielding
    dL/dZ = (A - y)/n for every input, saturated or not.

    This is why the clipping lives in ``Sigmoid`` and not here: clipping y_pred
    locally *and* caching the clipped copy would make this class divide by a
    different number than Sigmoid multiplies back, and a saturated unit would
    then get a gradient scaled toward zero instead of the correct +/-1/n.

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

        # 1. Predictions and targets must already line up (fit guarantees it).
        validate_matching_shapes(y_pred, y_true)

        # 2. Cache the arrays exactly as handed in, so backward divides by the
        #    same y_pred the upstream activation will multiply back. Skipped for
        #    pure-metric calls, which leave no backward-pass cache behind.
        if training:
            self.y_pred = y_pred
            self.y_true = y_true

        # 3. Clip the log arguments only -- locally, for the returned scalar -- to
        #    avoid log(0). The cached y_pred above stays unmodified.
        y_pred_safe = np.clip(y_pred, self.eps, 1 - self.eps)
        return -np.mean(
            y_true * np.log(y_pred_safe) + (1 - y_true) * np.log(1 - y_pred_safe)
        )

    def backward(self) -> np.ndarray:
        """
        Returns:
            dL/d(y_pred), same shape as forward's y_pred. Assumes y_pred lies
            strictly inside (0, 1) -- see the gradient contract on the class.
        """
        n = self.y_true.shape[0]
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * n)
