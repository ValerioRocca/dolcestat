"""Gradient descent in its batch, stochastic and mini-batch flavors."""

import math

import numpy as np

from .base import BaseOptimizer
from .input_validation import (
    validate_alpha,
    validate_batch_fraction,
    validate_flavor,
    validate_l1,
    validate_l2,
    validate_momentum_rate,
    validate_momentum_type,
)
from .utils import sample_rows_Xy


class GradientDescent(BaseOptimizer):
    """Gradient descent for linear (MSE) and logistic (BCE) objectives.

    Supports batch, stochastic and mini-batch updates, Polyak or Nesterov
    momentum, and L1/L2 penalties. After ``fit(data)`` the run is available
    through ``get_weights()`` and ``get_loss()``; ``predict(data)`` applies the
    final weights.
    """

    def __init__(
        self,
        *,
        alpha=0.01,
        n_iters=1000,
        tol=1e-6,
        loss_function=None,
        flavor="batch",
        batch_fraction=0.1,
        momentum_type=None,
        momentum_rate=0.9,
        l1=0.0,
        l2=0.0,
    ):
        """Gradient descent optimizer for linear/logistic models.

        Built from its settings alone; ``fit(data)`` runs the descent and
        returns the optimizer.

        Parameters
        ----------
        alpha : float, default 0.01
            Learning rate; must be in (0, 1).
        n_iters : int, default 1000
            Maximum number of iterations; must be positive.
        tol : float, default 1e-6
            Convergence tolerance: stop once the absolute loss change drops
            below this value.
        loss_function : {"mse", "bce", None}, default None
            Loss to optimize. Inferred from y at fit time when None.
        flavor : {"batch", "sgd", "mini_batch"}, default "batch"
            Which subset of rows to use per iteration.
        batch_fraction : float, default 0.1
            Fraction of rows per iteration for "mini_batch"; must be in (0, 1].
        momentum_type : {None, "polyak", "nesterov"}, default None
            Momentum variant. None disables momentum.
        momentum_rate : float, default 0.9
            Momentum coefficient in [0, 1); ignored when momentum_type is None.
        l1 : float, default 0.0
            L1 (Lasso) coefficient; must be non-negative. Adds ``l1 * |w|`` to
            the objective.
        l2 : float, default 0.0
            L2 (Ridge) coefficient; must be non-negative. Adds ``(l2 / 2) *
            ||w||^2`` to the objective.
        """

        # 1. Validate GD-specific input
        validate_alpha(alpha)
        validate_flavor(flavor)
        if flavor == "mini_batch":
            validate_batch_fraction(batch_fraction)
        validate_momentum_type(momentum_type)
        validate_momentum_rate(momentum_rate)
        validate_l1(l1)
        validate_l2(l2)

        # 2. Assign GD-specific attributes
        self.alpha = alpha
        self.flavor = flavor
        self.batch_fraction = batch_fraction
        self.momentum_type = momentum_type
        self.momentum_rate = momentum_rate
        self.mini_batch_size = None
        self.l1 = l1
        self.l2 = l2

        # 3. Shared validation and attribute setup
        super().__init__(n_iters, tol, loss_function)

    def _penalty(self, weights):
        # The regularization term of the objective, whose gradient fit() adds.
        # The bias is the last weight and stays unpenalized.
        w = weights[:-1]
        return self.l1 * np.sum(np.abs(w)) + 0.5 * self.l2 * np.sum(w**2)

    def fit(self, data):
        """Runs gradient descent on ``data`` and returns self."""

        self._load_training(data)
        # Derive the mini-batch size once the row count is known.
        self.mini_batch_size = max(1, math.ceil(self.batch_fraction * self.n_samples))

        for iter in range(self.n_iters):

            # 1. Current weights
            weights = self.get_weights(iteration=iter)

            # 2. Apply chosen GD strategy
            # 2a. Batch GD: whole dataset at each iteration
            if self.flavor == "batch":
                sub_X, sub_y = self.X, self.y

            # 2b. Stochastic GD: one row at each iteration
            elif self.flavor == "sgd":
                sub_X, sub_y = sample_rows_Xy(self.X, self.y, 1)

            # 2c. Mini-batch GD: a sample at each iteration
            else:
                sub_X, sub_y = sample_rows_Xy(self.X, self.y, self.mini_batch_size)

            # Stores a copy of weights for logging purposes
            pre_momentum_weights = weights

            # 3. Gradient on the selected rows (predictions must match sub_X).

            # 3a. Nesterov momentum: add weights acceleration to the prediction
            if self.momentum_type == "nesterov":
                prev_weights = self.get_weights(iteration=iter - 1)
                momentum = self.momentum_rate * (prev_weights - weights)
                weights = weights - momentum
            sub_pred = self._apply_activation_function(np.matmul(sub_X, weights))
            grad = self._compute_gradient(sub_pred, sub_X, sub_y)

            # 4. Add regularization to the gradient if applicable. The bias
            #    is the last weight and stays unpenalized.
            if self.l1 > 0 or self.l2 > 0:
                penalty = self.l1 * np.sign(weights) + self.l2 * weights
                penalty[-1] = 0.0
                grad += penalty

            # 5. Update weights based on gradients
            # 5a. Polyak: momentum is weights distance
            if self.momentum_type == "polyak":
                prev_weights = self.get_weights(iteration=iter - 1)
                momentum = self.momentum_rate * (prev_weights - weights)
                updated_weights = weights - self.alpha * grad - momentum
            # 5b. Nesterov: momentum is already added before
            # Note: if Nesterov, below weights are accelerated ones
            elif self.momentum_type == "nesterov" or self.momentum_type is None:
                updated_weights = weights - self.alpha * grad
            self.append_weights(updated_weights)

            # 6. Track full-batch predictions and loss so the convergence
            #    curve is comparable across flavors. The loss is the objective
            #    being minimised, so it includes the l1/l2 penalty.
            full_pred = self._apply_activation_function(
                np.matmul(self.X, pre_momentum_weights)
            )
            self.append_predictions(full_pred)
            loss = self._compute_loss(full_pred) + self._penalty(pre_momentum_weights)
            self.append_loss(loss)

            # 7. If converged, break GD
            if iter > 0:
                prev_loss = self.get_loss(iteration=iter - 1)
                delta_loss = abs(loss - prev_loss)
                if delta_loss < self.tol:
                    break

        self.is_fitted = True
        return self
