import math

import numpy as np

from .base import BaseOptimizer
from .input_validation import (
    validate_alpha,
    validate_batch_fraction,
    validate_flavor,
    validate_if_fitting_without_target,
    validate_momentum_rate,
    validate_momentum_type,
)
from .utils import sample_rows_Xy


class GradientDescent(BaseOptimizer):

    def __init__(
        self,
        data,
        alpha=0.01,
        n_iters=1000,
        tol=1e-6,
        loss_function=None,
        flavor="batch",
        batch_fraction=0.1,
        momentum_type=None,
        momentum_rate=0.9,
    ):
        """Gradient descent optimizer for linear/logistic models.

        Parameters
        ----------
        data : DolceSet
            Holds X (and y when training). A bias column is appended internally.
        alpha : float, default 0.01
            Learning rate; must be in (0, 1).
        n_iters : int, default 1000
            Maximum number of iterations; must be positive.
        tol : float, default 1e-6
            Convergence tolerance: stop once the absolute loss change drops
            below this value.
        loss_function : {"mse", "bce", None}, default None
            Loss to optimize. Inferred from y when None and training is possible.
        flavor : {"batch", "sgd", "mini_batch"}, default "batch"
            Which subset of rows to use per iteration.
        batch_fraction : float, default 0.1
            Fraction of rows per iteration for "mini_batch"; must be in (0, 1].
        momentum_type : {None, "polyak", "nesterov"}, default None
            Momentum variant. None disables momentum.
        momentum_rate : float, default 0.9
            Momentum coefficient in [0, 1); ignored when momentum_type is None.
        """

        # 1. Validate GD-specific input
        validate_alpha(alpha)
        validate_flavor(flavor)
        if flavor == "mini_batch":
            validate_batch_fraction(batch_fraction)
        validate_momentum_type(momentum_type)
        validate_momentum_rate(momentum_rate)

        # 2. Shared validation and attribute setup
        super().__init__(data, n_iters, tol, loss_function)

        # 3. Assign GD-specific attributes
        self.alpha = alpha
        self.flavor = flavor
        self.batch_fraction = batch_fraction
        self.momentum_type = momentum_type
        self.momentum_rate = momentum_rate

        # Mini-batch size as a row count, derived from the fraction of rows.
        self.mini_batch_size = max(1, math.ceil(batch_fraction * self.n_samples))

    def fit(self):

        validate_if_fitting_without_target(self.can_train)

        for iter in range(self.n_iters):

            # 1. Current weights
            weights = self.get_weights(iteration=iter)

            # 2. Apply chosen GD strategy
            # 1a. Batch GD: whole dataset at each iteration
            if self.flavor == "batch":
                sub_X, sub_y = self.X, self.y

            # 1b. Stochastic GD: one row at each iteration
            elif self.flavor == "sgd":
                sub_X, sub_y = sample_rows_Xy(self.X, self.y, 1)

            # 1c. Mini-batch GD: a sample at each iteration
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

            # 4. Update weights based on gradients
            # 4a. Polyak: momentum is weights distance
            if self.momentum_type == "polyak":
                prev_weights = self.get_weights(iteration=iter - 1)
                momentum = self.momentum_rate * (prev_weights - weights)
                updated_weights = weights - self.alpha * grad - momentum
            # 4b. Nesterov: momentum is already added before
            # Note: if Nesterov, below weights are accelerated ones
            elif self.momentum_type == "nesterov" or self.momentum_type == None:
                updated_weights = weights - self.alpha * grad
            self.append_weights(updated_weights)

            # 5. Track full-batch predictions and loss so the convergence
            #    curve is comparable across flavors.
            full_pred = self._apply_activation_function(
                np.matmul(self.X, pre_momentum_weights)
            )
            self.append_predictions(full_pred)
            loss = self._compute_loss(full_pred)
            self.append_loss(loss)

            # 6. If converged, break GD
            if iter > 0:
                prev_loss = self.get_loss(iteration=iter - 1)
                delta_loss = abs(loss - prev_loss)
                if delta_loss < self.tol:
                    break
