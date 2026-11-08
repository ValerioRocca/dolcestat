"""Newton's method for linear (MSE) and logistic (BCE) objectives."""

import numpy as np

from .base import BaseOptimizer


class NewtonMethod(BaseOptimizer):
    """Newton's method: each step solves with the exact Hessian of the loss.

    Needs far fewer iterations than gradient descent but inverts a
    ``(n_features + 1)`` square matrix at every step, and fails on perfectly
    collinear features. After ``fit(data)`` the run is available through
    ``get_weights()`` and ``get_loss()``.
    """

    def __init__(self, *, n_iters=100, tol=1e-6, loss_function=None):
        """Vanilla Newton method with explicit matrix inversion.

        Built from its settings alone; ``fit(data)`` runs the iterations and
        returns the optimizer.
        """
        super().__init__(n_iters, tol, loss_function)

    def fit(self, data):
        """Runs Newton's method on ``data`` and returns self."""

        self._load_training(data)

        for iter in range(self.n_iters):

            # 1. Current weights
            weights = self.get_weights(iteration=iter)

            # 2. Compute gradient and Hessian
            pred = self._apply_activation_function(np.matmul(self.X, weights))
            grad = self._compute_gradient(pred, self.X, self.y)
            hessian = self._compute_hessian(pred, self.X, self.y)

            # 3. Compute step and update weights
            try:
                step = -np.matmul(np.linalg.inv(hessian), grad)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "The Hessian is singular, so Newton's method cannot take a "
                    "step. This usually means some features are perfectly "
                    "collinear (or constant): drop the redundant ones, or use "
                    "gradient descent."
                ) from None
            weights = weights + step
            self.append_weights(weights)

            # 4. Track full-batch predictions and loss so the convergence
            #    curve is comparable across flavors.
            self.append_predictions(pred)
            loss = self._compute_loss(pred)
            self.append_loss(loss)

            # 5. If converged, break iteration
            if iter > 0:
                prev_loss = self.get_loss(iteration=iter - 1)
                delta_loss = abs(loss - prev_loss)
                if delta_loss < self.tol:
                    break

        self.is_fitted = True
        return self
