import numpy as np

from .base import BaseOptimizer
from .input_validation import (
    validate_if_fitting_without_target,
    validate_if_training_not_loaded,
)


class NewtonMethod(BaseOptimizer):

    def __init__(self, data=None, n_iters=100, tol=1e-6, loss_function=None):
        """Vanilla Newton method with explicit matrix inversion.

        ``data`` is optional: when omitted, build a data-less optimizer and
        supply data later via ``load_training``.
        """
        super().__init__(data, n_iters, tol, loss_function)

    def fit(self):

        validate_if_training_not_loaded(self.is_training_loaded)
        validate_if_fitting_without_target(self.can_train)

        for iter in range(self.n_iters):

            # 1. Current weights
            weights = self.get_weights(iteration=iter)

            # 2. Compute gradient and Hessian
            pred = self._apply_activation_function(np.matmul(self.X, weights))
            grad = self._compute_gradient(pred, self.X, self.y)
            hessian = self._compute_hessian(pred, self.X, self.y)

            # 3. Compute step and update weights
            step = -np.matmul(np.linalg.inv(hessian), grad)
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
