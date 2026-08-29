import numpy as np

from dolcestat.optimization.base import BaseOptimizer
from dolcestat.optimization.input_validation import (
    validate_alpha,
    validate_if_fitting_without_target,
    validate_if_training_not_loaded,
)


class RosenblattPerceptron(BaseOptimizer):
    """Rosenblatt's single-layer perceptron for binary classification.

    A linear threshold unit trained with the classic perceptron learning rule.
    The prediction is a Heaviside step of the weighted input and the weights are
    updated online, one random sample at a time::

        y_pred = 1 if w . x > 0 else 0
        w      = w + alpha * (y - y_pred) * x

    The bias term is folded into the weights: ``BaseOptimizer.load_training``
    appends a constant column of ones to ``X``, so the last weight is the bias.
    Targets must be encoded as {0, 1}.
    """

    def __init__(self, data=None, alpha=0.01, n_iters=1000):
        """Build the perceptron.

        Parameters
        ----------
        data : DolceSet, optional
            Holds X (and y when training). A bias column is appended internally.
            When omitted, build a data-less optimizer and supply data later via
            ``load_training``.
        alpha : float, default 0.01
            Learning rate; must be in (0, 1).
        n_iters : int, default 1000
            Number of online updates (one random sample each); must be positive.
        """
        # 1. Validate perceptron-specific input.
        validate_alpha(alpha)

        # 2. Assign perceptron-specific attributes before super().__init__, which
        #    may call load_training.
        self.alpha = alpha

        # 3. Shared validation and attribute setup. The perceptron uses its own
        #    step activation and 0/1 error, so no mse/bce loss is requested and
        #    tol is left at the base default (unused: there is no tol-based stop).
        super().__init__(data=data, n_iters=n_iters, loss_function=None)

    @staticmethod
    def _step(z):
        """Heaviside step activation: 1 where z > 0, else 0."""
        return (z > 0).astype(int)

    def fit(self):

        validate_if_training_not_loaded(self.is_training_loaded)
        validate_if_fitting_without_target(self.can_train)

        for iter in range(self.n_iters):

            # 1. Current weights (last row of the weight history).
            weights = self.get_weights(iteration=iter)

            # 2. Draw one random training sample (online / stochastic update).
            idx = np.random.choice(self.n_samples)
            x = self.X[idx]
            y = self.y[idx]

            # 3. Prediction via the step activation on the single sample.
            y_pred = self._step(np.dot(x, weights))

            # 4. Perceptron learning rule: no update on correct predictions,
            #    push weights toward x on a miss (and away from it otherwise).
            updated_weights = weights + self.alpha * (y - y_pred) * x
            self.append_weights(updated_weights)

            # 5. Track full-batch predictions and the 0/1 misclassification rate
            #    so the convergence curve is available to the analyzer.
            full_pred = self._step(np.matmul(self.X, updated_weights))
            self.append_predictions(full_pred)
            self.append_loss(np.mean(full_pred != self.y))

    def predict(self, data):
        # Predict with the step activation on the final weights, adding the same
        # bias column that training used.
        X_with_bias = np.column_stack((data.X, np.ones(data.X.shape[0])))
        final_weights = self.get_weights(iteration=-1)
        return self._step(np.matmul(X_with_bias, final_weights))
