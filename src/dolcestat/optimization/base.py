"""Base class shared by the gradient-based optimizers."""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat._validation import (
    validate_is_fitted,
    validate_prediction_data,
    validate_training_data,
)

from .input_validation import (
    validate_get_weights_iteration,
    validate_loss_function,
    validate_n_iters,
    validate_tol,
)
from .loss_and_activation import (
    compute_bce_gradient,
    compute_bce_hessian,
    compute_bce_loss,
    compute_mse_gradient,
    compute_mse_hessian,
    compute_mse_loss,
    identity,
    sigmoid,
)


class BaseOptimizer(ABC):
    """Shared base for gradient-based optimizers.

    An optimizer is built from its settings alone and handed the data at
    ``fit(data)``, which returns the optimizer itself. This base holds the
    common constructor block (validation), the data loading (bias-column append
    and weight/prediction/loss buffers), the loss-function dispatchers, and the
    weight/prediction/loss accessors. Subclasses implement only their own
    ``fit`` and any algorithm-specific constructor parameters.
    """

    def __init__(
        self,
        n_iters=1000,
        tol=1e-6,
        loss_function=None,
    ):
        # 1. Validate the hyperparameters
        validate_n_iters(n_iters)
        validate_tol(tol)
        validate_loss_function(loss_function)

        # 2. Assign them. The requested loss may be None and is only resolved
        #    once training data is available.
        self.n_iters = n_iters
        self.tol = tol
        self._requested_loss_function = loss_function
        self.loss_function = loss_function
        self.is_fitted = False

    def _require_loss_function(self, loss_function):
        """Pins the loss this optimizer must minimise (used by model wrappers).

        An optimizer left to infer its loss gets ``loss_function``; one that
        was told a different loss is a conflict and raises.
        """
        if self._requested_loss_function is None:
            self._requested_loss_function = loss_function
            self.loss_function = loss_function
        elif self._requested_loss_function != loss_function:
            raise ValueError(
                f"This optimizer was built with loss_function="
                f"'{self._requested_loss_function}', but the model needs "
                f"'{loss_function}'."
            )

    def _load_training(self, data):
        """Validates the training data and (re)initialises every buffer.

        Called at the start of ``fit``: fitting again, with the same or other
        data, starts from scratch.

        Parameters
        ----------
        data : DolceSet
            Holds X and y. A bias column is appended internally.
        """
        # 1. Validate the data
        validate_training_data(data)

        # 2. Assign data-dependent attributes
        X = data.X
        self.y = data.y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # 3. Resolve the loss function: infer it from y when none was requested
        if self._requested_loss_function is None:
            self.loss_function = self._infer_loss_function()
        else:
            self.loss_function = self._requested_loss_function

        # 4. Add bias term as last column
        self.X = np.column_stack((X, np.ones(X.shape[0])))

        # 5. Weights initialized to zeros; bias term is the last element
        self.weights = np.zeros((1, self.n_features + 1))
        self.predictions = np.empty((0, self.n_samples))
        self.loss = np.array([])

    def _infer_loss_function(self):
        if np.issubdtype(self.y.dtype, np.floating) or (
            np.issubdtype(self.y.dtype, np.integer) and len(np.unique(self.y)) > 2
        ):
            return "mse"
        unique = np.unique(self.y)
        if (
            np.array_equal(unique, [0, 1])
            or np.array_equal(unique, [0])
            or np.array_equal(unique, [1])
        ):
            return "bce"
        raise ValueError(
            f"Cannot infer loss function from y with dtype '{self.y.dtype}' and values {unique}. Provide a loss_function explicitly."
        )

    @property
    def loss_label(self):
        """Name of the tracked loss, as shown on plots."""
        return str(self.loss_function).upper()

    def get_weights(self, iteration=None):
        """Weights recorded during the run.

        Parameters
        ----------
        iteration : int, optional
            Index of the iteration (negative values count from the end, so
            ``-1`` is the final weights). The bias is the last entry. When None,
            the whole history is returned, one row per iteration, starting with
            the all-zero initial weights.
        """
        validate_get_weights_iteration(iteration, self.n_iters)
        if iteration is not None:
            return self.weights[iteration, :]
        return self.weights

    def append_weights(self, new_weights):
        """Records the weights reached at the end of an iteration."""
        self.weights = np.vstack((self.weights, new_weights))

    def append_predictions(self, new_predictions):
        """Records the full-data predictions made during an iteration."""
        self.predictions = np.vstack((self.predictions, new_predictions))

    def get_loss(self, iteration=None):
        """Loss recorded during the run.

        Parameters
        ----------
        iteration : int, optional
            Index of the iteration. When None, the whole loss history is
            returned as an array.
        """
        if iteration is not None:
            return self.loss[iteration]
        return self.loss

    def append_loss(self, new_loss):
        """Records the loss measured at an iteration."""
        self.loss = np.append(self.loss, new_loss)

    def _apply_activation_function(self, weighted_inputs):
        if self.loss_function == "mse":
            return identity(weighted_inputs)
        elif self.loss_function == "bce":
            return sigmoid(weighted_inputs)
        else:
            raise ValueError(f"Unsupported loss function '{self.loss_function}'.")

    def _compute_gradient(self, y_pred, sub_X, sub_y):
        if self.loss_function == "mse":
            return compute_mse_gradient(sub_X, sub_y, y_pred)
        elif self.loss_function == "bce":
            return compute_bce_gradient(sub_X, sub_y, y_pred)
        else:
            raise ValueError(f"Unsupported loss function '{self.loss_function}'.")

    def _compute_hessian(self, y_pred, sub_X, sub_y):
        if self.loss_function == "mse":
            return compute_mse_hessian(sub_X, sub_y, y_pred)
        elif self.loss_function == "bce":
            return compute_bce_hessian(sub_X, sub_y, y_pred)
        else:
            raise ValueError(f"Unsupported loss function '{self.loss_function}'.")

    def _compute_loss(self, y_pred):
        if self.loss_function == "mse":
            return compute_mse_loss(self.y, y_pred)
        elif self.loss_function == "bce":
            return compute_bce_loss(self.y, y_pred)
        else:
            raise ValueError(f"Unsupported loss function '{self.loss_function}'.")

    @abstractmethod
    def fit(self, data):
        """Run the optimization loop on ``data``; returns self."""

    def predict(self, data):
        """Predictions for ``data`` with the final weights, as a plain array."""
        validate_is_fitted(self.is_fitted, type(self).__name__)
        validate_prediction_data(data, self.n_features)

        # 1. Initialize X
        X = data.X
        X_new_with_bias = np.column_stack((X, np.ones(X.shape[0])))

        # 2. Predict using the last weights
        final_weights = self.get_weights(iteration=-1)
        y_pred = self._apply_activation_function(
            np.matmul(X_new_with_bias, final_weights)
        )
        return y_pred
