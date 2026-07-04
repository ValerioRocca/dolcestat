from abc import ABC, abstractmethod

import numpy as np

from .input_validation import (
    validate_get_weights_iteration,
    validate_loss_function,
    validate_n_iters,
    validate_tol,
    validate_X,
    validate_X_y,
    validate_y,
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

    Holds the common constructor block (validation, bias-column append, weight/
    prediction/loss buffers), the loss-function dispatchers, and the weight/
    prediction/loss accessors. Subclasses implement only their own ``fit`` and
    any algorithm-specific constructor parameters.
    """

    def __init__(
        self,
        data=None,
        n_iters=1000,
        tol=1e-6,
        loss_function=None,
    ):
        # 1. Validate the data-independent hyperparameters
        validate_n_iters(n_iters)
        validate_tol(tol)
        validate_loss_function(loss_function)

        # 2. Assign data-independent attributes. The requested loss may be None
        #    and is only resolved once training data is available.
        self.n_iters = n_iters
        self.tol = tol
        self._requested_loss_function = loss_function
        self.loss_function = loss_function

        # 3. Training data is optional at construction. When omitted, the
        #    optimizer is a data-less template that must be populated later via
        #    load_training (e.g. by a model wrapper such as LinearRegression).
        self.can_train = False
        self.is_training_loaded = False
        if data is not None:
            self.load_training(data)

    def load_training(self, data):
        """Attach training (or prediction) data to the optimizer.

        Runs the data-dependent setup that used to live in ``__init__``:
        validation, loss inference, bias-column append and (re)initialisation of
        the weight/prediction/loss buffers. Calling it again with new data
        resets those buffers so an optimizer can be reused.

        Parameters
        ----------
        data : DolceSet
            Holds X (and y when training). A bias column is appended internally.
        """
        # 1. Validate the data
        validate_X(data.X)
        if data.can_train:
            validate_y(data.y)
            validate_X_y(data.X, data.y)

        # 2. Assign data-dependent attributes
        self.can_train = data.can_train
        X = data.X
        if self.can_train:
            self.y = data.y
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]

        # 3. Resolve the loss function: infer it from y when none was requested
        #    and training is possible, otherwise keep the requested value.
        if self._requested_loss_function is None and self.can_train:
            self.loss_function = self._infer_loss_function()
        else:
            self.loss_function = self._requested_loss_function

        # 4. Add bias term as last column
        self.X = np.column_stack((X, np.ones(X.shape[0])))

        # 5. Weights initialized to zeros; bias term is the last element
        self.weights = np.zeros((1, self.n_features + 1))
        self.predictions = np.empty((0, self.n_samples))
        self.loss = np.array([])

        self.is_training_loaded = True

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

    def get_weights(self, iteration=None):
        validate_get_weights_iteration(iteration, self.n_iters)
        if iteration is not None:
            return self.weights[iteration, :]
        return self.weights

    def append_weights(self, new_weights):
        self.weights = np.vstack((self.weights, new_weights))

    def get_predictions(self, iteration=None):
        if iteration is not None:
            return self.predictions[iteration, :]
        return self.predictions

    def append_predictions(self, new_predictions):
        self.predictions = np.vstack((self.predictions, new_predictions))

    def get_loss(self, iteration=None):
        if iteration is not None:
            return self.loss[iteration]
        return self.loss

    def append_loss(self, new_loss):
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
    def fit(self):
        """Run the optimization loop; implemented by each optimizer."""

    def predict(self, data):
        # 1. Initialize X
        X = data.X
        X_new_with_bias = np.column_stack((X, np.ones(X.shape[0])))

        # 2. Predict using the last weights
        final_weights = self.get_weights(iteration=-1)
        y_pred = self._apply_activation_function(
            np.matmul(X_new_with_bias, final_weights)
        )
        return y_pred
