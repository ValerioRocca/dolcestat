"""Base class of the generalized linear models."""

import copy
from abc import ABC
from typing import Callable

import numpy as np

from dolcestat._validation import (
    validate_is_fitted,
    validate_prediction_data,
    validate_training_data,
)
from dolcestat.optimization.analyzer import OptimizerAnalyzer
from dolcestat.optimization.gradient_descent import GradientDescent
from dolcestat.optimization.newton import NewtonMethod

from .input_validation import validate_optimizer


class GLMAbstract(ABC):
    """Shared base for generalized linear models (linear/logistic regression).

    A concrete model is defined by the four class attributes below: its loss
    function, the activation (inverse link) applied to the linear predictor, the
    analyzer for its task, and the optimizers it accepts. This base owns the
    shared ``fit``/``predict`` flow and the optimizer dispatch.

    The ``optimizer`` argument is either a string ("closed form" — linear only,
    "gradient descent" or "newton") or a ``GradientDescent`` / ``NewtonMethod``
    instance carrying the settings you want. The model never runs your
    instance: each ``fit`` works on a fresh copy of it, so the instance stays a
    reusable template and the model can be fitted again.
    """

    # --- The model spec: filled in by each concrete subclass ---
    _loss_function: str  # loss key passed to the optimizer ("mse" / "bce")
    _activation: Callable  # inverse link applied to the linear predictor Xw
    _analyzer_class: type  # PredictionAnalyzer subclass for the task
    _allowed_optimizers: set  # optimizer strings this model accepts

    def __init__(self, optimizer):
        validate_optimizer(optimizer, self._allowed_optimizers, type(self).__name__)
        # The raw optimizer request (string or optimizer instance).
        self.optimizer = optimizer
        # The optimizer actually used to fit. None for the closed-form path.
        self.optimizer_obj = None
        self.weights = None
        self.n_features = None
        self.is_fitted = False
        # Populated at fit time: a PredictionAnalyzer over the training data.
        self.training_metrics = None

    def _validate_target(self, y):
        """Hook for models that restrict the target (e.g. binary labels)."""

    def _build_optimizer(self):
        """A fit-ready optimizer for this model, or None when fitting is
        handled without one (closed-form path, see LinearRegression)."""
        # 1. An optimizer instance: work on a copy, pinned to this model's loss.
        if isinstance(self.optimizer, (GradientDescent, NewtonMethod)):
            optimizer = copy.deepcopy(self.optimizer)
            optimizer._require_loss_function(self._loss_function)
            return optimizer

        # 2. A string request: build the matching optimizer with sane defaults.
        if self.optimizer == "gradient descent":
            return GradientDescent(
                loss_function=self._loss_function,
                flavor="mini_batch",
                momentum_type="nesterov",
            )
        if self.optimizer == "newton":
            return NewtonMethod(loss_function=self._loss_function)
        raise ValueError(f"Unsupported optimizer '{self.optimizer}'.")

    def _fit_closed_form(self, data):
        raise NotImplementedError(f"{type(self).__name__} has no closed-form solution.")

    def fit(self, data):
        """Fits the model on ``data`` (a DolceSet with a target); returns self."""
        validate_training_data(data)
        self._validate_target(data.y)

        optimizer = self._build_optimizer()
        if optimizer is None:
            self._fit_closed_form(data)
            self.optimizer_obj = None
        else:
            optimizer.fit(data)
            self.optimizer_obj = optimizer
            self.weights = optimizer.get_weights(iteration=-1)

        self.n_features = data.X.shape[1]
        self.is_fitted = True
        self.training_metrics = self.predict(data)
        return self

    def _predict_values(self, data):
        """Raw fitted values (activated linear predictor) as a NumPy array."""
        validate_is_fitted(self.is_fitted, type(self).__name__)
        validate_prediction_data(data, self.n_features)
        X = np.column_stack((data.X, np.ones(data.X.shape[0])))
        return self._activation(np.matmul(X, self.weights))

    def predict(self, data):
        """A PredictionAnalyzer over the model's predictions for ``data``.

        ``data`` needs no target: without one the analyzer still carries the
        predictions in ``y_fit``. Subclasses whose analyzer needs extra
        arguments (e.g. LinearRegression's ``n_features``) override this.
        """
        return self._analyzer_class(data.y, self._predict_values(data))

    @property
    def optimization(self):
        """OptimizerAnalyzer over the fitted optimizer (loss curves, etc.)."""
        if not self.is_fitted:
            raise ValueError("Cannot analyze the optimizer: call fit(data) first.")
        if self.optimizer_obj is None:
            raise ValueError(
                "No optimizer to analyze: this model was fitted with the "
                "closed-form solution, which has no iterative loss history."
            )
        return OptimizerAnalyzer(self.optimizer_obj)
