from abc import ABC
from typing import Callable

import numpy as np

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
    "gradient descent" or "newton") or a *data-less* ``GradientDescent`` /
    ``NewtonMethod`` instance whose training data is loaded at fit time.
    """

    # --- The model spec: filled in by each concrete subclass ---
    _loss_function: str  # loss key passed to the optimizer ("mse" / "bce")
    _activation: Callable  # inverse link applied to the linear predictor Xw
    _analyzer_class: type  # PredictionAnalyzer subclass for the task
    _allowed_optimizers: set  # optimizer strings this model accepts

    def __init__(self, optimizer):
        # The raw optimizer request (string or data-less optimizer instance).
        self.optimizer = optimizer
        # The optimizer actually used to fit. None for the closed-form path.
        self.optimizer_obj = None
        self.weights = None
        self.is_fitted = False
        # Populated at fit time: a PredictionAnalyzer over the training data.
        self.training_metrics = None

    def _resolve_optimizer(self, data):
        """Build/configure the optimizer for ``data`` and return it.

        Returns the fit-ready optimizer instance, or None when fitting is
        handled without an optimizer (closed-form path, see LinearRegression).
        """
        # 1. A pre-built, data-less optimizer instance: just load the data.
        if isinstance(self.optimizer, (GradientDescent, NewtonMethod)):
            self.optimizer.load_training(data)
            return self.optimizer

        # 2. A string request: build the matching optimizer with sane defaults.
        if self.optimizer == "gradient descent":
            optimizer = GradientDescent(
                loss_function=self._loss_function,
                flavor="mini_batch",
                momentum_type="nesterov",
            )
        elif self.optimizer == "newton":
            optimizer = NewtonMethod(loss_function=self._loss_function)
        else:
            raise ValueError(f"Unsupported optimizer '{self.optimizer}'.")

        optimizer.load_training(data)
        return optimizer

    def fit(self, data):
        validate_optimizer(
            self.optimizer, self._allowed_optimizers, type(self).__name__
        )

        optimizer = self._resolve_optimizer(data)
        if optimizer is not None:
            optimizer.fit()
            self.optimizer_obj = optimizer
            self.weights = optimizer.get_weights(iteration=-1)

        self.is_fitted = True
        self.training_metrics = self.predict(data)
        return self

    def _predict_values(self, data):
        """Raw fitted values (activated linear predictor) as a NumPy array."""
        if not self.is_fitted:
            raise ValueError("Cannot predict: call fit(data) first.")
        X = np.column_stack((data.X, np.ones(data.X.shape[0])))
        return self._activation(np.matmul(X, self.weights))

    def predict(self, data):
        """A PredictionAnalyzer over the model's predictions for ``data``.

        Subclasses whose analyzer needs extra arguments (e.g. LinearRegression's
        ``n_features``) override this.
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
