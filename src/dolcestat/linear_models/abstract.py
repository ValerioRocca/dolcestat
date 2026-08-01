"""Stores the shared base for generalized linear models."""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.core.parameters import ZeroParametersInitializer
from dolcestat.core.samplers import MiniBatchSampler, Sampler
from dolcestat.core.trainer import Trainer
from dolcestat.neural_networks.layers import DenseLayer
from dolcestat.neural_networks.networks import Sequential
from dolcestat.optimization.analyzer import OptimizerAnalyzer
from dolcestat.optimization.gradient_descent import GradientDescent

from .input_validation import validate_optimizer


def unpack_flat_weights(w_flat: np.ndarray, layer: DenseLayer) -> None:
    """
    Writes a flat ``[w1, ..., wp, b]`` vector into a layer's Parameters.

    The closed form and Newton's method both derive naturally on the
    bias-augmented design matrix ``[X | 1]``, which yields a single flat vector
    with the intercept as its last entry. The layer instead keeps the bias as an
    explicit parameter, so the vector is split on the way out. This is the price
    of the explicit bias, and this function is the whole of it.

    Args:
        w_flat: flat weight vector, intercept last
        layer: DenseLayer whose W and b are overwritten
    """
    layer.W.value = w_flat[:-1].reshape(-1, 1).astype(float)
    layer.b.value = w_flat[-1:].astype(float)


class GLMAbstract(ABC):
    """Shared base for generalized linear models (linear/logistic regression).

    A GLM *is* a one-layer network: its prediction is ``activation(XW + b)``,
    which is exactly what a ``DenseLayer`` computes. So rather than reimplement
    the linear algebra, each model **owns** a one-layer ``Sequential`` and fills
    in its parameters.

    The model is not the fitting algorithm. ``Sequential``/``DenseLayer`` define
    only the parameterisation and the forward map; *how* good values for W and b
    are found is a separate concern, and three interchangeable strategies do it:

    | strategy         | mechanism                          | gradients |
    |------------------|------------------------------------|-----------|
    | gradient descent | Trainer loop -> optimizer.step     | backprop  |
    | newton           | own loop, joint Hessian on [X | 1] | own       |
    | closed form      | one shot, (XtX)^-1 Xt y            | none      |

    All three end by writing into the same ``Parameters``, so ``predict`` works
    identically regardless of which one ran.

    Composition, not inheritance: a GLM *has* a Sequential rather than *being*
    one, so it keeps its own teaching-oriented surface (``fit(data)``,
    ``predict(data) -> PredictionAnalyzer``, the ``optimizer`` string API)
    instead of inheriting the network's.
    """

    # --- The model spec: filled in by each concrete subclass ---
    _loss_class: type  # Loss class used to fit and to report training loss
    _activation_class: type  # ActivationFunction applied to the linear predictor
    _analyzer_class: type  # PredictionAnalyzer subclass for the task
    _allowed_optimizers: set  # optimizer strings this model accepts

    def __init__(self, optimizer, sampler: Sampler = None):
        """
        Args:
            optimizer: how to fit — a string this model allows, or an Optimizer
                instance to drive the training loop with
            sampler: batching strategy used by the gradient-descent path.
                Defaults to MiniBatchSampler(). Ignored by the closed form and
                by Newton, which are both full-dataset methods.
        """
        self.optimizer = optimizer
        self.sampler = sampler
        # The underlying one-layer network, built at fit time (it needs to know
        # how many features there are).
        self.model = None
        # Record of the run, for the iterative strategies only.
        self.history = None
        self.is_fitted = False
        # Populated at fit time: a PredictionAnalyzer over the training data.
        self.training_metrics = None

    # --- The underlying one-layer network ---

    def _build_model(self, n_features: int) -> None:
        """
        Builds the one-layer network this model fits.

        Args:
            n_features: number of input features
        """
        # Zero initialization: the iterative strategies start from the origin,
        # and the one-shot strategies overwrite these values anyway.
        self.model = Sequential(
            DenseLayer(
                activation=self._activation_class(),
                input_size=n_features,
                output_size=1,
                parameters_initializer=ZeroParametersInitializer(),
            )
        )

    @property
    def _layer(self) -> DenseLayer:
        """
        Returns:
            The single DenseLayer holding this model's parameters
        """
        return self.model.layers[0]

    @property
    def weights(self) -> np.ndarray:
        """
        Returns:
            Coefficient matrix of shape (n_features, 1). The intercept is *not*
            included — see ``bias``.
        """
        self._require_fitted()
        return self._layer.W.value

    @property
    def bias(self) -> np.ndarray:
        """
        Returns:
            Intercept, shape (1,)
        """
        self._require_fitted()
        return self._layer.b.value

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Model is not fitted: call fit(data) first.")

    # --- Fitting ---

    def fit(
        self,
        data,
        n_epochs: int = 1000,
        tol: float = 1e-6,
        **run_kwargs,
    ):
        """
        Fits the model to ``data``.

        Data is supplied here rather than at construction, so one configured
        model can be fitted to several datasets and an optimizer never holds a
        reference to a training set.

        Args:
            data: training data (DolceSet)
            n_epochs: maximum passes over the data for the iterative strategies.
                For Newton, one epoch is one Newton iteration.
            tol: convergence tolerance on the full-dataset loss
            **run_kwargs: forwarded to Trainer.run on the gradient-descent path

        Returns:
            self
        """
        validate_optimizer(
            self.optimizer, self._allowed_optimizers, type(self).__name__
        )

        # 1. Build the one-layer network this model is a wrapper around.
        self._build_model(n_features=data.X.shape[1])

        # 2. Dispatch to the requested fitting strategy. Each one leaves the
        #    layer's Parameters holding the fitted values.
        if isinstance(self.optimizer, str) and self.optimizer != "gradient descent":
            self._fit_direct(data)
        else:
            self._fit_iterative(data, n_epochs=n_epochs, tol=tol, **run_kwargs)

        self.is_fitted = True
        self.training_metrics = self.predict(data)
        return self

    def _fit_direct(self, data) -> None:
        """
        Runs a strategy that works on the bias-augmented matrix directly.

        Args:
            data: training data (DolceSet)
        """
        X_augmented = np.column_stack((data.X, np.ones(data.X.shape[0])))

        if self.optimizer == "closed form":
            w_flat = self._solve_closed_form(X_augmented, data.y)
        elif self.optimizer == "newton":
            w_flat = self._solve_newton(X_augmented, data.y)
        else:
            raise ValueError(f"Unsupported optimizer '{self.optimizer}'.")

        unpack_flat_weights(w_flat, self._layer)

    def _fit_iterative(self, data, n_epochs: int, tol: float, **run_kwargs) -> None:
        """
        Runs the gradient-descent path through the shared training loop.

        Args:
            data: training data (DolceSet)
            n_epochs: maximum passes over the data
            tol: convergence tolerance on the full-dataset loss
            **run_kwargs: forwarded to Trainer.run
        """
        # A string request builds an optimizer with sensible defaults; an
        # instance is used as given.
        if isinstance(self.optimizer, str):
            optimizer = GradientDescent(momentum_type="nesterov")
        else:
            optimizer = self.optimizer

        sampler = self.sampler if self.sampler is not None else MiniBatchSampler()

        trainer = Trainer(
            model=self.model,
            loss=self._loss_class(),
            optimizer=optimizer,
            sampler=sampler,
        )
        self.history = trainer.run(data, n_epochs=n_epochs, tol=tol, **run_kwargs)

    def _solve_closed_form(self, X_augmented: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solves the model in one shot, when a closed form exists.

        Args:
            X_augmented: design matrix with a trailing column of ones
            y: targets

        Returns:
            Flat weight vector, intercept last
        """
        raise ValueError(
            f"{type(self).__name__} has no closed-form solution; "
            "use 'gradient descent' or 'newton'."
        )

    def _solve_newton(self, X_augmented: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Runs Newton's method on the bias-augmented problem.

        Newton is deliberately *not* an ``Optimizer``: it needs the design
        matrix and a single joint Hessian over one flat parameter vector, and
        the ``step(parameters)`` protocol can supply neither. It stays a fitting
        strategy owned by the model, which is the only place those objects
        exist.

        Args:
            X_augmented: design matrix with a trailing column of ones
            y: targets

        Returns:
            Flat weight vector, intercept last
        """
        from dolcestat.core.history import History

        history = History()
        w_flat = np.zeros(X_augmented.shape[1])
        loss = self._loss_class()

        for iteration in range(self._newton_max_iters):

            # 1. Predictions, and the loss at the weights we start this
            #    iteration with.
            predictions = self._linear_predictor(X_augmented, w_flat)
            loss_value = loss.forward(predictions, y, training=False)
            history.record_epoch(iteration, loss_value)
            history.record_step(iteration, iteration, loss_value)

            # 2. Gradient and joint Hessian, both supplied by the concrete model.
            gradient = self._newton_gradient(X_augmented, y, predictions)
            hessian = self._newton_hessian(X_augmented, y, predictions)

            # 3. Newton step: rescale the gradient by the inverse curvature.
            w_flat = w_flat - np.matmul(np.linalg.inv(hessian), gradient)

            # 4. Stop once the loss stops moving.
            if iteration > 0:
                previous = history.epochs[-2]["epoch_loss"]
                if abs(loss_value - previous) < self._newton_tol:
                    break

        self.history = history
        return w_flat

    def _linear_predictor(
        self, X_augmented: np.ndarray, w_flat: np.ndarray
    ) -> np.ndarray:
        """
        Returns:
            activation(X_augmented @ w_flat), the model's fitted values
        """
        return self._activation_class().forward(
            np.matmul(X_augmented, w_flat), training=False
        )

    @abstractmethod
    def _newton_gradient(
        self, X_augmented: np.ndarray, y: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """dL/dw on the augmented design matrix, for Newton's method."""

    @abstractmethod
    def _newton_hessian(
        self, X_augmented: np.ndarray, y: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """Joint Hessian over the flat weight vector, for Newton's method."""

    # --- Prediction ---

    def _predict_values(self, data) -> np.ndarray:
        """
        Returns:
            Fitted values as a 1-D array. The network emits an (n, 1) column;
            it is flattened here so ``metrics`` keeps receiving 1-D vectors.
        """
        self._require_fitted()
        return self.model.predict(data.X).ravel()

    def predict(self, data):
        """
        A PredictionAnalyzer over the model's predictions for ``data``.

        Subclasses whose analyzer needs extra arguments (e.g. LinearRegression's
        ``n_features``) override this.

        Args:
            data: data to predict on (DolceSet)

        Returns:
            PredictionAnalyzer subclass instance for the task
        """
        return self._analyzer_class(data.y, self._predict_values(data))

    @property
    def optimization(self):
        """OptimizerAnalyzer over the fitted run (loss curves, etc.)."""
        self._require_fitted()
        if self.history is None:
            raise ValueError(
                "No training history to analyze: this model was fitted with the "
                "closed-form solution, which has no iterative loss history."
            )
        return OptimizerAnalyzer(self.history)

    # Newton's own loop budget. Kept small: Newton converges in a handful of
    # iterations, and each one inverts a matrix.
    _newton_max_iters = 100
    _newton_tol = 1e-6
