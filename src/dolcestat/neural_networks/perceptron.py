"""Stores Rosenblatt's perceptron."""

import numpy as np

from dolcestat.core.activations import Step
from dolcestat.core.losses import PerceptronCriterion
from dolcestat.core.parameters import ZeroParametersInitializer
from dolcestat.core.samplers import MiniBatchSampler
from dolcestat.core.trainer import Trainer
from dolcestat.metrics import ClassificationAnalyzer
from dolcestat.neural_networks.layers import DenseLayer
from dolcestat.neural_networks.networks import Sequential
from dolcestat.optimization.gradient_descent import GradientDescent
from dolcestat.preprocessing.core import DolceSet


class RosenblattPerceptron:
    """Rosenblatt's single-layer perceptron for binary classification.

    A linear threshold unit trained with the classic perceptron learning rule::

        y_pred = 1 if w . x + b > 0 else 0
        w      = w + alpha * (y - y_pred) * x

    The point of this class is that **none of that is implemented here**. A
    perceptron is a one-layer network with a ``Step`` activation, and its
    learning rule is plain gradient descent over one sample at a time against
    ``PerceptronCriterion``. So it is assembled from the same four pieces
    everything else in the library is assembled from — layer, loss, sampler,
    optimizer — and the 1958 rule falls out of the composition:

    | piece                        | supplies                        |
    |------------------------------|---------------------------------|
    | ``DenseLayer(Step(), n, 1)`` | the threshold unit              |
    | ``PerceptronCriterion``      | (y_pred - y) as the error       |
    | ``MiniBatchSampler(1)``      | one random sample per update    |
    | ``GradientDescent(alpha)``   | w <- w - alpha * grad           |

    Targets must be encoded as {0, 1}.
    """

    def __init__(self, alpha: float = 0.01):
        """
        Args:
            alpha: learning rate; must be in (0, 1)
        """
        self.alpha = alpha
        self.model = None
        self.history = None
        self.is_fitted = False
        self.training_metrics = None

    def fit(self, data: DolceSet, n_epochs: int = 1000) -> "RosenblattPerceptron":
        """
        Trains the perceptron on ``data``.

        Args:
            data: training data (DolceSet) with targets in {0, 1}
            n_epochs: number of online updates. One sample is drawn per epoch,
                so an epoch here is a single update — the same unit the classic
                formulation counts.

        Returns:
            self
        """
        # 1. The threshold unit, starting from the origin.
        self.model = Sequential(
            DenseLayer(
                activation=Step(),
                input_size=data.X.shape[1],
                output_size=1,
                parameters_initializer=ZeroParametersInitializer(),
            )
        )

        # 2. One random sample per update, and the rule itself.
        trainer = Trainer(
            model=self.model,
            loss=PerceptronCriterion(),
            optimizer=GradientDescent(alpha=self.alpha),
            sampler=MiniBatchSampler(batch_size=1),
        )

        # 3. tol=None: the error rate is unchanged by any update that lands on
        #    an already-correct sample, so a tolerance-based stop would fire on
        #    the first such step rather than at convergence.
        self.history = trainer.run(data, n_epochs=n_epochs, tol=None)

        self.is_fitted = True
        self.training_metrics = self.predict(data)
        return self

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Model is not fitted: call fit(data) first.")

    @property
    def weights(self) -> np.ndarray:
        """
        Returns:
            Coefficient matrix of shape (n_features, 1), intercept excluded
        """
        self._require_fitted()
        return self.model.layers[0].W.value

    @property
    def bias(self) -> np.ndarray:
        """
        Returns:
            Intercept, shape (1,)
        """
        self._require_fitted()
        return self.model.layers[0].b.value

    def predict(self, data: DolceSet) -> ClassificationAnalyzer:
        """
        Predicts class labels for ``data``.

        Args:
            data: data to predict on (DolceSet)

        Returns:
            ClassificationAnalyzer over the predicted {0, 1} labels
        """
        self._require_fitted()
        return ClassificationAnalyzer(data.y, self.model.predict(data.X).ravel())
