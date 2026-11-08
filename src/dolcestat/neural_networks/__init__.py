"""Neural networks: layers, containers, and the pieces that train them.

Everything a network needs -- activations, losses, parameter initializers,
samplers, the optimizer and the training loop -- lives here. From the rest of
the library the package takes the DolceSet data container, the analyzers that
``predict`` returns, and, for the ``RosenblattPerceptron`` only, the base class
of ``dolcestat.optimization``'s optimizers. In particular the
``GradientDescent`` exported here is *not* ``dolcestat.optimization``'s: see
``optimizers.py`` for why the two coexist.
"""

from .activations import LU, ActivationFunction, ReLU, Sigmoid, Softplus
from .layers import DenseLayer, Layer
from .losses import (
    BinaryCrossEntropy,
    GaussianNegativeLogLikelihood,
    Loss,
    MeanSquaredError,
)
from .networks import MultiHead, NeuralNetwork, Sequential
from .optimizers import GradientDescent, Optimizer
from .parameters import (
    GaussianInitializer,
    HeInitializer,
    Parameters,
    ParametersInitializer,
    XavierInitializer,
    ZeroParametersInitializer,
)
from .perceptron import RosenblattPerceptron
from .samplers import (
    BatchSampler,
    MiniBatchIteratingSampler,
    MiniBatchSampler,
    Sampler,
)

__all__ = [
    # containers
    "NeuralNetwork",
    "Sequential",
    "MultiHead",
    "RosenblattPerceptron",
    # layers
    "Layer",
    "DenseLayer",
    # activations
    "ActivationFunction",
    "LU",
    "Sigmoid",
    "Softplus",
    "ReLU",
    # losses
    "Loss",
    "MeanSquaredError",
    "BinaryCrossEntropy",
    "GaussianNegativeLogLikelihood",
    # parameters
    "Parameters",
    "ParametersInitializer",
    "ZeroParametersInitializer",
    "GaussianInitializer",
    "XavierInitializer",
    "HeInitializer",
    # samplers
    "Sampler",
    "BatchSampler",
    "MiniBatchSampler",
    "MiniBatchIteratingSampler",
    # optimizers
    "Optimizer",
    "GradientDescent",
]
