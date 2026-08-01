"""Shared building blocks used across the library.

This package holds the vocabulary that is not specific to any one model family:
parameters and their initializers, loss functions, activation functions and
(mini-)batch samplers. Linear models, neural networks and metrics all build on
these, so they live here rather than inside any single model package.

Dependencies flow one way: ``core`` imports nothing from the model packages.
"""

from .activations import LU, ActivationFunction, ReLU, Sigmoid, Step
from .history import History
from .losses import (
    BinaryCrossEntropy,
    Loss,
    MeanSquaredError,
    PerceptronCriterion,
)
from .optimizer import Optimizer
from .parameters import (
    GaussianInitializer,
    HeInitializer,
    Parameters,
    ParametersInitializer,
    XavierInitializer,
    ZeroParametersInitializer,
)
from .samplers import (
    BatchSampler,
    MiniBatchIteratingSampler,
    MiniBatchSampler,
    Sampler,
)
from .trainer import Trainer

__all__ = [
    "ActivationFunction",
    "LU",
    "ReLU",
    "Sigmoid",
    "Step",
    "Loss",
    "MeanSquaredError",
    "BinaryCrossEntropy",
    "PerceptronCriterion",
    "Parameters",
    "ParametersInitializer",
    "ZeroParametersInitializer",
    "GaussianInitializer",
    "XavierInitializer",
    "HeInitializer",
    "Sampler",
    "BatchSampler",
    "MiniBatchSampler",
    "MiniBatchIteratingSampler",
    "Optimizer",
    "Trainer",
    "History",
]
