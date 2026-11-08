import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_global_rng():
    # Trees, forests, k-means, perceptron and mini-batch descent draw from
    # NumPy's global generator: seed it so every test is reproducible.
    np.random.seed(0)
