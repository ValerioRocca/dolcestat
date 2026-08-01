"""Stores distance functions."""

import numpy as np


def minkowski(a, b, p):
    abs_dist = np.abs(a - b)
    return np.sum(abs_dist**p) ** (1 / p)
