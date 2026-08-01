"""Stores data sampling strategy classes"""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.preprocessing.core import DolceSet


def _make_batch(data: DolceSet, X: np.ndarray, y: np.ndarray) -> DolceSet:
    """
    Builds a DolceSet batch from raw arrays.

    DolceSet.__init__ takes no arguments (it's normally populated via
    load_from_polars_dataframe), so batches are built by constructing an
    empty instance and assigning the sliced arrays directly.

    Args:
        data: source dataset the batch is drawn from (used for its class and can_train)
        X: batch input features
        y: batch targets

    Returns:
        A DolceSet holding the given batch
    """
    batch = type(data)()
    batch.X = X
    batch.y = y
    batch.can_train = data.can_train
    return batch


class Sampler(ABC):
    """Abstract base class for (mini-)batching strategies."""

    @abstractmethod
    def sample(self, data: DolceSet) -> list:
        """
        Splits a DolceSet into one or more batches.

        Args:
            data: dataset to sample from (DolceSet)

        Returns:
            List of DolceSet batches
        """


class BatchSampler(Sampler):
    """Returns a list storing the whole dataset"""

    def sample(self, data: DolceSet) -> list:
        """
        Args:
            data: dataset to sample from (DolceSet)

        Returns:
            Single-element list containing the whole dataset
        """
        return [data]


class MiniBatchSampler(Sampler):
    """Returns a list storing one mini-batch"""

    def __init__(self, batch_size: int = 32):
        """
        Args:
            batch_size: number of rows drawn (without replacement) into the batch
        """
        self.batch_size = batch_size

    def sample(self, data: DolceSet) -> list:
        """
        Draws one mini-batch without replacement.

        Args:
            data: dataset to sample from (DolceSet)

        Returns:
            Single-element list containing one mini-batch
        """
        X = data.X
        y = data.y
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.batch_size, replace=False)
        return [_make_batch(data, X[indices], y[indices])]


class MiniBatchIteratingSampler(Sampler):
    """Returns a list of mini-batches such that all rows are covered."""

    def __init__(self, batch_size: int = 32):
        """
        Args:
            batch_size: number of rows per mini-batch (the last one may be smaller)
        """
        self.batch_size = batch_size

    def sample(self, data: DolceSet) -> list:
        """
        Shuffles the data and splits it into consecutive mini-batches.

        Args:
            data: dataset to sample from (DolceSet)

        Returns:
            List of mini-batches covering every row of data exactly once
        """
        X = data.X
        y = data.y
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        return [
            _make_batch(
                data,
                X[indices[start : start + self.batch_size]],
                y[indices[start : start + self.batch_size]],
            )
            for start in range(0, n_samples, self.batch_size)
        ]
