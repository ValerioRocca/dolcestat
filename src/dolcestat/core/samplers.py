"""Stores data sampling strategy classes"""

from abc import ABC, abstractmethod

import numpy as np

from dolcestat.preprocessing.core import DolceSet


def _validate_can_train(data: DolceSet) -> None:
    """
    Checks that ``data`` carries targets, and so can be sampled for training.

    Args:
        data: dataset about to be sampled (DolceSet)

    Raises:
        ValueError: if data has no target column, which would otherwise surface
            as an opaque TypeError when indexing y (None) inside a sampler.
    """
    if not data.can_train:
        raise ValueError(
            "Cannot sample training batches: this DolceSet has no target column "
            "(can_train is False). Load it with "
            "load_from_polars_dataframe(df, target_col=...) to train on it."
        )


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

        Raises:
            ValueError: if data carries no targets
        """
        _validate_can_train(data)
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
            Single-element list containing one mini-batch. When batch_size
            exceeds the number of rows, the batch is the whole dataset.

        Raises:
            ValueError: if data carries no targets
        """
        _validate_can_train(data)
        X = data.X
        y = data.y
        n_samples = X.shape[0]

        # Sampling without replacement cannot draw more rows than exist, so cap
        # the request at the dataset size instead of raising.
        size = min(self.batch_size, n_samples)

        indices = np.random.choice(n_samples, size, replace=False)
        return [
            type(data)(
                X=X[indices], y=y[indices], features=data.features, target=data.target
            )
        ]


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

        Raises:
            ValueError: if data carries no targets
        """
        _validate_can_train(data)
        X = data.X
        y = data.y
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        return [
            type(data)(
                X=X[indices[start : start + self.batch_size]],
                y=y[indices[start : start + self.batch_size]],
                features=data.features,
                target=data.target,
            )
            for start in range(0, n_samples, self.batch_size)
        ]
