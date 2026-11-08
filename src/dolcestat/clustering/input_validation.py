"""Checks on the settings of the clustering models."""

from dolcestat._validation import (  # noqa: F401  (shared data checks, re-exported)
    validate_X,
)


def validate_init_strategy(init_strategy):
    """Checks the centroid initialization: 'random' or 'kmeans++'."""
    valid = {"random", "kmeans++"}
    if init_strategy not in valid:
        raise ValueError(
            f"init_strategy must be one of {sorted(valid)}, got '{init_strategy}'."
        )


def validate_flavor(flavor):
    """Checks the clustering flavor: 'kmeans' or 'kmedians'."""
    valid = {"kmeans", "kmedians"}
    if flavor not in valid:
        raise ValueError(f"flavor must be one of {sorted(valid)}, got '{flavor}'.")


def validate_max_iters(max_iters):
    """Checks the iteration budget: a positive integer."""
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError(f"max_iters must be a positive integer, got {max_iters}.")


def validate_linkage(linkage):
    """Checks the linkage: 'single', 'complete', 'average' or 'ward'."""
    valid = {"single", "complete", "average", "ward"}
    if linkage not in valid:
        raise ValueError(f"linkage must be one of {sorted(valid)}, got '{linkage}'.")
