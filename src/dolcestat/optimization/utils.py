"""Row sampling used by the stochastic and mini-batch flavors of gradient descent."""

import numpy as np


def sample_rows_Xy(
    X: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draws ``n`` distinct rows of ``X`` and the matching entries of ``y``.

    Rows are picked without replacement from NumPy's global random generator,
    so seed it (``np.random.seed``) to repeat a run.
    """
    idx = np.random.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]
