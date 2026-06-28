import numpy as np


def sample_rows_X(X: np.ndarray, n: int) -> np.ndarray:
    idx = np.random.choice(X.shape[0], size=n, replace=False)
    return X[idx]


def sample_rows_Xy(
    X: np.ndarray, y: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]
