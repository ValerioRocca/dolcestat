import numpy as np

# --- Loss functions and their gradients ---


def compute_mse_loss(y_true, y_pred):
    n = len(y_true)
    mse_loss = np.sum((y_true - y_pred) ** 2) / n
    return mse_loss


def compute_mse_gradient(X, y_true, y_pred):
    n = len(y_true)
    error = y_pred - y_true
    mse_gradient = (2 / n) * np.matmul(X.T, error)
    return mse_gradient


def compute_mse_hessian(X, y_true, y_pred):
    n = len(y_true)
    mse_hessian = (2 / n) * np.matmul(X.T, X)
    return mse_hessian


def identity(z):
    return z


def compute_bce_loss(y_true, y_pred):
    n = len(y_true)
    # To avoid log(0), we clip y_pred to a small value
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    bce_loss = (
        -np.sum(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        / n
    )
    return bce_loss


def compute_bce_gradient(X, y_true, y_pred):
    n = len(y_true)
    # To avoid division by zero, we clip y_pred to a small value
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    error = y_pred_clipped - y_true
    bce_gradient = (1 / n) * np.matmul(X.T, error)
    return bce_gradient


def compute_bce_hessian(X, y_true, y_pred):
    n = len(y_true)
    # To avoid extreme values, we clip y_pred to a small value
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    # Diagonal weights from the variance of each prediction: p * (1 - p)
    weights = y_pred_clipped * (1 - y_pred_clipped)
    bce_hessian = (1 / n) * np.matmul(X.T * weights, X)
    return bce_hessian


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
