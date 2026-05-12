"""
Ridge regression solvers for direct and transfer learning.

Implements standard ridge regression (paper Eq 29) and the transfer ridge
regression (paper Eq 31) where the source-model parameters act as an offset
in the penalty term, pulling the target solution towards the source estimate.

All solvers use the closed-form solution with /n normalization as in the paper.
"""

import numpy as np


def ridge_solve(X, y, alpha):
    """Solve standard ridge regression: theta = (X^T X/n + alpha I)^{-1} (X^T y/n).

    The /n normalization matches the formulation in paper Eq (29).

    Args:
        X (np.ndarray): Feature matrix of shape (n, d).
        y (np.ndarray): Target vector of shape (n,).
        alpha (float): Ridge regularization strength λ.

    Returns:
        np.ndarray: Coefficient vector theta of shape (d,).
    """
    n, d = X.shape
    A = (X.T @ X) / n + alpha * np.eye(d)
    b = (X.T @ y) / n
    theta = np.linalg.solve(A, b)
    return theta


def transfer_ridge_solve(X, y, theta_S, alpha):
    """Solve transfer ridge regression with source model offset.

    Minimizes ||X·θ - y||²/n + α·||θ - θ_S||², which yields the closed form
        θ = (X^T X/n + αI)^{-1} (X^T y/n + α·θ_S)

    This matches paper Eq (31).  The term α·θ_S pulls the solution toward
    the source-model parameters θ_S.

    Args:
        X (np.ndarray): Feature matrix of shape (n, d).
        y (np.ndarray): Target vector of shape (n,).
        theta_S (np.ndarray): Source-model parameters of shape (d,).
        alpha (float): Transfer regularization strength λ_T.

    Returns:
        np.ndarray: Coefficient vector theta of shape (d,).
    """
    n, d = X.shape
    A = (X.T @ X) / n + alpha * np.eye(d)
    b = (X.T @ y) / n + alpha * theta_S  # paper Eq (31): pulls towards theta_S
    theta = np.linalg.solve(A, b)
    return theta


def predict(X, theta):
    """Linear prediction: ŷ = X·θ.

    Args:
        X (np.ndarray): Feature matrix of shape (n, d).
        theta (np.ndarray): Coefficient vector of shape (d,).

    Returns:
        np.ndarray: Predicted values of shape (n,).
    """
    return X @ theta
