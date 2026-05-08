import numpy as np


def ridge_solve(X, y, alpha):
    n, d = X.shape
    A = (X.T @ X) / n + alpha * np.eye(d)
    b = (X.T @ y) / n
    theta = np.linalg.solve(A, b)
    return theta


def transfer_ridge_solve(X, y, theta_S, alpha):
    n, d = X.shape
    A = (X.T @ X) / n + alpha * np.eye(d)
    b = (X.T @ y) / n + alpha * theta_S
    theta = np.linalg.solve(A, b)
    return theta


def predict(X, theta):
    return X @ theta
