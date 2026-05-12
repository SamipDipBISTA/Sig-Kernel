"""
Evaluation metrics for the transfer learning experiments.

Provides standard regression metrics — MSE, R-squared, and Pearson
correlation — used to compare direct and transfer learning performance
as reported in paper Tables 1 and 2.
"""

import numpy as np


def compute_mse(y_true, y_pred):
    """Mean squared error.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: MSE = mean((y_true - y_pred)²).
    """
    return float(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true, y_pred):
    """Coefficient of determination (R-squared).

    R² = 1 - SS_res / SS_tot, where SS_res is the residual sum of squares
    and SS_tot is the total sum of squares around the mean.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: R² score. Returns 0.0 when SS_tot = 0.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_correlation(y_true, y_pred):
    """Pearson correlation coefficient between true and predicted values.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Correlation coefficient in [-1, 1]. Returns 0.0 when either
            series has zero variance.
    """
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_all_metrics(y_true, y_pred):
    """Compute MSE, R², and Pearson correlation in one call.

    Args:
        y_true (np.ndarray): Ground-truth target values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        dict: {"mse": float, "r2": float, "corr": float}.
    """
    return {
        "mse": compute_mse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "corr": compute_correlation(y_true, y_pred),
    }
