"""
Transfer risk computation under the Gaussian model (paper Section 3.3).

Implements the Wasserstein-distance-based transfer risk decomposition of
Theorem 3.1 (Eq 27):

    Transfer Risk = bias² + (√var_S - √var_T)²

The bias term captures mismatch between output distributions, and the
variance term captures mismatch between input distributions.  Lower risk
indicates better transferability from source to target.

Reference: Cao, Gu, Guo, Rosenbaum (arXiv:2311.03283v1), Section 3.3.
"""

import numpy as np


def _estimate_gaussian_params(X, y):
    """Estimate mean, covariance, and cross-covariance of (X, y).

    Under the Gaussian model of paper Eq (15), the joint distribution of
    (X, y) is assumed multivariate normal.  This function estimates the
    required moments from data.

    Args:
        X (np.ndarray): Feature matrix of shape (n, d).
        y (np.ndarray): Target vector of shape (n,).

    Returns:
        tuple: (mu_X, mu_Y, Sigma_X, Sigma_XY)
            mu_X (np.ndarray): Mean of X, shape (d,).
            mu_Y (float): Mean of y.
            Sigma_X (np.ndarray): Covariance matrix of X, shape (d, d).
            Sigma_XY (np.ndarray): Cross-covariance between X and y, shape (d,).
    """
    d = X.shape[1]
    data = np.column_stack([X, y.reshape(-1, 1)])
    cov = np.cov(data, rowvar=False)
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(y)
    Sigma_X = cov[:d, :d]
    Sigma_XY = cov[:d, d]
    return mu_X, mu_Y, Sigma_X, Sigma_XY


def compute_transfer_risk(X_source, y_source, X_target, y_target, reg=1e-6):
    """Compute the Wasserstein-based transfer risk (paper Eq 27).

    Steps:
      1. Estimate Gaussian parameters for source and target.
      2. Solve for optimal linear predictors w_S and w_T (paper Eq 16).
      3. Compute bias² = (mu_TY - mu_SY - w_S·(mu_TX - mu_SX))².
      4. Compute variance term = (√(w_S·Σ_TX·w_S) - √(w_T·Σ_TX·w_T))².
      5. Return bias² + variance term.

    Args:
        X_source (np.ndarray): Source feature matrix (n_S, d).
        y_source (np.ndarray): Source target vector (n_S,).
        X_target (np.ndarray): Target feature matrix (n_T, d).
        y_target (np.ndarray): Target target vector (n_T,).
        reg (float): Small regularization added to covariance diagonals for
            numerical stability when solving linear systems.

    Returns:
        float: Transfer risk value (non-negative).
    """
    mu_SX, mu_SY, Sigma_SX, Sigma_SXY = _estimate_gaussian_params(X_source, y_source)
    mu_TX, mu_TY, Sigma_TX, Sigma_TXY = _estimate_gaussian_params(X_target, y_target)

    d = X_source.shape[1]
    # Regularize covariance matrices for numerical stability
    Sigma_SX_reg = Sigma_SX + reg * np.eye(d)
    Sigma_TX_reg = Sigma_TX + reg * np.eye(d)

    # Optimal linear predictors (paper Eq 16): w = Σ_X^{-1} Σ_XY
    w_S = np.linalg.solve(Sigma_SX_reg, Sigma_SXY)
    w_T = np.linalg.solve(Sigma_TX_reg, Sigma_TXY)

    # Bias term: output-distribution mismatch (paper Eq 27, first term)
    bias = mu_TY - mu_SY - w_S @ (mu_TX - mu_SX)
    bias_sq = bias ** 2

    # Variance term: input-distribution mismatch (paper Eq 27, second term)
    var_source = w_S @ Sigma_TX @ w_S
    var_target = w_T @ Sigma_TX @ w_T

    # Guard against negative values from numerical issues
    if var_source < 0:
        var_source = 0.0
    if var_target < 0:
        var_target = 0.0

    var_term = (np.sqrt(var_source) - np.sqrt(var_target)) ** 2

    return float(bias_sq + var_term)
