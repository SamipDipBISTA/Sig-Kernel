import numpy as np


def _estimate_gaussian_params(X, y):
    d = X.shape[1]
    data = np.column_stack([X, y.reshape(-1, 1)])
    cov = np.cov(data, rowvar=False)
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(y)
    Sigma_X = cov[:d, :d]
    Sigma_XY = cov[:d, d]
    return mu_X, mu_Y, Sigma_X, Sigma_XY


def compute_transfer_risk(X_source, y_source, X_target, y_target, reg=1e-6):
    mu_SX, mu_SY, Sigma_SX, Sigma_SXY = _estimate_gaussian_params(X_source, y_source)
    mu_TX, mu_TY, Sigma_TX, Sigma_TXY = _estimate_gaussian_params(X_target, y_target)

    d = X_source.shape[1]
    Sigma_SX_reg = Sigma_SX + reg * np.eye(d)
    Sigma_TX_reg = Sigma_TX + reg * np.eye(d)

    w_S = np.linalg.solve(Sigma_SX_reg, Sigma_SXY)
    w_T = np.linalg.solve(Sigma_TX_reg, Sigma_TXY)

    bias = mu_TY - mu_SY - w_S @ (mu_TX - mu_SX)
    bias_sq = bias ** 2

    var_source = w_S @ Sigma_TX @ w_S
    var_target = w_T @ Sigma_TX @ w_T

    if var_source < 0:
        var_source = 0.0
    if var_target < 0:
        var_target = 0.0

    var_term = (np.sqrt(var_source) - np.sqrt(var_target)) ** 2

    return float(bias_sq + var_term)
