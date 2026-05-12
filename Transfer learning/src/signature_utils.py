"""
Path-signature feature construction via sliding windows (paper Section 4.1).

Implements the truncated signature transform (Definition 4.1 of the paper)
applied to a 3-dimensional path (time, log-price, log-volume) over rolling
windows of length L (lag).  Signature truncation order M and lag L are the
primary hyperparameters varied in Table 1.
"""

import numpy as np
import iisignature


def build_path(log_prices_window, log_volumes_window):
    """Construct a 3D path from (time, log_prices, log_volumes).

    The time dimension is uniformly scaled to [0, 1] as per paper Section 4.1.

    Args:
        log_prices_window (np.ndarray): 1D array of log-prices over the window.
        log_volumes_window (np.ndarray): 1D array of log-volumes over the window.

    Returns:
        np.ndarray: 2D array of shape (n, 3) — columns are (time, log_price, log_volume).
    """
    n = len(log_prices_window)
    time = np.linspace(0, 1, n)
    path = np.column_stack([time, log_prices_window, log_volumes_window])
    return path


def get_signature_dim(path_dim=3, order=2):
    """Return the dimension of a truncated signature.

    Args:
        path_dim (int): Dimension of the input path (default 3).
        order (int): Truncation order M.

    Returns:
        int: Total number of signature coefficients up to order M.
    """
    return iisignature.siglength(path_dim, order)


def compute_signature_features(log_prices, log_volumes, lag, order):
    """Compute sliding-window truncated signature features for a single stock.

    For each time t in [lag, n-1], a 3D path is built from the window
    [t-lag+1, t] of (log_prices, log_volumes) and its M-th order truncated
    signature is computed via iisignature.

    This yields (n - lag) feature vectors, each of dimension sig_dim, which
    are aligned so that features at time t predict the return at time t.

    Args:
        log_prices (np.ndarray): 1D array of log-prices.
        log_volumes (np.ndarray): 1D array of log-volumes.
        lag (int): Window length L for the sliding signature.
        order (int): Truncation order M of the signature.

    Returns:
        np.ndarray: Feature matrix of shape (n_valid, sig_dim), where
            n_valid = len(log_prices) - lag.
    """
    n = len(log_prices)
    if n < lag + 1:
        raise ValueError(
            f"Series too short for lag={lag}: need at least {lag + 1} points, got {n}"
        )
    n_valid = n - lag  # number of valid windows
    sig_dim = get_signature_dim(3, order)
    features = np.empty((n_valid, sig_dim))

    # For each time t from (lag-1) to (n-2), the window spans
    #   log_prices[t-lag+1 : t+1], log_volumes[t-lag+1 : t+1]
    # and the feature at index i predicts log_return at time t+1.
    for i, t in enumerate(range(lag - 1, n - 1)):
        idx_start = t - lag + 1
        idx_end = t + 1
        window_p = log_prices[idx_start:idx_end]
        window_v = log_volumes[idx_start:idx_end]
        path = build_path(window_p, window_v)
        features[i] = iisignature.sig(path, order)

    return features
