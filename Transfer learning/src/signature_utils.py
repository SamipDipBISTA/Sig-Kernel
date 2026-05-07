import numpy as np
import iisignature


def build_path(log_prices_window, log_volumes_window):
    n = len(log_prices_window)
    time = np.linspace(0, 1, n)
    path = np.column_stack([time, log_prices_window, log_volumes_window])
    return path


def get_signature_dim(path_dim=3, order=2):
    return iisignature.siglength(path_dim, order)


def compute_signature_features(log_prices, log_volumes, lag, order):
    n = len(log_prices)
    n_valid = n - lag
    sig_dim = get_signature_dim(3, order)
    features = np.empty((n_valid, sig_dim))

    for i, t in enumerate(range(lag - 1, n - 1)):
        idx_start = t - lag + 1
        idx_end = t + 1
        window_p = log_prices[idx_start:idx_end]
        window_v = log_volumes[idx_start:idx_end]
        path = build_path(window_p, window_v)
        features[i] = iisignature.sig(path, order)

    return features
