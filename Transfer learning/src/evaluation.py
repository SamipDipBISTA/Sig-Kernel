import numpy as np


def compute_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_correlation(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_all_metrics(y_true, y_pred):
    return {
        "mse": compute_mse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "corr": compute_correlation(y_true, y_pred),
    }
