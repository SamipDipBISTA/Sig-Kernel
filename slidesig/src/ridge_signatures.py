"""
ridge_signatures.py — Function for fitting the ridge regression model on sliding-window signature features.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from src.signatures import sig_sliding

METRICS = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
}

def ridge_signatures(
    covariates,
    target,
    window,
    delay,
    n_train,
    n_valid=None,
    trunc_order=4,
    penalties=(1.0,),
    tune_penalty=False,
    metric="rmse" 
):
    """
    Ridge regression on truncated signatures with optional ridge regularization parameter (penalty) tuning.

    Parameters
    ----------
    covariates : np.ndarray
        Covariate series for signature calculation.
    target : np.ndarray
        Target time series.
    window : int
        Sliding window size.
    delay : int
        Forecasting delay for target increments.
    n_train : int
        Number of training samples.
    n_valid : int, optional
        Number of validation samples (required if tune_penalty=True).
    trunc_order : int, default=4
        Truncation order for signatures.
    penalties : tuple of float, default=(1.0,)
        Candidate ridge penalties.
    tune_penalty : bool, default=False
        Whether to tune penalty using the validation set.
    metric : {"rmse","mae","mape"}, default="rmse"
        Metric used for tuning and reporting.
    """

    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}', choose from {list(METRICS.keys())}")

    # Compute truncated signatures
    signatures = sig_sliding(covariates=covariates, window=window, trunc_order=trunc_order, drop=True)

    # Train-validation-test split
    if tune_penalty:
        if n_valid is None:
            raise ValueError("n_valid must be specified when tune_penalty=True")

        train_start = window
        train_end = n_train + window
        val_end = n_train + n_valid + window

        X_train = signatures[train_start:train_end, :]
        y_train = target[train_start:train_end] - target[train_start-delay:train_end-delay]

        X_val = signatures[train_end:val_end, :]
        y_val = target[train_end:val_end] - target[train_end-delay:val_end-delay]

        X_test = signatures[val_end:, :]

    else:
        train_start = window
        train_end = n_train + window

        X_train = signatures[train_start:train_end, :]
        y_train = target[train_start:train_end] - target[train_start-delay:train_end-delay]

        X_val, y_val = None, None
        X_test = signatures[train_end:, :]

    # Penalty tuning
    best_penalty = penalties[0]
    if tune_penalty:
        best_score = np.inf
        for penalty in penalties:
            model = Ridge(alpha=penalty)
            model.fit(X_train, y_train)
            y_val_pred_diff = model.predict(X_val)
            y_val_pred = y_val_pred_diff + target[train_end-delay:val_end-delay]
            y_val_true = target[train_end:val_end]

            score = METRICS[metric](y_val_true, y_val_pred)
            if score < best_score:
                best_score = score
                best_penalty = penalty

    # Final training (on the train + validation set if tuning)
    if tune_penalty:
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
    else:
        X_train_full = X_train
        y_train_full = y_train

    model_best = Ridge(alpha=best_penalty)
    model_best.fit(X_train_full, y_train_full)

    # Predictions
    y_pred_train_diff = model_best.predict(X_train_full)
    y_past_train = target[train_start-delay : train_start-delay+len(y_pred_train_diff)]
    y_pred_train = y_pred_train_diff + y_past_train
    y_true_train = target[train_start:train_start+len(y_pred_train_diff)]

    y_pred_test_diff = model_best.predict(X_test)
    y_past_test = target[train_end + (n_valid if tune_penalty else 0) - delay : -delay]
    y_pred_test = y_pred_test_diff + y_past_test
    y_true_test = target[train_end + (n_valid if tune_penalty else 0):]

    # Calculate the error metric
    error_train = METRICS[metric](y_true_train, y_pred_train)
    error_test  = METRICS[metric](y_true_test, y_pred_test)

    return {
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_true_test": y_true_test,
        "error_train": error_train,
        "error_test": error_test,
        "best_penalty": best_penalty,
    }