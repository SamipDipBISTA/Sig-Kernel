#%%
import sys
import os
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
import iisignature
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nwsig import NWSignatureRegressor


METRICS = {
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
}

def split_train_val_test(X, y, delay=0, max_window=0, train_start=0, train_end=None, val_start=None, val_end=None, verbose=True, standard_scale = False):
    """
    Split data into training, validation, and test sets with optional differencing and standardization.

    X: Feature array of shape (n_samples, n_features).
    y: Target array of shape (n_samples,).
    delay: Integer number of time steps to compute differences for y.
    max_window: Maximum window length used for sliding window features.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_start: Starting index of the validation set.
    val_end: Ending index of the validation set.
    verbose: Boolean to print shapes of splits.
    standard_scale: Boolean indicating whether to standardize X features.
    Returns: Tuple of arrays for training, validation, and test sets.
             If val_end is provided: X_train, y_train, X_val, y_val, X_test, y_test.
             Otherwise: X_train, y_train, X_test, y_test.
    """
    # Validate input indices
    if train_end and val_end and train_end >= val_end:
        raise ValueError('Train end index must be less than Validation end index')
    
    # Training data
    X_train = X[train_start:train_end, :]
    
    if delay > 0:
        y_train = (y[train_start + max_window:train_end + max_window] - 
                   y[train_start + max_window - delay:train_end + max_window - delay])
    else:
        y_train = y[train_start + max_window:train_end + max_window]
    
    if val_end:
        # Validation set
        X_val = X[val_start:val_end, :]
        y_val = (y[val_start + max_window:val_end + max_window] - 
                 y[val_start - delay + max_window:val_end + max_window - delay])
        # Test set
        X_test = X[val_end:, :]
        y_test = y[val_end:]  # y_test is not the diff but y_train is

        if standard_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        if verbose:
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_val shape:", X_val.shape)
            print("y_val shape:", y_val.shape)
            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        # Test set (no validation)
        X_test = X[train_end:, :]
        y_test = y[train_end:]  # y_test is not the diff but y_train is
        if standard_scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        if verbose:
            print("X_train shape:", X_train.shape)
            print("y_train shape:", y_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_test shape:", y_test.shape)
        
        return X_train, y_train, X_test, y_test



def local_reg_nw_signature(
    X, 
    y,
    X0,
    delay=0, 
    max_window=0, 
    train_start=0, 
    train_end=None,
    val_start=None,
    val_end=None,
    bandwidth = None,
    bandwidth_grid = np.linspace(0.001, 10, 50),
    tune=True,
    cv = 3,
    verbose=True,
    standard_scale=False
):
    """
    Fit a Nadaraya–Watson signature-based regressor with optional bandwidth tuning and compute errors.

    X: Feature array of shape (n_samples, n_features).
    y: Target array of shape (n_samples,).
    X0: Auxiliary initial condition array for NW regression.
    delay: Integer number of time steps for differencing.
    max_window: Maximum sliding window length used for features.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_start: Starting index of the validation set.
    val_end: Ending index of the validation set.
    bandwidth: Kernel bandwidth for NW regression. If None and tune=True, it will be optimized.
    bandwidth_grid: Array or list of candidate bandwidths for tuning.
    tune: Boolean indicating whether to tune bandwidth on validation set.
    cv: Number of cross-validation folds used in NW regressor.
    verbose: Boolean controlling printing of shapes and errors.
    standard_scale: Boolean indicating whether to standardize features before regression.
    Returns: Dictionary containing training and test errors, predictions, feature splits, and best bandwidth.
    """

    X = np.array(X)
    y = np.array(y)

    if tune and val_end is not None:

        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
            X, 
            y, 
            delay=delay,
            max_window=max_window,
            train_start = train_start, 
            train_end = train_end, 
            val_start = val_start,
            val_end = val_end, 
            verbose = verbose,
            standard_scale=standard_scale,
        )

        X0_train = X0[train_start : train_end]
        X0_val   = X0[val_start : val_end]
        X0_test  = X0[val_end :]

        errors = {}

        for bw in bandwidth_grid:

            nw_regressor = NWSignatureRegressor(
                signature_level=3,
                bandwidth = bw,
                input_type='signatures',
                cv_folds=cv,
            )

            nw_regressor.fit(X_train, X0_train, y_train)
            y_pred_val_diff = nw_regressor.predict(X_val, X0_val)

            if delay > 0:
                y_past_val = y[train_end+max_window-delay : train_end+max_window-delay+len(y_pred_val_diff)]
                y_pred_val = y_pred_val_diff + y_past_val
            else:
                y_pred_val = y_pred_val_diff

            errors[bw] = mean_squared_error(y_val, y_pred_val)

        bandwidth = min(errors, key=errors.get)

    X_train, y_train, X_test, y_test = split_train_val_test(
        X, 
        y, 
        delay=delay,
        max_window=max_window,
        train_start = train_start, 
        train_end = train_end, 
        val_end = None, 
        verbose = verbose,
        standard_scale = standard_scale,
    )

    X0_train = X0[train_start : train_end]
    X0_test  = X0[train_end :]
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X0_train:", X0_train.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("X0_test:", X0_test.shape)
    error_train = {}
    error_test = {}

    nw_regressor = NWSignatureRegressor(
        signature_level=3,
        bandwidth = bandwidth,
        input_type='signatures',
        cv_folds=cv,
    )

    nw_regressor.fit(X_train, X0_train, y_train)

    y_pred_train_diff = nw_regressor.predict(X_train, X0_train)
    y_pred_test_diff  = nw_regressor.predict(X_test, X0_test)

    if delay > 0:
        y_past_train = y[train_start+max_window-delay : train_start+max_window-delay+len(y_pred_train_diff)]
        y_pred_train = y_pred_train_diff + y_past_train
    else:
        y_pred_train = y_pred_train_diff

    if delay > 0:
        y_past_test = y[-y_pred_test_diff.shape[0]-delay : -delay]
        y_pred_test = y_pred_test_diff + y_past_test
    else:
        y_pred_test = y_pred_test_diff

    for metric in METRICS:
        error_train[metric] = METRICS[metric](y_train, y_pred_train)
        error_test[metric]  = METRICS[metric](y_test, y_pred_test)

    if verbose:
        print("Train errors:")
        for metric, val in error_train.items():
            print(f"{metric}: {val:.4f}")
        print("----------------")
        print("Test errors:")
        for metric, val in error_test.items():
            print(f"{metric}: {val:.4f}")
    print('-----------------------------------------')

    return {
        "error_train": error_train,
        "error_test": error_test,
        "y_pred_test": y_pred_test,
        "y_pred_train": y_pred_train,
        "y_test": y_test,
        "y_train": y_train,
        "X_test": X_test,
        "X_train": X_train,
        "best_bandwith": bandwidth
    }




def ridge_regression_diff_and_reconstruct(
    X, 
    y,
    tune=False,
    max_window = 0, #if EFM signature was computed
    train_start = None, #train_val_test
    train_end = None, #train_val_test
    val_start = None,
    val_end=None,  #train_val_test
    penalties=None, #train_val_test
    delay = 0, #Delta for the target
    alpha=None, 
    verbose=False,
    fit_intercept = True,
    standard_scale = False,
):
    """
    Fit a Ridge regression model with optional differencing on the target, perform hyperparameter tuning, and compute train/test metrics.

    X: Feature array of shape (n_samples, n_features).
    y: Target array of shape (n_samples,).
    tune: Boolean indicating whether to tune Ridge penalty using validation set.
    max_window: Maximum window length used for sliding window or EFM features.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_start: Starting index of the validation set.
    val_end: Ending index of the validation set.
    penalties: List of Ridge alpha values to evaluate if tune=True.
    delay: Integer number of time steps for differencing the target.
    alpha: Ridge alpha to use if not tuning.
    verbose: Boolean controlling printing of shapes and errors.
    fit_intercept: Boolean indicating whether Ridge model fits intercept.
    standard_scale: Boolean indicating whether to standardize features before regression.
    Returns: Dictionary containing training and test errors, predictions, feature splits, and best penalty (alpha).
    """


    X = np.array(X)
    y = np.array(y)

    #Dictionaries to store metrics
    error_train = {}
    error_test = {}
    
    if tune:
        #Parameter tuning for ridge
        best_penalty = penalties[0]
        best_score = np.inf
        if train_start is None or train_end is None or val_end is None or penalties is None or len(penalties) == 0:
            raise ValueError('Specify the necessary parameters, check the function for more info')

        X_train, y_train, X_val, y_val, X_test, y_test=split_train_val_test(
                                                                            X, 
                                                                            y, 
                                                                            delay=delay,
                                                                            max_window=max_window,
                                                                            train_start = train_start, 
                                                                            train_end = train_end, 
                                                                            val_start = val_start,
                                                                            val_end = val_end, 
                                                                            verbose = verbose,
                                                                            standard_scale = standard_scale,
                                                                            )
    
        
        for penalty in penalties:
            model = Ridge(fit_intercept=fit_intercept, alpha=penalty)
            model.fit(X_train, y_train)
            y_val_pred_diff = model.predict(X_val)
            y_val_pred = y_val_pred_diff + y[train_end+max_window-delay : val_end+max_window-delay]
            y_val_true = y_val

            score = METRICS["rmse"](y_val_true, y_val_pred)
            if score < best_score:
                best_score = score
                alpha = penalty
                 

    X_train, y_train, X_test, y_test =  split_train_val_test(
                                                        X, 
                                                        y, 
                                                        delay=delay,
                                                        max_window=max_window,
                                                        train_start = train_start, 
                                                        train_end = train_end, 
                                                        val_end = None, 
                                                        verbose = verbose,
                                                        standard_scale = standard_scale,
                                                        )
    model_best = Ridge(fit_intercept=fit_intercept, alpha=alpha)
    model_best.fit(X_train, y_train)

    y_pred_train_diff = model_best.predict(X_train)

    if delay == 0:
        y_past_train = y[train_start+max_window : train_start+max_window+len(y_pred_train_diff)]
        y_pred_train = y_pred_train_diff
    else:
        y_past_train = y[train_start+max_window-delay : train_start+max_window-delay+len(y_pred_train_diff)]
        y_pred_train = y_pred_train_diff + y_past_train

    y_train = y[train_start+max_window:train_start+len(y_pred_train_diff)+max_window]

    y_pred_test_diff = model_best.predict(X_test)
    if delay == 0:
        y_past_test = y[-y_pred_test_diff.shape[0]:]
        y_pred_test = y_pred_test_diff
    else:
        y_past_test = y[-y_pred_test_diff.shape[0]-delay : -delay]
        y_pred_test = y_pred_test_diff + y_past_test
    
    #Compute metrics
    for metric in METRICS:
        error_train[metric] = METRICS[metric](y_train, y_pred_train)
        error_test[metric]  = METRICS[metric](y_test, y_pred_test)

    if verbose:
        print("Train errors:")
        for metric, val in error_train.items():
            print(f"{metric}: {val:.4f}")
        print("----------------")
        print("Test errors:")
        for metric, val in error_test.items():
            print(f"{metric}: {val:.4f}")

    return {
        "error_train": error_train,
        "error_test": error_test,
        "y_pred_test": y_pred_test,
        "y_pred_train": y_pred_train,
        "y_test": y_test,
        "y_train": y_train,
        "X_test": X_test,
        "X_train": X_train,
        "best_penalty": best_penalty
    }


def simple_ridge(X, y,  train_start, train_end, alpha, verbose=False, fit_intercept=True):
    error_train = {}
    error_test = {}

    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]
    X_test = X[train_end:]
    y_test = y[train_end:]

    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    for metric in METRICS:
        error_train[metric] = METRICS[metric](y_train, y_pred_train)
        error_test[metric] = METRICS[metric](y_test, y_pred_test)

    if verbose:
        print("Train errors:")
        for metric, val in error_train.items():
            print(f"{metric}: {val:.4f}")
        print("----------------")
        print("Test errors:")
        for metric, val in error_test.items():
            print(f"{metric}: {val:.4f}")

    return {
        "error_train": error_train,
        "error_test": error_test,
        "y_pred_test": y_pred_test,
        "y_pred_train": y_pred_train,
        "y_test": y_test,
        "y_train": y_train,
        "X_test": X_test,
        "X_train": X_train
    }

#%%
if __name__ == "__main__":
#%%
    import iisignature as ii

    N = 600
    T = 10

    level = 2
    t = np.linspace(0, 2 * np.pi, T)
    X = np.array([np.sin(t + np.random.rand() * 2 * np.pi) for _ in range(N)])
    #X = np.random.randn(N, T)

    y = X[:, -1]       
    X0 = X[:,0]      

    X_sig = []
    for i in range(N):
        time_augmented = np.vstack([X[i], np.linspace(0, 1, T)])
        X_sig.append(iisignature.sig(time_augmented.T, level))
    X_sig = np.array(X_sig)

    print("X:", X.shape)
    print("X_sig:", X_sig.shape)
    print("y:", y.shape)
    print("X0:", X0.shape)

#%%
    train_start = 0
    train_end = val_start =300
    val_end = 400

    results = local_reg_nw_signature(
        X=X_sig,
        y=y,
        X0=X0.reshape(-1,1),
        delay=0,
        max_window=0,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        bandwidth_grid=np.linspace(0.001, 10, 10),
        tune=True,
        cv=3,
        verbose=True
    )

    print(results['error_train'], results['error_test'])
# %%
