
import sys
import os
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
import importlib
import numpy as np
from signature.path_signature import path_to_fm_signature
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

from nwsig import NWSignatureRegressor

import experiments.src.signature_functions as signature_functions
importlib.reload(signature_functions)
import experiments.src.regression_functions as regression_functions
importlib.reload(regression_functions)
from experiments.src.signature_functions import *
from experiments.src.regression_functions import *
from experiments.src.optimisation_functions import *

def optimize_lambda(path, y, trunc, time_grid, bounds, train_start, train_end, val_start, val_end, model, NW=False, X0=None, standard_scale = False):
    """
    Optimize the lambda parameter for signature-based feature extraction by minimizing validation mean squared error.

    path: Multidimensional time-series array of shape (n_samples, dim).
    y: Target values aligned with the path observations.
    trunc: Truncation level of the signature transform.
    time_grid: Time discretization grid used to compute the signature.
    bounds: Tuple (a, b) defining the lower and upper bounds for lambda.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_start: Starting index of the validation set.
    val_end: Ending index of the validation set.
    model: Regression model implementing fit() and predict().
    NW: Boolean indicating whether a Nadaraya–Watson type model is used.
    X0: Auxiliary initial condition input required when NW=True.
    standard_scale: Boolean indicating whether to standardize features before training.
    Returns: Validation mean squared error for a given lambda.
    """
        
    dim = path.shape[1]
    a, b = bounds
    
    def loss(x):
        StatSig = path_to_fm_signature(path=path, trunc=trunc, t_grid=time_grid, lam=x).array.T

        X_train = StatSig[train_start:train_end]
        y_train = y[train_start:train_end]
        X_val = StatSig[val_start:val_end]
        y_val = y[val_start:val_end]
        if standard_scale:
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)

        if NW:
            X0_train = X0[train_start:train_end, 0].reshape(-1, 1)
            X0_val = X0[val_start:val_end, 0].reshape(-1, 1)
            model.fit(X_train, X0_train, y_train)
            y_val_pred = model.predict(X_val, X0_val)
        else:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)

        return mean_squared_error(y_val, y_val_pred)

    
    optimizer = minimize(loss, x0=np.ones(dim) * 4, bounds=([a, b],) * dim, method='Powell')
    return optimizer.x, optimizer.fun

def ridge_lambda_grid_search(
    alphas,
    path,
    y,
    trunc,
    time_grid,
    lambda_bounds=(0.001, 50),
    lambdas=None,
    train_start=0,
    train_end=None,
    val_start=None,
    val_end=None,
    delay=0,
    grid_search=False,
    verbose=True,
    standard_scale=False,
):
    """
    Perform joint optimization of Ridge penalty (alpha) and signature decay parameter (lambda) using either grid search or nested optimization.

    alphas: List or array of Ridge regularization parameters to evaluate.
    path: Multidimensional time-series array of shape (n_samples, dim).
    y: Target values aligned with the path observations.
    trunc: Truncation level of the signature transform.
    time_grid: Time discretization grid used to compute the signature.
    lambda_bounds: Tuple (lower, upper) bounds for lambda optimization when not using grid search.
    lambdas: Optional list of lambda values used for explicit grid search.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_start: Starting index of the validation set.
    val_end: Ending index of the validation set.
    delay: Integer time delay applied in regression reconstruction.
    grid_search: Boolean indicating whether to perform explicit lambda grid search.
    verbose: Boolean controlling progress printing.
    standard_scale: Boolean indicating whether to standardize features before regression.
    Returns: Best hyperparameters and corresponding validation error.
    """
    results = []

    if grid_search and lambdas is not None:
        hyperparameters = {}
        for lam in lambdas:
            start_time = time.time()
            X = path_to_fm_signature(path=path, trunc=trunc, t_grid=time_grid, lam=lam).array.T
            if val_start is None:
                val_start=train_end
            results = ridge_regression_diff_and_reconstruct(
                        X=X,
                        y=y,
                        tune=True,                                                                
                        train_start=train_start,
                        train_end=train_end,
                        val_start=val_start,
                        val_end= val_end,
                        penalties=alphas,
                        delay = delay,
                        verbose=True,
                        fit_intercept= True,
                        standard_scale=standard_scale
                    )
            elapsed = time.time() - start_time
            print(f"lambda={lam:.4g} done in {elapsed:.2f} sec")
            best_alpha = results['best_penalty']
            hyperparameters[(lam, best_alpha)] = results['error_train']['mse']
        best_combo = min(hyperparameters, key=hyperparameters.get)
        best_score = hyperparameters[best_combo]

        if verbose:
            print(f"Best lambda: {best_combo[0]}, Best alpha: {best_combo[1]}, Min error RMSE: {np.sqrt(best_score)}")
        return best_combo[0], best_combo[1], best_score
                
    else:
        for alpha in alphas:
            start_time = time.time()
        
            if verbose:
                print(f"Processing for alpha: {alpha}")
            model = Ridge(alpha=alpha)
            best_lambda, best_score = optimize_lambda(
                path=path,
                y=y,
                trunc=trunc,
                time_grid=time_grid,
                bounds=lambda_bounds,
                train_start=train_start,
                val_start=val_start,
                train_end=train_end,
                val_end=val_end,
                model=model,
                standard_scale=standard_scale,
            )
            
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"alpha={alpha:.4g} done in {elapsed:.2f} sec")
                print(f"(alpha={alpha:.4g}, lambda={best_lambda}, MSE={best_score:.6f})")
            
            results.append((alpha, best_lambda, best_score))
        
        best_alpha, best_lambda, best_score = min(results, key=lambda x: x[2])
        
        if verbose:
            print(f"(best alpha={best_alpha:.4g}, best lambda={best_lambda}, best RMSE={np.sqrt(best_score):.6f})")
    
        return best_alpha, best_lambda, best_score, results

def NW_bandwith_lambda_grid_search(
    path,
    y,
    X0,
    trunc,
    time_grid,
    lambda_bounds=(0.001, 50),
    lambdas=None,
    train_start=0,
    train_end=None,
    val_start=None,
    val_end=None,
    bandwidth_grid=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    grid_search=False,
    verbose=True,
    standard_scale=False,
):
    """
    Perform joint optimization of kernel bandwidth and signature decay parameter (lambda) for Nadaraya–Watson signature regression.

    path: Multidimensional time-series array of shape (n_samples, dim).
    y: Target values aligned with the path observations.
    X0: Auxiliary initial condition features required by the NW regressor.
    trunc: Truncation level of the signature transform.
    time_grid: Time discretization grid used to compute the signature.
    lambda_bounds: Tuple (lower, upper) bounds for lambda optimization when not using grid search.
    lambdas: Optional list of lambda values used for explicit grid search.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_start: Starting index of the validation set.
    val_end: Ending index of the validation set.
    bandwidth_grid: List of kernel bandwidth values to evaluate.
    grid_search: Boolean indicating whether to perform explicit lambda grid search.
    verbose: Boolean controlling progress printing.
    standard_scale: Boolean indicating whether to standardize features before regression.
    Returns: Best bandwidth, best lambda, and corresponding validation error.
    """
    
    results = []
    hyperparameters={}
    if grid_search and lambdas is not None:
        for bandwidth in bandwidth_grid:
            start_time = time.time()
            for lam in lambdas:
                model = NWSignatureRegressor(
                signature_level=trunc,
                bandwidth = bandwidth,
                input_type='signatures',
                efm_signatures = True,
                )

                if val_start is None:
                    val_start=train_end

                X_train = X[train_start:train_end]
                X0_train = X0[train_start:train_end]
                y_train = y[train_start:train_end]

                X_val = X[val_start:val_end]
                X0_val = X0[val_start:val_end]
                y_val = y[val_start:val_end]
                if standard_scale:
                    scaler_X = StandardScaler()
                    scaler_X0 = StandardScaler()

                    X_train = scaler_X.fit_transform(X_train)
                    X_val = scaler_X.transform(X_val)

                    X0_train = scaler_X0.fit_transform(X0_train)
                    X0_val = scaler_X0.transform(X0_val)

                model.fit(X_train, X0_train, y_train)

                y_pred = model.predict(X_val, X0_val)

                error = np.mean((y_val - y_pred) ** 2)
                
                hyperparameters[(lam, bandwidth)] = error
            elapsed = time.time() - start_time
            print(f"bandwidth={bandwidth:.4g} done in {elapsed:.2f} sec")
        best_combo = min(hyperparameters, key=hyperparameters.get)
        best_score = hyperparameters[best_combo]

        if verbose:
            print(f"Best window: {best_combo[0]}, Best bandwidth: {best_combo[1]}, Min error: {best_score}")

        return best_combo[0], best_combo[1],  best_score
                
    
    else:
        for bandwidth in bandwidth_grid:
            start_time = time.time()
            
            model = NWSignatureRegressor(
                signature_level=3,
                bandwidth=bandwidth,
                input_type='signatures',
            )                           #no standard scale here because it's included in optimize lambda
            
            best_lambda, best_score = optimize_lambda(
                path=path,
                y=y,
                trunc=trunc,
                time_grid=time_grid,
                bounds=lambda_bounds,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                model=model,
                NW=True,
                X0=X0,
                standard_scale=standard_scale,
            )
            
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"bandwidth={bandwidth:.4g} done in {elapsed:.2f} sec")
                print(f"(bandwidth={bandwidth:.4g}, lambda={best_lambda}, MSE={best_score:.6f})")
            
            results.append((bandwidth, best_lambda, best_score))
    
        best_bandwidth, best_lambda, best_score = min(results, key=lambda x: x[2])
        
        if verbose:
            print(f"(best bandwidth={best_bandwidth:.4g}, best lambda={best_lambda}, best RMSE={np.sqrt(best_score):.6f})")
        
        return best_bandwidth, best_lambda, best_score, results

def ridge_window_grid_search(
        windows,
        alphas,
        path,
        y,
        train_start = 0,
        train_end = None,
        val_end=None,
        trunc_order=3,
        delay = 0,
        verbose=True,
        standard_scale = False,
        ):
    """
    Perform grid search over sliding window sizes and Ridge penalties to minimize validation error.

    windows: List of integers specifying sliding window sizes.
    alphas: List or array of Ridge regularization parameters to evaluate.
    path: Multidimensional time-series array of shape (n_samples, dim).
    y: Target values aligned with the path observations.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_end: Ending index of the validation set.
    trunc_order: Truncation order for signature computation within each window.
    delay: Integer time delay applied in regression reconstruction.
    verbose: Boolean controlling progress printing.
    standard_scale: Boolean indicating whether to standardize features before regression.
    Returns: Best window size, best Ridge alpha, and corresponding error.
    """
    hyperparameters = {}
    for window in windows:
        _, X = compute_sliding_window_signatures_optimized(
                covariates = path,
                window = window,
                trunc_order=trunc_order,
            )


        if val_start is None:
            val_start=train_end
        results = ridge_regression_diff_and_reconstruct(
                    X=X,
                    y=y,
                    tune=True,                                                                
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end= val_end,
                    penalties=alphas,
                    delay = delay,
                    verbose=True,
                    fit_intercept= True,
                    standard_scale = standard_scale
                )
        best_alpha = results['best_penalty']
        hyperparameters[(window, best_alpha)] = results['error_train']['mse']
    best_combo = min(hyperparameters, key=hyperparameters.get)
    best_score = hyperparameters[best_combo]

    if verbose:
        print(f"Best window: {best_combo[0]}, Best alpha: {best_combo[1]}, Min error RMSE: {np.sqrt(best_score)}")
    return best_combo[0], best_combo[1], best_score


def NW_bandwidth_window_grid_search(
        windows,
        path,
        y,
        X0,
        trunc_order = 3,
        train_start=None,
        train_end = None,
        val_start=None,
        val_end=None,
        bandwidth_grid = [0.001, 0.01, 0.1],
        verbose=True,
        standard_scale = False,
        ):
    """
    Perform grid search over sliding window sizes and NW kernel bandwidths to minimize validation error.

    windows: List of integers specifying sliding window sizes.
    path: Multidimensional time-series array of shape (n_samples, dim).
    y: Target values aligned with the path observations.
    X0: Auxiliary initial condition features required by the NW regressor.
    trunc_order: Truncation order for signature computation within each window.
    train_start: Starting index of the training set.
    train_end: Ending index of the training set.
    val_start: Starting index of the validation set.
    val_end: Ending index of the validation set.
    bandwidth_grid: List of kernel bandwidth values to evaluate.
    verbose: Boolean controlling progress printing.
    standard_scale: Boolean indicating whether to standardize features before regression.
    Returns: Best window size, best bandwidth, and corresponding validation error.
    """
    hyperparameters = {}
    for window in windows:
        for bandwidth in bandwidth_grid:

            _, X = compute_sliding_window_signatures_optimized(
                covariates = path,
                window = window,
                trunc_order=trunc_order,
            ) 

            model = NWSignatureRegressor(
                signature_level=trunc_order,
                bandwidth = bandwidth,
                input_type='signatures',
            )

            if val_start is None:
                val_start=train_end

            X_train = X[train_start:train_end]
            X0_train = X0[train_start:train_end]
            y_train = y[train_start:train_end]

            X_val = X[val_start:val_end]
            X0_val = X0[val_start:val_end]
            y_val = y[val_start:val_end]
            if standard_scale:
                scaler_X = StandardScaler()
                scaler_X0 = StandardScaler()

                X_train = scaler_X.fit_transform(X_train)
                X_val = scaler_X.transform(X_val)

                X0_train = scaler_X0.fit_transform(X0_train)
                X0_val = scaler_X0.transform(X0_val)

            model.fit(X_train, X0_train, y_train)

            y_pred = model.predict(X_val, X0_val)

            error = np.mean((y_val - y_pred) ** 2)

            hyperparameters[(window, bandwidth)] = error

    best_combo = min(hyperparameters, key=hyperparameters.get)
    best_score = hyperparameters[best_combo]

    if verbose:
        print(f"Best window: {best_combo[0]}, Best bandwidth: {best_combo[1]}, Min error: {best_score}")

    return best_combo[0], best_combo[1],  best_score


