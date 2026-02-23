"""
ridge_signatures.py — Function for fitting the ridge regression model on sliding-window signature features.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    #root_mean_squared_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

from src.signatures import sig_sliding
import sys, os
sys.path.append(os.path.abspath(os.path.join("..", "..")))

from playground.helper import *


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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
    metric="rmse",
    signature_type = 'sw',
    lam = None,
    T = 1,
    max_window = 32*48, # max window set,
    scale = False,
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
    #Initialize parameters for EFM sig
#-------------------------------------------
    n_obs = len(covariates)
    t_grid = np.linspace(0, T, n_obs)
    t_grid = t_grid.reshape(1,-1)
#-------------------------------------------
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}', choose from {list(METRICS.keys())}")

    if signature_type == 'sw': # Signature+sliding window
        # Compute truncated signatures
        signatures = sig_sliding(covariates=covariates, window=window, trunc_order=trunc_order, drop=True)
        print('Signature shape:', signatures.shape)
        # Train-validation-test split
        if tune_penalty:
            if n_valid is None:
                raise ValueError("n_valid must be specified when tune_penalty=True")

            train_start = max_window
            train_end = n_train 
            val_end = n_train + n_valid

            X_train = signatures[train_start:train_end, :]
            y_train = target[train_start:train_end] - target[train_start-delay:train_end-delay]

            X_val = signatures[train_end:val_end, :]
            y_val = target[train_end:val_end] - target[train_end-delay:val_end-delay]

            X_test = signatures[val_end:, :]

            print("X_train:", X_train.shape) 
            print("X_val:", X_val.shape)      
            print("X_test:", X_test.shape)   
            #print('-----------------------------------------')
            #print("train_start:", train_start)  
            #print("train_end:", train_end)  
            #print("val end:", val_end)
            print('-----------------------------------------')
            print("total train+val+test:", X_train.shape[0]+X_val.shape[0]+X_test.shape[0])
            print('total data:', len(covariates)-max_window)                
        else:
            train_start = max_window
            train_end = n_train 

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
        
        #print('-----------------------------------------')

        #print("y_pred_test_diff:", y_pred_test_diff.shape)  
        #print("y_past_test:", y_past_test.shape)            
        #print("y_pred_test:", y_pred_test.shape)           
        #print("y_true_test:", y_true_test.shape)          


        # Calculate the error metric
        error_train = METRICS[metric](y_true_train, y_pred_train)
        error_test  = METRICS[metric](y_true_test, y_pred_test)
    #-----------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------------
    elif signature_type == 'efm': #EFM signatures
        if len(lam)==0:
            raise ValueError('efm signatures need lambda')
        _, signatures = compute_efm_signatures(n_obs=n_obs, t_grid=t_grid, simulations=covariates.reshape(1,-1), trunc = trunc_order, start_idx=max_window, lam=lam) # function imported from helper.py
        print('EFM signatures shape:', signatures.shape)

        # Train-validation-test split
        if tune_penalty:
            if n_valid is None:
                raise ValueError("n_valid must be specified when tune_penalty=True")

            train_start = 0
            train_end = n_train - max_window
            val_end = n_train + n_valid - max_window

            X_train = signatures[train_start:train_end, :]
            y_train = target[train_start+max_window:train_end+max_window] - target[train_start+max_window-delay:train_end+max_window-delay]

            X_val = signatures[train_end:val_end, :]
            y_val = target[train_end+max_window:val_end+max_window] - target[train_end-delay+max_window:val_end+max_window-delay]

            X_test = signatures[val_end:, :]
                       
            print("X_train:", X_train.shape) 
            print("X_val:", X_val.shape)      
            print("X_test:", X_test.shape)   
            #print('-----------------------------------------')
            #print("train_start:", train_start)  
            #print("train_end:", train_end)  
            #print("val end:", val_end)
            print('-----------------------------------------')
            print("total train+val+test:", X_train.shape[0]+X_val.shape[0]+X_test.shape[0])
            print('total data:', len(covariates)-max_window)
                    

        else:
            train_start = 0
            train_end = n_train - max_window
            X_train = signatures[train_start:train_end, :]
            y_train = target[train_start+max_window:train_end+max_window] - target[train_start-delay+max_window:train_end-delay+max_window]

            X_val, y_val = None, None
            X_test = signatures[train_end:, :]

        # Penalty tuning
        best_penalty = penalties[0]
        if tune_penalty:
            best_score = np.inf
            for penalty in penalties:
                if scale:
                    model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("regressor", Ridge(fit_intercept=True, alpha=penalty))
                    ])
                else:
                    model = Ridge(alpha=penalty)
                model.fit(X_train, y_train)
                y_val_pred_diff = model.predict(X_val)
                y_val_pred = y_val_pred_diff + target[train_end-delay:val_end-delay]
                y_val_true = target[train_end:val_end]

                score = METRICS[metric](y_val_true, y_val_pred)
                print(score, best_score, penalty, best_penalty)
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
        if scale:
            model_best = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(fit_intercept=True, alpha=best_penalty))
            ])
        else:
            model_best = Ridge(alpha=best_penalty)
        model_best.fit(X_train_full, y_train_full)

        # Predictions
        y_pred_train_diff = model_best.predict(X_train_full)
        y_past_train = target[train_start+max_window-delay : train_start+max_window-delay+len(y_pred_train_diff)]
        y_pred_train = y_pred_train_diff + y_past_train
        y_true_train = target[train_start+max_window:train_start+len(y_pred_train_diff)+max_window]

        y_pred_test_diff = model_best.predict(X_test)
        y_past_test = target[-len(y_pred_test_diff)-delay: -delay] # target[train_end + (n_valid if tune_penalty else 0) - delay : -delay]
        y_pred_test = y_pred_test_diff + y_past_test
        y_true_test = target[-len(y_pred_test_diff):]#target[train_end + (n_valid if tune_penalty else 0):]

        #print('-----------------------------------------')

        #print("y_pred_test_diff:", y_pred_test_diff.shape)  
        #print("y_past_test:", y_past_test.shape)            
        #print("y_pred_test:", y_pred_test.shape)           
        #print("y_true_test:", y_true_test.shape)          

        
        # Calculate the error metric
        error_train = METRICS[metric](y_true_train, y_pred_train)
        error_test  = METRICS[metric](y_true_test, y_pred_test)
    
        if scale:
            model_best = model_best["regressor"]

    return {
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_true_test": y_true_test,
        "error_train": error_train,
        "error_test": error_test,
        "best_penalty": best_penalty,
        "model": model_best
    }
