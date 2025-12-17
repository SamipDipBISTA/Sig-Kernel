"""
baselines.py — Baseline linear regression models for synthetic and real data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error#, root_mean_squared_error

from src.utils import exponential_smoothing



def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Fit, predict, evaluate

def fit_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Fit a regression model, obtain predictions on train and test, and return error metrics.
    """
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics_train = {
        "rmse": root_mean_squared_error(y_train, y_pred_train),
        "mae": mean_absolute_error(y_train, y_pred_train),
        "mape": mean_absolute_percentage_error(y_train, y_pred_train),
    }

    metrics_test = {
        "rmse": root_mean_squared_error(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "mape": mean_absolute_percentage_error(y_test, y_pred_test),
    }

    return {
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_test": y_test,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
    }



# Baselines for synthetic data

def run_baselines_synthetic(temp, consumption_sim, n_train, smooth_param):
    """
    Run baseline models on synthetic data.

    Parameters
    ----------
    temp : np.ndarray
        Temperature series
    consumption_sim : np.ndarray
        Simulated consumption
    n_train : int
        Number of training samples
    smooth_param : float
        Exponential smoothing factor

    Returns
    -------
    dict
        Dictionary of results for each baseline
    """
    results = {}

    temp_smooth = exponential_smoothing(temp, smooth_param)
    temp_smooth_sq = temp_smooth ** 2
    temp_sq = temp ** 2

    # Baseline 1 - linear regression on temperature
    X_train = temp[:n_train].reshape(-1, 1)
    X_test = temp[n_train:].reshape(-1, 1)
    y_train = consumption_sim[:n_train]
    y_test = consumption_sim[n_train:]
    results["temp"] = fit_and_evaluate(LinearRegression(), X_train, y_train, X_test, y_test)

    # Baseline 2 - linear regression on temperature + temperature^2
    X_train = np.column_stack((temp[:n_train], temp_sq[:n_train]))
    X_test = np.column_stack((temp[n_train:], temp_sq[n_train:]))
    results["temp_sq"] = fit_and_evaluate(LinearRegression(), X_train, y_train, X_test, y_test)

    # Baseline 3 - linear regression on smoothed temperature
    X_train = temp_smooth[:n_train].reshape(-1, 1)
    X_test = temp_smooth[n_train:].reshape(-1, 1)
    results["temp_smooth"] = fit_and_evaluate(LinearRegression(), X_train, y_train, X_test, y_test)

    # Best model - smoothed temperature + squared
    X_train = np.column_stack((temp_smooth[:n_train], temp_smooth_sq[:n_train]))
    X_test = np.column_stack((temp_smooth[n_train:], temp_smooth_sq[n_train:]))
    results["best_model"] = fit_and_evaluate(LinearRegression(), X_train, y_train, X_test, y_test)

    return results



# Baselines for real data

def run_baselines_real(temp, consumption, n_train, alpha1, delay_days, window=0):
    """
    Run baseline models on real data with optional lagged consumption.

    Parameters
    ----------
    temp : np.ndarray
        Temperature series
    consumption : np.ndarray
        Observed consumption
    n_train : int
        Number of training samples
    alpha1 : float
        Smoothing parameter
    delay_days : int
        Number of days to lag consumption for baseline3
    window : int
        Optional argument to align the train-test split with the signature model

    Returns
    -------
    dict
        Dictionary of results for each baseline
    """
    results = {}
    delay = delay_days * 48  # assuming 30-min intervals
    temp_smooth1 = exponential_smoothing(temp, smooth_param=alpha1)

    df_reg = pd.DataFrame({
        "Temperature": temp,
        "Temp_sq": temp**2,
        "Temp_smooth1": temp_smooth1,
        "Temp_smooth1_sq": temp_smooth1**2,
        "Consumption": consumption
    })

    # Baseline 1 - smoothed temperature and squared
    X_train = df_reg.loc[:n_train-1, ["Temp_smooth1", "Temp_smooth1_sq"]]
    X_test = df_reg.loc[n_train:, ["Temp_smooth1", "Temp_smooth1_sq"]]
    y_train = consumption[:n_train]
    y_test = consumption[n_train:]
    results["baseline1"] = fit_and_evaluate(LinearRegression(), X_train, y_train, X_test, y_test)

    # Baseline 2 - all temperature features
    X_train = df_reg.loc[:n_train-1, ["Temperature", "Temp_sq", "Temp_smooth1", "Temp_smooth1_sq"]]
    X_test = df_reg.loc[n_train:, ["Temperature", "Temp_sq", "Temp_smooth1", "Temp_smooth1_sq"]]
    results["baseline2"] = fit_and_evaluate(LinearRegression(), X_train, y_train, X_test, y_test)

    # Baseline 3 - add lagged consumption
    df_reg["Consumption_lag"] = df_reg["Consumption"].shift(delay)
    df_reg_past = df_reg.dropna().reset_index(drop=True)
    X = df_reg_past[["Temperature", "Temp_sq", "Temp_smooth1", "Temp_smooth1_sq", "Consumption_lag"]]
    y = df_reg_past["Consumption"]

    X_train = X.iloc[window:n_train+window]
    X_test = X.iloc[n_train+window:]
    y_train = y.iloc[window:n_train+window]
    y_test = y.iloc[n_train+window:]
    results["baseline3"] = fit_and_evaluate(LinearRegression(), X_train, y_train, X_test, y_test)

    return results
