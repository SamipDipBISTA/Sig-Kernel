"""
exp_smoothing.py — Function to simulate synthetic consumption (demand)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from src.utils import exponential_smoothing 


def consumption_simulate(df, sd, smooth_param, fixed_seed=None):
    """
    Simulate electricity consumption using a regression on smoothed temperature
    and autoregressive noise.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing columns "DateTime", "Temperature" and "Consumption".
    sd : float
        Standard deviation of the Gaussian noise term.
    smooth_param : float
        Smoothing parameter for exponential smoothing (in (0,1]).
    fixed_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    consumption_sim : np.ndarray
        Simulated consumption time series with noise.
    """
    
    # Smooth temperature
    temp = df["Temperature"].astype(float)
    temp_smooth = exponential_smoothing(temp, smooth_param)

    # Regression features
    df_reg = df.copy()
    df_reg.insert(len(df_reg.columns), "Temp_smooth", temp_smooth)
    df_reg.insert(len(df_reg.columns), "Temp_smooth_sq", df_reg["Temp_smooth"] ** 2)
    X_sim = df_reg.iloc[:, -2:]
    y_sim = df_reg["Consumption"]

    # Fit regression model
    reg_smooth_temp = LinearRegression().fit(X_sim, y_sim)
    consumption_fit = reg_smooth_temp.predict(X_sim)

    # Fix random seed
    if fixed_seed is not None:
        np.random.seed(fixed_seed)

    rand_noise = np.random.normal(0, sd, len(consumption_fit))

    # Add noise
    consumption_sim = consumption_fit + rand_noise

    return consumption_sim