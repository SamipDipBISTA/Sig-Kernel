#%%

import sys
import os
# Add parent directory to path so we can import from nwsig package
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from aeon.datasets import load_regression
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.transformations.collection.signature_based import SignatureTransformer
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import time
import torch

import warnings
warnings.filterwarnings('ignore')


import helper
import importlib
importlib.reload(helper)

import run_NW_reg
import importlib
importlib.reload(run_NW_reg)

from nwsig import NWSignatureRegressor, augment_with_time
from helper import *
from run_NW_reg import *


#EFM signatures imports
import signature as efm_sig
from signature.path_signature import path_to_fm_signature
import jax.numpy as jnp
#%%
#FIX THE WINDOW FOR SLIDING WINDOW + STARTING INDEX FOR EFM

window = 5


#Simulate Garch(1,1) data
n_obs = 1000
garch_params, t_grid, simulations = simulate_garch(n_obs=n_obs, n_sims=1, seed=None)
print('garch_params:', garch_params)
print('t_grid shape:', t_grid.shape)
print('simulations shape:', simulations.shape)

#%%
'''
#BM
n_obs, n_sims, T, seed = 100, 1, 1.0, None
np.random.seed(seed)
dt = T/n_obs
t_grid = jnp.linspace(0, T, n_obs).reshape(1, -1)
simulations = np.cumsum(np.random.normal(0, np.sqrt(dt), (n_sims, n_obs)), axis=1)
simulations[:, 0] = 0  # start at 0

plt.plot(simulations[0,:])
print('t_grid shape:', t_grid.shape)
print('simulations shape:', simulations.shape)
print('simulation sample:', simulations[0, :5])
'''

#%%
#EFM signatures
lam = jnp.array([0.1])

y, efm_signatures = compute_efm_signatures(n_obs=n_obs, t_grid=t_grid, simulations=simulations, start_idx = window, lam=lam, trunc=3, verbose=True)

print(f"EFM signatures shape: {efm_signatures.shape}")
print(f"y shape: {np.array(y).shape}")

#%%
y, signatures = compute_sliding_window_signatures(simulations=simulations, t_grid=t_grid, window=window, verbose=True)#no expanding with vanilla signatures
print(f"Vanilla signatures shape: {np.array(signatures).shape}")
print(f"y shape: {np.array(y).shape}")


#%%
#Prepare training and test data
X_efm = efm_signatures
y_efm = np.array(y)

X = signatures
y = np.array(y)

y_series = pd.Series(y)
y_series.describe()

#%%
#NW
nw_regressor_efm, y_pred_nw_efm, mse_nw_efm, rmse_nw_efm, mae_nw_efm, r2_nw_efm = local_reg_nw_signature(X_efm, y_efm)

print(f"\nNW Signature Test Performance:")
print(f"  MSE:  {mse_nw_efm:.4f}")
print(f"  RMSE: {rmse_nw_efm:.4f}")
print(f"  MAE:  {mae_nw_efm:.4f}")
print(f"  R²:   {r2_nw_efm:.4f}")

#%%
#Ridge regression

ridge_model_efm, y_pred_ridge_efm, mse_ridge_efm, rmse_ridge_efm, mae_ridge_efm, r2_ridge_efm = ridge_regression(X_efm, y_efm, test_size=0.2, alpha=0.01)

print("Ridge Regression Test Performance:")
print(f"  MSE:  {mse_ridge_efm:.4f}")
print(f"  RMSE: {rmse_ridge_efm:.4f}")
print(f"  MAE:  {mae_ridge_efm:.4f}")
print(f"  R²:   {r2_ridge_efm:.4f}")
# %%
# Vanilla Signature
#NW

nw_regressor_sig, y_pred_nw_sig, mse_nw_sig, rmse_nw_sig, mae_nw_sig, r2_nw_sig = local_reg_nw_signature(X, y)

print(f"\nNW Signature Test Performance:")
print(f"  MSE:  {mse_nw_sig:.4f}")
print(f"  RMSE: {rmse_nw_sig:.4f}")
print(f"  MAE:  {mae_nw_sig:.4f}")
print(f"  R²:   {r2_nw_sig:.4f}")


#Ridge regression

ridge_model_sig, y_pred_ridge_sig, mse_ridge_sig, rmse_ridge_sig, mae_ridge_sig, r2_ridge_sig = ridge_regression(X, y, test_size=0.2, alpha=0.01)

print("Ridge Regression Test Performance:")
print(f"  MSE:  {mse_ridge_sig:.4f}")
print(f"  RMSE: {rmse_ridge_sig:.4f}")
print(f"  MAE:  {mae_ridge_sig:.4f}")
print(f"  R²:   {r2_ridge_sig:.4f}")
#%%
#Overall comparision
print("\n" + "="*80)
print(f"COMPARISON ON SIMULATED DATA Y")
print("="*80)
print(f"{'Method':<30} {'MSE':<15} {'RMSE':<15} {'MAE':<10} {'R²':<10}")
print("-"*80)
print(f"{'NW Signature (EFM)':<30} {mse_nw_efm:<15.4f} {rmse_nw_efm:<15.4f} {mae_nw_efm:<10.4f} {r2_nw_efm:<10.4f}")
print(f"{'NW Signature (Sliding Window)':<30} {mse_nw_sig:<15.4f} {rmse_nw_sig:<15.4f} {mae_nw_sig:<10.4f} {r2_nw_sig:<10.4f}")
print(f"{'Ridge Regression (EFM)':<30} {mse_ridge_efm:<15.4f} {rmse_ridge_efm:<15.4f} {mae_ridge_efm:<10.4f} {r2_ridge_efm:<10.4f}")
print(f"{'Ridge Regression (Sliding Window)':<30} {mse_ridge_sig:<15.4f} {rmse_ridge_sig:<15.4f} {mae_ridge_sig:<10.4f} {r2_ridge_sig:<10.4f}")

# %%
##################################################################################################################################
##################################################################################################################################
# Regression on delta Y

y_diff = y - simulations[0][window-1:-1]

# EFM Signature

# NW
nw_regressor_efm, y_pred_nw_efm_diff, mse_nw_efm_diff, rmse_nw_efm_diff, mae_nw_efm_diff, r2_nw_efm_diff = local_reg_nw_signature(X_efm, y_diff)

print(f"\nNW Signature Test Performance:")
print(f"  MSE:  {mse_nw_efm_diff:.4f}")
print(f"  RMSE: {rmse_nw_efm_diff:.4f}")
print(f"  MAE:  {mae_nw_efm_diff:.4f}")
print(f"  R²:   {r2_nw_efm_diff:.4f}")

#Ridge

ridge_model_efm_diff, y_pred_ridge_efm_diff, mse_ridge_efm_diff, rmse_ridge_efm_diff, mae_ridge_efm_diff, r2_ridge_efm_diff = ridge_regression(X_efm, y_diff, test_size=0.2, alpha=0.01)

print("Ridge Regression Test Performance:")
print(f"  MSE:  {mse_ridge_efm_diff:.4f}")
print(f"  RMSE: {rmse_ridge_efm_diff:.4f}")
print(f"  MAE:  {mae_ridge_efm_diff:.4f}")
print(f"  R²:   {r2_ridge_efm_diff:.4f}")


# %%
# Vanilla Signature
#NW
nw_regressor_sig_diff, y_pred_nw_sig_diff, mse_nw_sig_diff, rmse_nw_sig_diff, mae_nw_sig_diff, r2_nw_sig_diff = local_reg_nw_signature(X, y_diff)

print(f"\nNW Signature Test Performance:")
print(f"  MSE:  {mse_nw_sig_diff:.4f}")
print(f"  RMSE: {rmse_nw_sig_diff:.4f}")
print(f"  MAE:  {mae_nw_sig_diff:.4f}")
print(f"  R²:   {r2_nw_sig_diff:.4f}")


#Ridge

ridge_model_sig_diff, y_pred_ridge_sig_diff, mse_ridge_sig_diff, rmse_ridge_sig_diff, mae_ridge_sig_diff, r2_ridge_sig_diff = ridge_regression(X, y, test_size=0.2, alpha=0.01)

print("Ridge Regression Test Performance:")
print(f"  MSE:  {mse_ridge_sig_diff:.4f}")
print(f"  RMSE: {rmse_ridge_sig_diff:.4f}")
print(f"  MAE:  {mae_ridge_sig_diff:.4f}")
print(f"  R²:   {r2_ridge_sig_diff:.4f}")

# %%
print("\n" + "="*80)
print(f"COMPARISON ON delta Y")
print("="*80)
print(f"{'Method':<30} {'MSE':<15} {'RMSE':<15} {'MAE':<10} {'R²':<10}")
print("-"*80)
print(f"{'NW Signature (EFM)':<30} {mse_nw_efm_diff:<15.4f} {rmse_nw_efm_diff:<15.4f} {mae_nw_efm_diff:<10.4f} {r2_nw_efm_diff:<10.4f}")
print(f"{'NW Signature (Sliding Window)':<30} {mse_nw_sig_diff:<15.4f} {rmse_nw_sig_diff:<15.4f} {mae_nw_sig_diff:<10.4f} {r2_nw_sig_diff:<10.4f}")
print(f"{'Ridge Regression (EFM)':<30} {mse_ridge_efm_diff:<15.4f} {rmse_ridge_efm_diff:<15.4f} {mae_ridge_efm_diff:<10.4f} {r2_ridge_efm_diff:<10.4f}")
print(f"{'Ridge Regression (Sliding Window)':<30} {mse_ridge_sig_diff:<15.4f} {rmse_ridge_sig_diff:<15.4f} {mae_ridge_sig_diff:<10.4f} {r2_ridge_sig_diff:<10.4f}")

# %%
