#%%
import sys
import os
import time
import warnings

# Add parent directory to path so we can import from nwsig package
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# -------------------------------------------------------
# Imports for updating modifications
# -------------------------------------------------------
import importlib

import experiments.src.signature_functions as signature_functions
importlib.reload(signature_functions)

import experiments.src.optimisation_functions as optimisation_functions
importlib.reload(optimisation_functions)

import experiments.src.regression_functions as regression_functions
importlib.reload(regression_functions)
 
# -------------------------------------------------------
# Project imports
# -------------------------------------------------------
from nwsig import NWSignatureRegressor, augment_with_time
from experiments.src.signature_functions import *
from experiments.src.regression_functions import *
from experiments.src.optimisation_functions import *
from experiments.src.helper import *

# EFM signatures imports
import jax.numpy as jnp
from signature.path_signature import path_to_fm_signature
import signature.tensor_algebra as ta

# Load and plot data
load_from = pd.to_datetime('1995-01-01')  # Need at least 4 years prior to the first training day
train_start_date = pd.to_datetime('2000-01-01')
test_start_date = pd.to_datetime('2019-01-01')
test_end_date = pd.to_datetime('2022-05-15')

spx_data = pd.read_csv('spx_vix_data/spx_data.csv', index_col=0)
vix_data = pd.read_csv('spx_vix_data/vix_data.csv', index_col=0)

spx_data.index = pd.to_datetime(spx_data.index)
vix_data.index = pd.to_datetime(vix_data.index)

# Define timeseries
index = spx_data.index
spx = spx_data['Close']
vix = vix_data['Close'] / 100

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
time_grid = np.array((index - train_start_date).days / 365)
dates_grid = pd.to_datetime(pd.Series(index))
ax[0].plot(dates_grid, spx, label="Price")
ax[0].legend()
# ax[1].plot(dates_grid, df_mkt.rv, label="RV")
ax[1].plot(dates_grid, vix, label="VIX")
ax[1].legend()

# Define Path
path = np.vstack([
    time_grid,
    np.log(spx),
    np.concatenate([[0], np.cumsum(np.diff(np.log(spx))**2)]),
]).T
dim = path.shape[1]
trunc = 3

# Split variables
mask = (index < test_start_date)
val_end = np.where(mask)[0][-1]

#%%
#===================================================================
# EFM + ridge
#===================================================================
# Optimize alpha for ridge and lambda for EFM
best_alpha, best_lambda, best_score, results = ridge_lambda_grid_search(
    alphas=np.logspace(-3, 3, 1),
    path=path[(index < test_start_date)],
    y=vix[(index >= train_start_date) & (index < test_start_date)],
    time_grid=time_grid[index < test_start_date],
    trunc=trunc,
    lambda_bounds=(0.001, 50),
    train_start=0,
    train_end=val_end,
    val_start=0,
    val_end=val_end
)

# Fit and predict EFM+ridge
in_sample_T = 18
time_grid_stat = time_grid[index >= train_start_date]
mask_time_grid = time_grid_stat <= in_sample_T

# model = Ridge(fit_intercept=True, alpha=best_alpha)
# best_alpha=0.001
# best_lambda = np.array([17.19153111 , 0.09110839 , 8.77044406])

StatSig = path_to_fm_signature(path=path, trunc=trunc, t_grid=time_grid, lam=best_lambda)
X = StatSig.array.T
y = vix[index >= train_start_date]

print('path shape:', path.shape)
print('X shape:', X.shape)
print('y shape:', y.shape)
#%%
results = ridge_regression_diff_and_reconstruct(
    X=X,
    y=y,
    train_start=np.where(mask_time_grid)[0][0],
    train_end=np.where(mask_time_grid)[0][-1],
    alpha=best_alpha,
    verbose=True,
    fit_intercept=True
)

y_pred_train = results['y_pred_train']
y_pred_test = results['y_pred_test']
y_pred = np.concatenate((y_pred_train, y_pred_test))
print('EFM + simple ridge')
plot_predictions_mask(dates_grid, index, y, y_pred, train_start_date, test_start_date)

#%%
#===================================================================
# Sliding sig + ridge
#===================================================================
best_window, best_alpha, best_score = ridge_window_grid_search(
    windows=np.arange(40, 70, 10),
    alphas=np.logspace(-3, 3, 1),
    path=path[index >= train_start_date,1:],
    y=vix[index >= train_start_date],
    train_start=0,
    train_end=int(len(path)*0.5),
    val_end=int(len(path)*0.7),
    trunc_order=3,
    delay=0,
    verbose=True,
)

X0, SlideSig_ = compute_sliding_window_signatures_optimized(
    covariates=path[np.argmax(time_grid >= 0)-best_window:,1:],
    window=best_window,
    trunc_order=3,
)
SlideSig = SlideSig_[np.any(SlideSig_ != 0, axis=1)]

print('X0 shape:', X0.shape)
print('Sliding sig shape:', SlideSig.shape)

X = SlideSig
y = vix[index >= train_start_date]

print('X shape:', X.shape)
print('y shape:', y.shape)
print(np.where(mask_time_grid)[0][0], np.where(mask_time_grid)[0][-1])
print('val end:', val_end)

results = ridge_regression_diff_and_reconstruct(
    X=X,
    y=y,
    train_start=np.where(mask_time_grid)[0][0],
    train_end=np.where(mask_time_grid)[0][-1],
    delay=0,
    alpha=best_alpha,
    verbose=True,
    fit_intercept=True
)

y_pred_train = results['y_pred_train']
y_pred_test = results['y_pred_test']
y_pred = np.concatenate((y_pred_train, y_pred_test))
plot_predictions_mask(dates_grid, index, y, y_pred, train_start_date, test_start_date)

#%%
#===================================================================
# EFM + Local Regression
#===================================================================
#Initial condition: 
X0=path[(index >= train_start_date) & (index < test_start_date)][:,1:]
print('X0 shape:', X0.shape)
print('path', path[(index >= train_start_date) & (index < test_start_date)].shape)
print('vix', vix[(index >= train_start_date) & (index < test_start_date)].shape)

# NW bandwidth + lambda grid search
bandwidth_grid = [0.001, 0.01, 0.1]
best_bandwidth, best_lambda, best_score, nw_results = NW_bandwith_lambda_grid_search(
    path=path[(index < test_start_date)],
    y=vix[(index >= train_start_date) & (index < test_start_date)],
    X0=X0,
    time_grid=time_grid[index < test_start_date],
    trunc=trunc,
    lambda_bounds=(0.1, 50),
    train_start = 0,
    train_end=int(len(path)*0.5),
    val_start=int(len(path)*0.5),
    val_end=val_end,
    bandwidth_grid=bandwidth_grid,
    verbose=True,
)

print("\nNW Bandwidth + Lambda Grid Search Results:")
print(nw_results)
print(path.shape)
#%%

# Fit and predict with local regression
model = NWSignatureRegressor(
    signature_level=3,
    bandwidth=best_bandwidth,
    input_type='signatures',
)

StatSig = ta.path_to_fm_signature(path=path, trunc=trunc, t_grid=time_grid, lam=best_lambda)
X = StatSig.array.T
y = vix[index >= train_start_date]
time_grid_stat = time_grid[index >= train_start_date]
X0 = path[:,1:]
X0 = X0[time_grid >= 0]

X_train, y_train = X[mask_time_grid], y[mask_time_grid]
X0_train = X0[mask_time_grid]
X_test, y_test = X[~mask_time_grid], y[~mask_time_grid]
X0_test = X0[~mask_time_grid]

model.fit(X_train, X0_train, y_train)
y_pred = model.predict(X, X0)

plot_predictions_mask(dates_grid, index, y, y_pred, train_start_date, test_start_date)

results = local_reg_nw_signature(
    X, y, X0,
    train_start=np.where(mask_time_grid)[0][0],
    train_end=np.where(mask_time_grid)[0][-1],
    bandwidth=0.01,
    bandwidth_grid=None,
)
print(results)

#%%
#===================================================================
# Sliding sig + local regression
#===================================================================
best_window, best_bandwidth, best_score = NW_bandwidth_window_grid_search(
    windows=np.arange(40,60,1),
    path=path[index >= train_start_date,1:],
    y=vix[index >= train_start_date],
    X0=path[index >= train_start_date,1:][:,1].reshape(-1,1),
    trunc_order=3,
    train_start=0,
    train_end=int(len(path)*0.5),
    val_end=int(len(path)*0.7),
    bandwidth_grid= np.linspace(0.001, 10, 5),
    verbose=True,
)

X0, SlideSig_ = compute_sliding_window_signatures_optimized(
    covariates=path[np.argmax(time_grid >= 0)-best_window:,1:],
    window=best_window,
    trunc_order=3,
)
SlideSig = SlideSig_[np.any(SlideSig_ != 0, axis=1)]
print(X0.shape)
#%%
results = local_reg_nw_signature(
    X=X,
    y=y,
    X0=X0,
    tune=False,
    train_start=np.where(mask_time_grid)[0][0],
    train_end=np.where(mask_time_grid)[0][-1],
    bandwidth=best_bandwidth,
    delay=0,
    verbose=True,
)

y_pred_train = results['y_pred_train']
y_pred_test = results['y_pred_test']
y_pred = np.concatenate((y_pred_train, y_pred_test))
plot_predictions_mask(dates_grid, index, y, y_pred, train_start_date, test_start_date)

