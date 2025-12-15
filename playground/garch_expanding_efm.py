#%%

import sys
import os
# Add parent directory to path so we can import from nwsig package
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
from aeon.datasets import load_regression
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.transformations.collection.signature_based import SignatureTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import time

import warnings
warnings.filterwarnings('ignore')


from nwsig import NWSignatureRegressor, augment_with_time
from helper import *
#%%
#EFM signatures imports
import signature as efm_sig
from signature.path_signature import path_to_fm_signature
import jax.numpy as jnp


# %%
#Simulate Garch(1,1) data
np.random.seed(0)
ret = np.random.randn(1000)

am = arch_model(ret, vol="Garch", p=1, q=1)
res = am.fit(disp="off")

n_sims = 1
n_obs = 10
t_grid = jnp.linspace(0, 1, n_obs).reshape(1,-1)
simulations = np.zeros((n_sims, n_obs))

for i in range(n_sims):
    sim = am.simulate(res.params, nobs=n_obs)
    simulations[i, :] = sim['data']

print('t_grid shape:', t_grid.shape)
print('simulations shape:', simulations.shape)

#%%

lam = jnp.array([0.1])

y, efm_signatures = compute_efm_signatures(n_obs=n_obs, t_grid=t_grid, simulations=simulations, lam=lam, trunc=3, verbose=True)
print(f"EFM signatures shape: {np.array(efm_signatures).shape}")
print(f"y shape: {np.array(y).shape}")



#%%
#Prepare training and test data
X = np.array(efm_signatures)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")
print(f"y_train size: {y_train.shape}, y_test size: {y_test.shape}")

# %%

X0_train = np.ones((X_train.shape[0], 1))*X_train[0,:1]
X0_test = np.ones((X_test.shape[0], 1))*X_test[0,:1]

# Initialize regressor with cross-validation
nw_regressor = NWSignatureRegressor(
    signature_level=3,
    input_type='signatures',
    cv_folds=3,
    bandwidth_grid=np.linspace(0.001, 10, 50),
)

# Fit the model
print("Training NW Signature regressor...")
nw_regressor.fit(X_train, X0_train, y_train)
print(f"Best bandwidth: {nw_regressor.best_bandwidth_:.4f}")
# %%
# Make predictions
y_pred_nw = nw_regressor.predict(X_test, X0_test)

# Evaluate performance
mse_nw = mean_squared_error(y_test, y_pred_nw)
rmse_nw = np.sqrt(mse_nw)
mae_nw = mean_absolute_error(y_test, y_pred_nw)
r2_nw = r2_score(y_test, y_pred_nw)

print(f"\nNW Signature Test Performance:")
print(f"  MSE:  {mse_nw:.4f}")
print(f"  RMSE: {rmse_nw:.4f}")
print(f"  MAE:  {mae_nw:.4f}")
print(f"  R²:   {r2_nw:.4f}")
# %%
