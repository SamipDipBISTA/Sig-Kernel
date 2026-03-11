

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
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

import time

import numpy as np
import jax.numpy as jnp
from typing import List, Tuple, Optional
from signature.path_signature import path_to_fm_signature, path_to_signature


def simulate_garch(n_obs=100, n_sims=1, p=1, q=1, type = "Garch", seed=0):
    np.random.seed(seed)
    ret = np.random.randn(1000)
    am = arch_model(ret, vol=type, p=1, q=1)
    res = am.fit(disp="off")
    t_grid = jnp.linspace(0, 1, n_obs).reshape(1, -1)
    simulations = np.zeros((n_sims, n_obs))
    for i in range(n_sims):
        sim = am.simulate(res.params, nobs=n_obs)
        simulations[i, :] = sim['data']
    return res.params, t_grid, simulations




#compute signatures

def compute_efm_signatures(
    n_obs: int,
    t_grid: np.ndarray,
    simulations: np.ndarray,
    signature_type: str ='EFM',
    start_idx: int = 1,
    lam: Optional[jnp.ndarray] = None,
    trunc: int = 3,
    verbose: bool = True
):
    """
    Compute EFM signatures for a single path across time steps and
    """
    # Default lam if not provided
    if lam is None:
        lam = jnp.array([0.1])

    # Basic shape checks
    if t_grid.ndim != 2 or simulations.ndim != 2:
        raise ValueError("t_grid and simulations must be 2D arrays (N_paths, N_times).")

    n_times = t_grid.shape[1]

    y = []

    t_grid_shifted = t_grid - t_grid[0][start_idx]        
    sim_path = simulations 
    print('sim path:', sim_path.shape)
    
    path = np.vstack([t_grid_shifted, sim_path]).T 
    

    start = time.perf_counter()
    for t in range(start_idx, n_obs):
         y.append(simulations[0, t])

    print('t_grid:', t_grid_shifted.shape)
    print('Path shape:', path.shape)
    print('y size:', len(y))

    SigS = path_to_fm_signature(path=path, trunc=trunc, t_grid=t_grid_shifted.flatten(), lam=lam)
    end = time.perf_counter()
    elapsed = end - start
    print(f"Time taken to compute EFM signatures: {elapsed:.2f} seconds")
    
    return y, np.array(SigS.array.T)


def compute_sliding_window_signatures(
    simulations: np.ndarray,
    t_grid: np.ndarray,
    trunc: int = 3,
    window: int = 5,
    verbose: bool = True,
):
    if t_grid.ndim != 2 or simulations.ndim != 2:
        raise ValueError("t_grid and simulations must be 2D arrays")

    n_obs = simulations.shape[1]
    y = []
    signatures = []
    t_grid_t = t_grid[0, 0:window]
    sim_path = simulations[0, 0:window]
    path = np.vstack([t_grid_t, sim_path]).T

    Sig = path_to_signature(path=path, trunc=trunc)
    for t in range(window, n_obs):
        if verbose:
            print("Processing time step:", t)

        t_grid_t = t_grid[0, t - window:t]
        sim_path = simulations[0, t - window:t]
        path = np.vstack([t_grid_t, sim_path]).T

        Sig = path_to_signature(path=path, trunc=trunc)
        signatures.append(Sig[:, -1])
        y.append(simulations[0, t])

    return y, signatures



# Ridge


def ridge_regression(X, y, test_size=0.2, alpha=1.0):
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_pred, mse, rmse, mae, r2



def optimize_params(path, y, trunc, time_grid, model):
    dim = path.shape[1]

    def loss(x):
        StatSig = ta.path_to_fm_signature(path=path, trunc=trunc, t_grid=time_grid, lam=x)
        model.fit(StatSig.array.T, y)
        return mean_squared_error(y, model.predict(StatSig.array.T))

    optimizer = minimize(loss, x0=np.ones(dim) * 4, bounds=([0.00001, 50],) * dim,  method='Powell')
    return optimizer.x