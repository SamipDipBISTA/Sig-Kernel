import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "signature"))

import numpy as np
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from signature.path_signature import path_to_fm_signature

WARMUP_DAYS = 252


def get_efm_signature_dim(path_dim, trunc):
    return (path_dim ** (trunc + 1) - 1) // (path_dim - 1)


def build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days=WARMUP_DAYS):
    n = len(log_prices)
    base_date = dates[warmup_days]
    time_years = np.array([(d - base_date).days / 365.0 for d in dates], dtype=np.float64)
    path = np.column_stack([time_years, log_prices, log_volumes])
    return path, time_years


def compute_efm_features(log_prices, log_volumes, dates, lam, trunc,
                         warmup_days=WARMUP_DAYS):
    path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)

    lam_jax = jnp.array(lam, dtype=jnp.float64)
    SigS = path_to_fm_signature(
        path=jnp.array(path), trunc=trunc,
        t_grid=jnp.array(t_grid), lam=lam_jax,
    )
    features = np.array(SigS.array.T)

    features = features[:-1]
    log_returns = np.diff(log_prices)
    targets = log_returns[warmup_days:]
    feat_dates = dates[warmup_days:-1]

    return features, targets, feat_dates


def _tscv_mse(X, y, ridge_alpha, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mses = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        mu = np.mean(X_tr, axis=0)
        sd = np.std(X_tr, axis=0) + 1e-10
        X_tr_n = (X_tr - mu) / sd
        X_val_n = (X_val - mu) / sd
        y_mu, y_sd = np.mean(y_tr), np.std(y_tr) + 1e-10
        y_tr_n = (y_tr - y_mu) / y_sd
        y_val_n = (y_val - y_mu) / y_sd

        n, d = X_tr_n.shape
        A = (X_tr_n.T @ X_tr_n) / n + ridge_alpha * np.eye(d)
        b = (X_tr_n.T @ y_tr_n) / n
        theta = np.linalg.solve(A, b)
        y_pred = X_val_n @ theta
        mses.append(mean_squared_error(y_val_n, y_pred))

    return float(np.mean(mses))


def _loss_single(lam_flat, path, t_grid, log_returns, warmup_days, trunc,
                 ridge_alpha, train_mask, n_splits=5):
    try:
        lam_jax = jnp.array(lam_flat, dtype=jnp.float64)
        SigS = path_to_fm_signature(
            path=jnp.array(path), trunc=trunc,
            t_grid=jnp.array(t_grid), lam=lam_jax,
        )
        features = np.array(SigS.array.T)[:-1]
        targets = log_returns[warmup_days:]

        X = features[train_mask]
        y = targets[train_mask]

        if len(X) < 50:
            return 1e10

        return _tscv_mse(X, y, ridge_alpha, n_splits)
    except Exception:
        return 1e10


def _loss_basket(lam_flat, stocks_data, trunc, ridge_alpha, train_end_ts,
                 warmup_days, n_splits=5):
    try:
        lam_jax = jnp.array(lam_flat, dtype=jnp.float64)
        all_X, all_y = [], []
        for log_prices, log_volumes, dates in stocks_data:
            path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)
            SigS = path_to_fm_signature(
                path=jnp.array(path), trunc=trunc,
                t_grid=jnp.array(t_grid), lam=lam_jax,
            )
            features = np.array(SigS.array.T)[:-1]
            targets = np.diff(log_prices)[warmup_days:]
            feat_dates = dates[warmup_days:-1]
            train_mask = np.array([d <= train_end_ts for d in feat_dates])
            all_X.append(features[train_mask])
            all_y.append(targets[train_mask])

        X = np.vstack(all_X)
        y = np.concatenate(all_y)

        if len(X) < 50:
            return 1e10

        return _tscv_mse(X, y, ridge_alpha, n_splits)
    except Exception:
        return 1e10


def optimize_lambda(log_prices, log_volumes, dates, trunc, ridge_alpha=1.0,
                    warmup_days=WARMUP_DAYS, train_end="2021-09-01",
                    lambda_bounds=(0.001, 50)):
    path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)
    log_returns = np.diff(log_prices)
    feat_dates = dates[warmup_days:-1]
    train_end_ts = pd.Timestamp(train_end)
    train_mask = np.array([d <= train_end_ts for d in feat_dates])

    dim = 3
    scalar_lambdas = np.logspace(-2, 2, 20)
    best_lam, best_loss = np.ones(dim) * 1.0, 1e10
    for lam_scalar in scalar_lambdas:
        lam_vec = np.ones(dim) * lam_scalar
        loss = _loss_single(lam_vec, path, t_grid, log_returns, warmup_days,
                            trunc, ridge_alpha, train_mask)
        if loss < best_loss:
            best_loss = loss
            best_lam = lam_vec.copy()

    x0 = np.clip(best_lam, lambda_bounds[0], lambda_bounds[1])
    bounds = [lambda_bounds] * dim
    result = minimize(
        lambda x: _loss_single(x, path, t_grid, log_returns, warmup_days,
                               trunc, ridge_alpha, train_mask),
        x0=x0, bounds=bounds, method="Powell",
        options={"maxiter": 100, "ftol": 1e-6},
    )

    if result.fun < best_loss:
        return result.x, result.fun
    return best_lam, best_loss


def optimize_lambda_basket(stocks_data, trunc, ridge_alpha=1.0,
                           warmup_days=WARMUP_DAYS, train_end="2021-09-01",
                           lambda_bounds=(0.001, 50)):
    train_end_ts = pd.Timestamp(train_end)
    dim = 3

    scalar_lambdas = np.logspace(-2, 2, 20)
    best_lam, best_loss = np.ones(dim) * 1.0, 1e10
    for lam_scalar in scalar_lambdas:
        lam_vec = np.ones(dim) * lam_scalar
        loss = _loss_basket(lam_vec, stocks_data, trunc, ridge_alpha,
                            train_end_ts, warmup_days)
        if loss < best_loss:
            best_loss = loss
            best_lam = lam_vec.copy()

    x0 = np.clip(best_lam, lambda_bounds[0], lambda_bounds[1])
    bounds = [lambda_bounds] * dim
    result = minimize(
        lambda x: _loss_basket(x, stocks_data, trunc, ridge_alpha,
                               train_end_ts, warmup_days),
        x0=x0, bounds=bounds, method="Powell",
        options={"maxiter": 100, "ftol": 1e-6},
    )

    if result.fun < best_loss:
        return result.x, result.fun
    return best_lam, best_loss
