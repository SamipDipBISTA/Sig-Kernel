import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "signature"))

import numpy as np
import jax.numpy as jnp
import pandas as pd
from sklearn.metrics import mean_squared_error

from signature.path_signature import path_to_fm_signature

WARMUP_DAYS = 252


def get_efm_signature_dim(path_dim, trunc):
    return (path_dim ** (trunc + 1) - 1) // (path_dim - 1)


def build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days=WARMUP_DAYS):
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

    assert len(features) == len(targets) == len(feat_dates), \
        f"EFM alignment error: features={len(features)}, targets={len(targets)}, dates={len(feat_dates)}"

    return features, targets, feat_dates


def _fit_ridge(X_tr, y_tr, alpha):
    n, d = X_tr.shape
    A = (X_tr.T @ X_tr) / n + alpha * np.eye(d)
    b = (X_tr.T @ y_tr) / n
    return np.linalg.solve(A, b)


def _standardize_fit(X, y):
    mu_x = np.mean(X, axis=0)
    sd_x = np.std(X, axis=0) + 1e-10
    mu_y = np.mean(y)
    sd_y = np.std(y) + 1e-10
    return mu_x, sd_x, mu_y, sd_y


def _apply_standardize(X, y, mu_x, sd_x, mu_y, sd_y):
    return (X - mu_x) / sd_x, (y - mu_y) / sd_y


def _split_masks(feat_dates, train_end_ts, val_end_ts):
    train_mask = np.array([d <= train_end_ts for d in feat_dates])
    val_mask = np.array([(d > train_end_ts) & (d <= val_end_ts) for d in feat_dates])
    return train_mask, val_mask


def _compute_efm_features_from_path(path, t_grid, lam_jax, trunc, log_returns,
                                    warmup_days, feat_dates, train_mask, val_mask):
    SigS = path_to_fm_signature(
        path=jnp.array(path), trunc=trunc,
        t_grid=jnp.array(t_grid), lam=lam_jax,
    )
    features = np.array(SigS.array.T)[:-1]
    targets = log_returns[warmup_days:]

    X_train = features[train_mask]
    y_train = targets[train_mask]
    X_val = features[val_mask]
    y_val = targets[val_mask]

    if len(X_train) < 50 or len(X_val) < 10:
        return None, None, None, None

    return X_train, y_train, X_val, y_val


def _evaluate_val_mse(X_train, y_train, X_val, y_val, alpha_fixed=0.001):
    mu_x, sd_x, mu_y, sd_y = _standardize_fit(X_train, y_train)
    X_tr_n, y_tr_n = _apply_standardize(X_train, y_train, mu_x, sd_x, mu_y, sd_y)
    X_val_n, y_val_n = _apply_standardize(X_val, y_val, mu_x, sd_x, mu_y, sd_y)
    theta = _fit_ridge(X_tr_n, y_tr_n, alpha_fixed)
    y_pred = X_val_n @ theta
    return float(mean_squared_error(y_val_n, y_pred))


def optimize_params_single(log_prices, log_volumes, dates, trunc,
                           train_end, val_end,
                           lambda_grid, ridge_alphas,
                           warmup_days=WARMUP_DAYS):
    path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)
    log_returns = np.diff(log_prices)
    feat_dates = dates[warmup_days:-1]

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train_mask, val_mask = _split_masks(feat_dates, train_end_ts, val_end_ts)

    best_lam = None
    best_lam_mse = np.inf

    for lam in lambda_grid:
        lam_jax = jnp.array(lam, dtype=jnp.float64)
        result = _compute_efm_features_from_path(
            path, t_grid, lam_jax, trunc, log_returns,
            warmup_days, feat_dates, train_mask, val_mask,
        )
        X_train, y_train, X_val, y_val = result
        if X_train is None:
            continue

        mse = _evaluate_val_mse(X_train, y_train, X_val, y_val, alpha_fixed=0.001)
        if mse < best_lam_mse:
            best_lam_mse = mse
            best_lam = np.array(lam)

    if best_lam is None:
        print(f"    WARNING: no valid lambda found in grid")
        return None, None, np.inf

    print(f"    Stage 1 best lambda={np.round(best_lam, 4)}, val MSE={best_lam_mse:.4f}")

    lam_jax_best = jnp.array(best_lam, dtype=jnp.float64)
    result = _compute_efm_features_from_path(
        path, t_grid, lam_jax_best, trunc, log_returns,
        warmup_days, feat_dates, train_mask, val_mask,
    )
    X_train, y_train, X_val, y_val = result

    mu_x, sd_x, mu_y, sd_y = _standardize_fit(X_train, y_train)
    X_tr_n, y_tr_n = _apply_standardize(X_train, y_train, mu_x, sd_x, mu_y, sd_y)
    X_val_n, y_val_n = _apply_standardize(X_val, y_val, mu_x, sd_x, mu_y, sd_y)

    best_alpha = None
    best_alpha_mse = np.inf
    for alpha in ridge_alphas:
        theta = _fit_ridge(X_tr_n, y_tr_n, alpha)
        y_pred = X_val_n @ theta
        mse = float(mean_squared_error(y_val_n, y_pred))
        if mse < best_alpha_mse:
            best_alpha_mse = mse
            best_alpha = alpha

    print(f"    Stage 2 best alpha={best_alpha}, val MSE={best_alpha_mse:.4f}")

    return best_lam, best_alpha, best_alpha_mse


def optimize_params_basket(stocks_data, trunc, train_end, val_end,
                           lambda_grid, ridge_alphas,
                           warmup_days=WARMUP_DAYS):
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    best_lam = None
    best_lam_mse = np.inf

    for lam in lambda_grid:
        try:
            lam_jax = jnp.array(lam, dtype=jnp.float64)
            all_X_train, all_y_train = [], []
            all_X_val, all_y_val = [], []

            for log_prices, log_volumes, dates in stocks_data:
                path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)
                feat_dates = dates[warmup_days:-1]
                train_mask, val_mask = _split_masks(feat_dates, train_end_ts, val_end_ts)

                result = _compute_efm_features_from_path(
                    path, t_grid, lam_jax, trunc, np.diff(log_prices),
                    warmup_days, feat_dates, train_mask, val_mask,
                )
                X_tr, y_tr, X_val, y_val = result
                if X_tr is None:
                    continue
                all_X_train.append(X_tr)
                all_y_train.append(y_tr)
                all_X_val.append(X_val)
                all_y_val.append(y_val)

            if not all_X_train:
                continue

            X_train = np.vstack(all_X_train)
            y_train = np.concatenate(all_y_train)
            X_val = np.vstack(all_X_val)
            y_val = np.concatenate(all_y_val)

            mse = _evaluate_val_mse(X_train, y_train, X_val, y_val, alpha_fixed=0.001)
            if mse < best_lam_mse:
                best_lam_mse = mse
                best_lam = np.array(lam)
        except Exception as e:
            print(f"    WARNING: lambda={np.round(lam, 4)} failed: {e}")
            continue

    if best_lam is None:
        print(f"    WARNING: no valid lambda found in grid")
        return None, None, np.inf

    print(f"    Stage 1 best lambda={np.round(best_lam, 4)}, val MSE={best_lam_mse:.4f}")

    try:
        lam_jax_best = jnp.array(best_lam, dtype=jnp.float64)
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []

        for log_prices, log_volumes, dates in stocks_data:
            path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)
            feat_dates = dates[warmup_days:-1]
            train_mask, val_mask = _split_masks(feat_dates, train_end_ts, val_end_ts)

            result = _compute_efm_features_from_path(
                path, t_grid, lam_jax_best, trunc, np.diff(log_prices),
                warmup_days, feat_dates, train_mask, val_mask,
            )
            X_tr, y_tr, X_val, y_val = result
            if X_tr is None:
                continue
            all_X_train.append(X_tr)
            all_y_train.append(y_tr)
            all_X_val.append(X_val)
            all_y_val.append(y_val)

        X_train = np.vstack(all_X_train)
        y_train = np.concatenate(all_y_train)
        X_val = np.vstack(all_X_val)
        y_val = np.concatenate(all_y_val)

        mu_x, sd_x, mu_y, sd_y = _standardize_fit(X_train, y_train)
        X_tr_n, y_tr_n = _apply_standardize(X_train, y_train, mu_x, sd_x, mu_y, sd_y)
        X_val_n, y_val_n = _apply_standardize(X_val, y_val, mu_x, sd_x, mu_y, sd_y)

        best_alpha = None
        best_alpha_mse = np.inf
        for alpha in ridge_alphas:
            theta = _fit_ridge(X_tr_n, y_tr_n, alpha)
            y_pred = X_val_n @ theta
            mse = float(mean_squared_error(y_val_n, y_pred))
            if mse < best_alpha_mse:
                best_alpha_mse = mse
                best_alpha = alpha

        print(f"    Stage 2 best alpha={best_alpha}, val MSE={best_alpha_mse:.4f}")
        return best_lam, best_alpha, best_alpha_mse
    except Exception as e:
        print(f"    WARNING: Stage 2 failed: {e}")
        return best_lam, None, np.inf
