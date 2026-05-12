"""
Extended Feynman-Mueller (EFM) signature features.

EXPERIMENTAL EXTENSION — not part of the original paper.  EFM extends the
path-signature approach by incorporating a learnable exponential time-weighting
(sigma) for each channel via the `lam` parameter vector (lambda_time,
lambda_price, lambda_volume).  This is an alternative to the plain sliding-window
path signature used in the paper's Section 4.

The module uses `path_to_fm_signature` from the external `signature` package and
requires JAX for computation.
"""

import sys
import os
# Add the external signature package to the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "signature"))

import numpy as np
import jax.numpy as jnp

from signature.path_signature import path_to_fm_signature

# One trading year (≈252 business days) used as warm-up before features are extracted
WARMUP_DAYS = 252


def get_efm_signature_dim(path_dim, trunc):
    """Return the dimension of an EFM-truncated signature.

    Uses the geometric-series formula for the total number of monomials
    up to a given truncation order.

    Args:
        path_dim (int): Dimension of the input path.
        trunc (int): Truncation order.

    Returns:
        int: Signature dimension.
    """
    return (path_dim ** (trunc + 1) - 1) // (path_dim - 1)


def build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days=WARMUP_DAYS):
    """Construct a 3D path with time measured in years from a base date.

    The base date is taken as `dates[warmup_days]`, so the time axis
    represents the number of years elapsed since the warm-up period.

    Args:
        log_prices (np.ndarray): 1D log-price array.
        log_volumes (np.ndarray): 1D log-volume array.
        dates (pd.DatetimeIndex): Corresponding dates.
        warmup_days (int): Number of initial days to skip as warm-up.

    Returns:
        tuple: (path, time_years)
            path (np.ndarray): 2D array (n, 3) of (time_years, log_prices, log_volumes).
            time_years (np.ndarray): 1D array of time in years since base_date.
    """
    base_date = dates[warmup_days]
    time_years = np.array([(d - base_date).days / 365.0 for d in dates], dtype=np.float64)
    path = np.column_stack([time_years, log_prices, log_volumes])
    return path, time_years


def compute_efm_features(log_prices, log_volumes, dates, lam, trunc,
                         warmup_days=WARMUP_DAYS):
    """Compute EFM signature features for a single stock.

    Builds the 3D path, applies the EFM signature transform with lambda
    weighting, and returns features aligned with future one-step log-returns.

    Features at time t (after warm-up) are used to predict the return from
    t to t+1.  The last feature row is dropped because there is no target
    for the final time step.

    Args:
        log_prices (np.ndarray): 1D log-price array.
        log_volumes (np.ndarray): 1D log-volume array.
        dates (pd.DatetimeIndex): Corresponding dates.
        lam (np.ndarray): 3-element lambda vector [lambda_time, lambda_price, lambda_volume].
        trunc (int): EFM truncation order.
        warmup_days (int): Number of warm-up days (default 252).

    Returns:
        tuple: (features, targets, feat_dates)
            features (np.ndarray): (n-warmup-1, sig_dim) feature matrix.
            targets (np.ndarray): (n-warmup-1,) log-return targets.
            feat_dates (pd.DatetimeIndex): Dates corresponding to each feature row.
    """
    path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)

    # Convert to JAX arrays for the signature computation
    lam_jax = jnp.array(lam, dtype=jnp.float64)
    SigS = path_to_fm_signature(
        path=jnp.array(path), trunc=trunc,
        t_grid=jnp.array(t_grid), lam=lam_jax,
    )
    features = np.array(SigS.array.T)  # shape (n, sig_dim)

    # Drop last row: feature at time t predicts return from t to t+1
    features = features[:-1]
    log_returns = np.diff(log_prices)
    # Targets start after warm-up; features align one-by-one
    targets = log_returns[warmup_days:]
    feat_dates = dates[warmup_days:-1]

    assert len(features) == len(targets) == len(feat_dates), \
        f"EFM alignment error: features={len(features)}, targets={len(targets)}, dates={len(feat_dates)}"

    return features, targets, feat_dates



