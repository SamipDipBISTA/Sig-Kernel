# Complete Pipeline Documentation: EFM Signature vs Path Signature Approaches

This document provides the complete and detailed pipeline for both the **EFM (Exponential Forgetting Memory) Signature** approach and the **Path Signature Only** approach used in this repository.

---

## Table of Contents

1. [EFM Signature Approach](#1-efm-signature-approach)
   - 1.1 [Data Preparation](#11-data-preparation)
   - 1.2 [EFM Path & Grid Construction](#12-efm-path--grid-construction)
   - 1.3 [EFM Feature Computation](#13-efm-feature-computation)
   - 1.4 [Hyperparameter Optimization](#14-hyperparameter-optimization)
   - 1.5 [Model Training](#15-model-training)
   - 1.6 [Transfer Learning with EFM](#16-transfer-learning-with-efm)
   - 1.7 [Evaluation](#17-evaluation)

2. [Path Signature Only Approach](#2-path-signature-only-approach)
   - 2.1 [Data Preparation](#21-data-preparation)
   - 2.2 [Sliding Window Path Construction](#22-sliding-window-path-construction)
   - 2.3 [Path Signature Feature Computation](#23-path-signature-feature-computation)
   - 2.4 [Hyperparameter Optimization](#24-hyperparameter-optimization)
   - 2.5 [Model Training](#25-model-training)
   - 2.6 [Transfer Learning with Path Signatures](#26-transfer-learning-with-path-signatures)
   - 2.7 [Evaluation](#27-evaluation)

---

## 1. EFM Signature Approach

The EFM signature approach uses **exponential forgetting memory (EFM)** signatures computed via `path_to_fm_signature` from the signature package. The signature at each time point incorporates the entire history of the path with exponentially decaying weights controlled by a `lambda` parameter.

### 1.1 Data Preparation

**File:** `Transfer learning/src/data_utils.py:115-141`

The data preparation pipeline loads stock price and volume data, computes log-transformed values, and returns aligned time series.

```python
def prepare_stock_data(df):
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        close_col = [c for c in df.columns if "close" in c.lower()]
        close_col = close_col[0] if close_col else df.columns[0]
    close = df[close_col].values.astype(float).flatten()
    volume = df["Volume"].values.astype(float).flatten()

    mask = np.isfinite(close) & np.isfinite(volume) & (close > 0) & (volume > 0)
    close = close[mask]
    volume = volume[mask]
    dates = df.index[mask]

    log_prices = np.log(close)
    log_volumes = np.log(volume + 1)
    log_returns = np.diff(log_prices)

    return dates, log_prices, log_volumes, log_returns
```

**Key steps:**
- Remove duplicate indices and sort by date
- Extract `Adj Close` (or `Close`) prices and `Volume`
- Filter out invalid values (NaN, zero, negative)
- Compute `log_prices`, `log_volumes`, and `log_returns`

### 1.2 EFM Path & Grid Construction

**File:** `Transfer learning/src/efm_signature_utils.py:19-25`

Before computing EFM signatures, the path must be constructed with a time dimension. The time grid is measured in years from a base date (after a warmup period).

```python
WARMUP_DAYS = 252

def build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days=WARMUP_DAYS):
    n = len(log_prices)
    base_date = dates[warmup_days]
    time_years = np.array([(d - base_date).days / 365.0 for d in dates], dtype=np.float64)
    path = np.column_stack([time_years, log_prices, log_volumes])
    return path, time_years
```

**Key steps:**
- Set base date after `warmup_days` (252 trading days ≈ 1 year)
- Convert dates to years relative to base date
- Stack `[time, log_prices, log_volumes]` into a 3D path array of shape `(n, 3)`

### 1.3 EFM Feature Computation

**File:** `Transfer learning/src/efm_signature_utils.py:27-46`

The core EFM feature computation uses `path_to_fm_signature` from the signature package. This computes the functional signature with exponential forgetting at each time point.

```python
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
```

**Key steps:**
- Build the EFM path and time grid
- Convert `lam` to JAX array for the signature computation
- Call `path_to_fm_signature(path, trunc, t_grid, lam)` to compute EFM signatures
- Transpose and drop the last feature row to align with targets
- Extract `log_returns` from `warmup_days` onward as targets
- Return aligned `(features, targets, feat_dates)`

**EFM Signature Dimension:**

**File:** `Transfer learning/src/efm_signature_utils.py:15-16`

```python
def get_efm_signature_dim(path_dim, trunc):
    return (path_dim ** (trunc + 1) - 1) // (path_dim - 1)
```

For `path_dim=3` and `trunc=3`, the dimension is `(3^4 - 1) / (3 - 1) = 40`.

### 1.4 Hyperparameter Optimization

The EFM approach requires optimizing two key hyperparameters:
- **`lam`** (lambda): Controls the exponential decay rate in EFM signatures
- **`alpha`** (ridge penalty): Controls L2 regularization in ridge regression

#### Single-Stock Optimization

**File:** `Transfer learning/src/efm_signature_utils.py:72-120`

```python
def optimize_params_single(log_prices, log_volumes, dates, trunc,
                           train_end, val_end,
                           lambda_grid, ridge_alphas,
                           warmup_days=WARMUP_DAYS):
    path, t_grid = build_efm_path_and_grid(log_prices, log_volumes, dates, warmup_days)
    log_returns = np.diff(log_prices)
    feat_dates = dates[warmup_days:-1]

    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    train_mask = np.array([d <= train_end_ts for d in feat_dates])
    val_mask = np.array([(d > train_end_ts) & (d <= val_end_ts) for d in feat_dates])

    best_lam = None
    best_alpha = None
    best_val_mse = np.inf

    for lam in lambda_grid:
        lam_jax = jnp.array(lam, dtype=jnp.float64)
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
            continue

        mu_x, sd_x, mu_y, sd_y = _standardize_fit(X_train, y_train)
        X_tr_n, y_tr_n = _apply_standardize(X_train, y_train, mu_x, sd_x, mu_y, sd_y)
        X_val_n, y_val_n = _apply_standardize(X_val, y_val, mu_x, sd_x, mu_y, sd_y)

        for alpha in ridge_alphas:
            theta = _fit_ridge(X_tr_n, y_tr_n, alpha)
            y_pred = _predict_ridge(X_val_n, theta)
            mse = float(mean_squared_error(y_val_n, y_pred))

            if mse < best_val_mse:
                best_val_mse = mse
                best_lam = np.array(lam)
                best_alpha = alpha

    return best_lam, best_alpha, best_val_mse
```

**Key steps:**
- Build EFM path and grid once
- For each `lam` in `lambda_grid`:
  - Compute EFM signatures for the entire path
  - Split into train/validation using date masks
  - Standardize features and targets
  - For each `alpha` in `ridge_alphas`:
    - Fit ridge regression
    - Compute validation MSE
    - Track best `(lam, alpha)` combination

#### Basket (Multi-Stock) Optimization

**File:** `Transfer learning/src/efm_signature_utils.py:123-181`

For transfer learning, hyperparameters are optimized jointly across multiple source stocks:

```python
def optimize_params_basket(stocks_data, trunc, train_end, val_end,
                           lambda_grid, ridge_alphas,
                           warmup_days=WARMUP_DAYS):
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    best_lam = None
    best_alpha = None
    best_val_mse = np.inf

    for lam in lambda_grid:
        try:
            lam_jax = jnp.array(lam, dtype=jnp.float64)
            all_X_train, all_y_train = [], []
            all_X_val, all_y_val = [], []

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
                val_mask = np.array([(d > train_end_ts) & (d <= val_end_ts) for d in feat_dates])

                all_X_train.append(features[train_mask])
                all_y_train.append(targets[train_mask])
                all_X_val.append(features[val_mask])
                all_y_val.append(targets[val_mask])

            X_train = np.vstack(all_X_train)
            y_train = np.concatenate(all_y_train)
            X_val = np.vstack(all_X_val)
            y_val = np.concatenate(all_y_val)

            if len(X_train) < 50 or len(X_val) < 10:
                continue

            mu_x, sd_x, mu_y, sd_y = _standardize_fit(X_train, y_train)
            X_tr_n, y_tr_n = _apply_standardize(X_train, y_train, mu_x, sd_x, mu_y, sd_y)
            X_val_n, y_val_n = _apply_standardize(X_val, y_val, mu_x, sd_x, mu_y, sd_y)

            for alpha in ridge_alphas:
                theta = _fit_ridge(X_tr_n, y_tr_n, alpha)
                y_pred = _predict_ridge(X_val_n, theta)
                mse = float(mean_squared_error(y_val_n, y_pred))

                if mse < best_val_mse:
                    best_val_mse = mse
                    best_lam = np.array(lam)
                    best_alpha = alpha
        except Exception:
            continue

    return best_lam, best_alpha, best_val_mse
```

**Key steps:**
- Aggregate train/validation data across all source stocks for each `lam`
- Standardize jointly on the pooled data
- Grid search over `ridge_alphas`
- Select best `(lam, alpha)` minimizing pooled validation MSE

### 1.5 Model Training

#### Ridge Regression

**File:** `Transfer learning/src/models.py:4-9`

```python
def ridge_solve(X, y, lam):
    n, d = X.shape
    A = (X.T @ X) / n + lam * np.eye(d)
    b = (X.T @ y) / n
    theta = np.linalg.solve(A, b)
    return theta
```

**Key steps:**
- Solve the normal equations with L2 regularization: `theta = (X^T X / n + lam * I)^{-1} (X^T y / n)`
- Returns coefficient vector `theta`

#### Prediction

**File:** `Transfer learning/src/models.py:20-21`

```python
def predict(X, theta):
    return X @ theta
```

### 1.6 Transfer Learning with EFM

**File:** `Transfer learning/run_efm_benchmark.py:80-143`

The transfer learning pipeline for EFM signatures:

```python
def run_efm_transfer(all_data, source_tickers, target_ticker, truncs=(2, 3)):
    dates_t, lp_t, lv_t, lr_t = prepare_stock_data(all_data[target_ticker])

    source_stocks_data = []
    for ticker in source_tickers:
        dates_s, lp_s, lv_s, _ = prepare_stock_data(all_data[ticker])
        source_stocks_data.append((lp_s, lv_s, dates_s))

    lambda_grid = _make_lambda_grid(dim=3)

    all_results = {}
    for trunc in truncs:
        # Optimize hyperparameters on basket of source stocks
        best_lam, best_alpha, val_mse = optimize_params_basket(
            source_stocks_data, trunc=trunc,
            train_end=OPT_TRAIN_END, val_end=OPT_VAL_END,
            lambda_grid=lambda_grid, ridge_alphas=RIDGE_ALPHAS,
        )

        # Compute EFM features for all source stocks with best lambda
        source_X_list, source_y_list = [], []
        for lp_s, lv_s, dates_s in source_stocks_data:
            feats_s, tgts_s, fd_s = compute_efm_features(lp_s, lv_s, dates_s, best_lam, trunc)
            X_tr_s, y_tr_s, _, _ = split_by_date(fd_s, feats_s, tgts_s, TRAIN_END)
            source_X_list.append(X_tr_s)
            source_y_list.append(y_tr_s)

        X_source = np.vstack(source_X_list)
        y_source = np.concatenate(source_y_list)

        # Standardize source data
        mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
        my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
        X_source_n = (X_source - mu_s) / sd_s
        y_source_n = (y_source - my_s) / sy_s

        # Fit source model
        theta_S = ridge_solve(X_source_n, y_source_n, best_alpha)

        # Compute target features and standardize with SOURCE statistics
        feats_t, tgts_t, fd_t = compute_efm_features(lp_t, lv_t, dates_t, best_lam, trunc)
        X_tr, y_tr, X_te, y_te = split_by_date(fd_t, feats_t, tgts_t, TRAIN_END)

        X_tr_n = (X_tr - mu_s) / sd_s
        y_tr_n = (y_tr - my_s) / sy_s
        X_te_n = (X_te - mu_s) / sd_s
        y_te_n = (y_te - my_s) / sy_s

        # Transfer ridge: bias theta toward theta_S
        theta_T = transfer_ridge_solve(X_tr_n, y_tr_n, theta_S, LAMBDA_TRANSFER)
        y_pred = predict(X_te_n, theta_T)
        metrics = compute_all_metrics(y_te_n, y_pred)

        # Compute transfer risk
        risk = compute_transfer_risk(X_source, y_source, X_tr, y_tr)

        all_results[trunc] = {
            "lam": best_lam, "alpha": best_alpha,
            "metrics": metrics, "transfer_risk": risk,
        }

    return all_results
```

**Key steps:**
1. Prepare source and target stock data
2. Optimize `(lambda, alpha)` on the basket of source stocks via `optimize_params_basket`
3. Compute EFM features for all sources with the optimized `lambda`
4. Pool and standardize source data
5. Fit ridge regression on source data to get `theta_S`
6. Compute target EFM features (same `lambda`)
7. **Standardize target using SOURCE statistics** (critical for transfer)
8. Fit transfer ridge regression: `theta_T = argmin ||y - X theta||^2 + lambda_transfer ||theta - theta_S||^2`
9. Predict on test set and compute metrics + transfer risk

#### Transfer Ridge Regression

**File:** `Transfer learning/src/models.py:12-17`

```python
def transfer_ridge_solve(X, y, theta_S, lam):
    n, d = X.shape
    A = (X.T @ X) / n + lam * np.eye(d)
    b = (X.T @ y) / n + lam * theta_S
    theta = np.linalg.solve(A, b)
    return theta
```

**Key steps:**
- The transfer ridge penalty pulls `theta` toward the source model `theta_S`
- Equivalent to: `theta_T = argmin ||y - X theta||^2 + lam * ||theta - theta_S||^2`

### 1.7 Evaluation

**File:** `Transfer learning/src/evaluation.py`

```python
def compute_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def compute_correlation(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def compute_all_metrics(y_true, y_pred):
    return {
        "mse": compute_mse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "corr": compute_correlation(y_true, y_pred),
    }
```

#### Transfer Risk

**File:** `Transfer learning/src/transfer_risk.py`

```python
def compute_transfer_risk(X_source, y_source, X_target, y_target, reg=1e-6):
    mu_SX, mu_SY, Sigma_SX, Sigma_SXY = _estimate_gaussian_params(X_source, y_source)
    mu_TX, mu_TY, Sigma_TX, Sigma_TXY = _estimate_gaussian_params(X_target, y_target)

    d = X_source.shape[1]
    Sigma_SX_reg = Sigma_SX + reg * np.eye(d)
    Sigma_TX_reg = Sigma_TX + reg * np.eye(d)

    w_S = np.linalg.solve(Sigma_SX_reg, Sigma_SXY)
    w_T = np.linalg.solve(Sigma_TX_reg, Sigma_TXY)

    bias = mu_TY - mu_SY - w_S @ (mu_TX - mu_SX)
    bias_sq = bias ** 2

    var_source = w_S @ Sigma_TX @ w_S
    var_target = w_T @ Sigma_TX @ w_T

    if var_source < 0:
        var_source = 0.0
    if var_target < 0:
        var_target = 0.0

    var_term = (np.sqrt(var_source) - np.sqrt(var_target)) ** 2

    return float(bias_sq + var_term)
```

**Key steps:**
- Estimate Gaussian parameters (mean and covariance) for source and target
- Compute optimal linear predictors `w_S` and `w_T` under Gaussian assumption
- Decompose risk into bias squared + variance term
- Measure how well the source model would perform on target distribution

---

## 2. Path Signature Only Approach

The Path Signature approach computes **standard truncated path signatures** over sliding windows using the `iisignature` library. Each feature vector represents the signature of a local window of the most recent `lag` observations.

### 2.1 Data Preparation

**File:** `Transfer learning/src/data_utils.py:115-141`

Same as EFM approach:

```python
def prepare_stock_data(df):
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if close_col not in df.columns:
        close_col = [c for c in df.columns if "close" in c.lower()]
        close_col = close_col[0] if close_col else df.columns[0]
    close = df[close_col].values.astype(float).flatten()
    volume = df["Volume"].values.astype(float).flatten()

    mask = np.isfinite(close) & np.isfinite(volume) & (close > 0) & (volume > 0)
    close = close[mask]
    volume = volume[mask]
    dates = df.index[mask]

    log_prices = np.log(close)
    log_volumes = np.log(volume + 1)
    log_returns = np.diff(log_prices)

    return dates, log_prices, log_volumes, log_returns
```

### 2.2 Sliding Window Path Construction

**File:** `Transfer learning/src/signature_utils.py:5-9`

For each sliding window, a 3D path is constructed with time linearly spaced in `[0, 1]`:

```python
def build_path(log_prices_window, log_volumes_window):
    n = len(log_prices_window)
    time = np.linspace(0, 1, n)
    path = np.column_stack([time, log_prices_window, log_volumes_window])
    return path
```

**Key steps:**
- Time is normalized to `[0, 1]` within each window
- Path has shape `(window_size, 3)` = `[time, log_price, log_volume]`

### 2.3 Path Signature Feature Computation

**File:** `Transfer learning/src/signature_utils.py:16-34`

```python
def compute_signature_features(log_prices, log_volumes, lag, order):
    n = len(log_prices)
    if n < lag + 1:
        raise ValueError(
            f"Series too short for lag={lag}: need at least {lag + 1} points, got {n}"
        )
    n_valid = n - lag
    sig_dim = get_signature_dim(3, order)
    features = np.empty((n_valid, sig_dim))

    for i, t in enumerate(range(lag - 1, n - 1)):
        idx_start = t - lag + 1
        idx_end = t + 1
        window_p = log_prices[idx_start:idx_end]
        window_v = log_volumes[idx_start:idx_end]
        path = build_path(window_p, window_v)
        features[i] = iisignature.sig(path, order)

    return features
```

**Key steps:**
- For each time step `t` from `lag-1` to `n-2`:
  - Extract window of `lag` observations ending at `t`
  - Build path with time augmentation
  - Compute truncated signature of order `M` using `iisignature.sig(path, order)`
- Returns feature matrix of shape `(n - lag, sig_dim)`

**Signature Dimension:**

**File:** `Transfer learning/src/signature_utils.py:12-14`

```python
def get_signature_dim(path_dim=3, order=2):
    return iisignature.siglength(path_dim, order)
```

For `path_dim=3` and `order=2`, dimension = `3 + 3^2 = 12`. For `order=3`, dimension = `3 + 9 + 27 = 39`.

#### Optimized Sliding Window Computation

**File:** `experiments/src/signature_functions.py:92-154`

An optimized version uses `iisignature.sigcombine` to incrementally update signatures:

```python
def compute_sliding_window_signatures_optimized(
    covariates: np.ndarray,
    window: int,
    trunc_order: int = 4,
    drop: bool = False
) -> np.ndarray:

    # Time augmentation:
    n_samples = covariates.shape[0]
    time = np.arange(n_samples) / window
    path = np.column_stack((time, np.array(covariates)))

    path_dim = path.shape[1]
    sig_len = iisignature.siglength(path_dim, trunc_order)
    n_windows = n_samples - window
    sig_matrix = np.zeros((n_samples, sig_len))

    # First window:
    first_path = path[:window+1]
    current_sig = iisignature.sig(first_path, trunc_order)
    sig_matrix[window, :] = current_sig
    X0 = [covariates[0, :]]

    # Remaining windows:
    for i in range(1, n_windows):
        X0.append(covariates[i, :])
        old_segment = path[i-1:i+1][::-1]
        sig_old = iisignature.sig(old_segment, trunc_order)
        current_sig = iisignature.sigcombine(sig_old, current_sig, path_dim, trunc_order)

        new_segment = path[i-1+window:i+window+1]
        sig_new = iisignature.sig(new_segment, trunc_order)
        current_sig = iisignature.sigcombine(current_sig, sig_new, path_dim, trunc_order)

        sig_matrix[window+i, :] = current_sig

    if drop == True:
        # Drop time-related components
        powers = [path_dim**k for k in range(1, trunc_order)]
        indices = np.insert(np.cumsum(powers), 0, 0)
        sig_matrix = np.delete(sig_matrix, indices, axis=1)

    return np.array(X0), sig_matrix
```

**Key steps:**
- Augment covariates with normalized time: `time = np.arange(n) / window`
- Compute signature for the first window using `iisignature.sig`
- For subsequent windows:
  - Remove the oldest segment using Chen's identity via `sigcombine` with reversed segment
  - Add the newest segment using `sigcombine`
- Optionally drop time-related signature components

### 2.4 Hyperparameter Optimization

The Path Signature approach optimizes:
- **`lag` (window size)**: Number of past observations in each sliding window
- **`order` (truncation order)**: Truncation level of the signature
- **`alpha`** (ridge penalty): L2 regularization for ridge regression

#### Window & Ridge Grid Search

**File:** `experiments/src/optimisation_functions.py:314-376`

```python
def ridge_window_grid_search(
        windows,
        alphas,
        path,
        y,
        train_start = 0,
        train_end = None,
        val_end=None,
        trunc_order=3,
        delay = 0,
        verbose=True,
        standard_scale = False,
        ):
    hyperparameters = {}
    for window in windows:
        _, X = compute_sliding_window_signatures_optimized(
                covariates = path,
                window = window,
                trunc_order=trunc_order,
            )

        if val_start is None:
            val_start=train_end
        results = ridge_regression_diff_and_reconstruct(
                    X=X,
                    y=y,
                    tune=True,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end= val_end,
                    penalties=alphas,
                    delay = delay,
                    verbose=True,
                    fit_intercept= True,
                    standard_scale = standard_scale
                )
        best_alpha = results['best_penalty']
        hyperparameters[(window, best_alpha)] = results['error_train']['mse']
    best_combo = min(hyperparameters, key=hyperparameters.get)
    best_score = hyperparameters[best_combo]

    if verbose:
        print(f"Best window: {best_combo[0]}, Best alpha: {best_combo[1]}, Min error RMSE: {np.sqrt(best_score)}")
    return best_combo[0], best_combo[1], best_score
```

**Key steps:**
- For each `window` in `windows`:
  - Compute sliding window signatures with truncation order `trunc_order`
  - Tune ridge `alpha` on validation set
  - Track best `(window, alpha)` by validation MSE

#### NW Bandwidth & Window Grid Search

**File:** `experiments/src/optimisation_functions.py:378-460`

```python
def NW_bandwidth_window_grid_search(
        windows,
        path,
        y,
        X0,
        trunc_order = 3,
        train_start=None,
        train_end = None,
        val_start=None,
        val_end=None,
        bandwidth_grid = [0.001, 0.01, 0.1],
        verbose=True,
        standard_scale = False,
        ):
    hyperparameters = {}
    for window in windows:
        for bandwidth in bandwidth_grid:

            _, X = compute_sliding_window_signatures_optimized(
                covariates = path,
                window = window,
                trunc_order=trunc_order,
            )

            model = NWSignatureRegressor(
                signature_level=trunc_order,
                bandwidth = bandwidth,
                input_type='signatures',
            )

            if val_start is None:
                val_start=train_end

            X_train = X[train_start:train_end]
            X0_train = X0[train_start:train_end]
            y_train = y[train_start:train_end]

            X_val = X[val_start:val_end]
            X0_val = X0[val_start:val_end]
            y_val = y[val_start:val_end]
            if standard_scale:
                scaler_X = StandardScaler()
                scaler_X0 = StandardScaler()

                X_train = scaler_X.fit_transform(X_train)
                X_val = scaler_X.transform(X_val)

                X0_train = scaler_X0.fit_transform(X0_train)
                X0_val = scaler_X0.transform(X0_val)

            model.fit(X_train, X0_train, y_train)

            y_pred = model.predict(X_val, X0_val)

            error = np.mean((y_val - y_pred) ** 2)

            hyperparameters[(window, bandwidth)] = error

    best_combo = min(hyperparameters, key=hyperparameters.get)
    best_score = hyperparameters[best_combo]

    if verbose:
        print(f"Best window: {best_combo[0]}, Best bandwidth: {best_combo[1]}, Min error: {best_score}")

    return best_combo[0], best_combo[1],  best_score
```

### 2.5 Model Training

#### Ridge Regression

Same as EFM approach:

**File:** `Transfer learning/src/models.py:4-9`

```python
def ridge_solve(X, y, lam):
    n, d = X.shape
    A = (X.T @ X) / n + lam * np.eye(d)
    b = (X.T @ y) / n
    theta = np.linalg.solve(A, b)
    return theta
```

#### Nadaraya-Watson Local Regression

**File:** `experiments/src/regression_functions.py:104-261`

```python
def local_reg_nw_signature(
    X,
    y,
    X0,
    delay=0,
    max_window=0,
    train_start=0,
    train_end=None,
    val_start=None,
    val_end=None,
    bandwidth = None,
    bandwidth_grid = np.linspace(0.001, 10, 50),
    tune=True,
    cv = 3,
    verbose=True,
    standard_scale=False
):
    X = np.array(X)
    y = np.array(y)

    if tune and val_end is not None:
        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(
            X, y, delay=delay, max_window=max_window,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            verbose=verbose, standard_scale=standard_scale,
        )

        X0_train = X0[train_start : train_end]
        X0_val   = X0[val_start : val_end]
        X0_test  = X0[val_end :]

        errors = {}
        for bw in bandwidth_grid:
            nw_regressor = NWSignatureRegressor(
                signature_level=3,
                bandwidth = bw,
                input_type='signatures',
                cv_folds=cv,
            )
            nw_regressor.fit(X_train, X0_train, y_train)
            y_pred_val_diff = nw_regressor.predict(X_val, X0_val)

            if delay > 0:
                y_past_val = y[train_end+max_window-delay : train_end+max_window-delay+len(y_pred_val_diff)]
                y_pred_val = y_pred_val_diff + y_past_val
            else:
                y_pred_val = y_pred_val_diff

            errors[bw] = mean_squared_error(y_val, y_pred_val)

        bandwidth = min(errors, key=errors.get)

    X_train, y_train, X_test, y_test = split_train_val_test(
        X, y, delay=delay, max_window=max_window,
        train_start=train_start, train_end=train_end,
        val_end=None, verbose=verbose, standard_scale=standard_scale,
    )

    X0_train = X0[train_start : train_end]
    X0_test  = X0[train_end :]

    nw_regressor = NWSignatureRegressor(
        signature_level=3,
        bandwidth = bandwidth,
        input_type='signatures',
        cv_folds=cv,
    )

    nw_regressor.fit(X_train, X0_train, y_train)

    y_pred_train_diff = nw_regressor.predict(X_train, X0_train)
    y_pred_test_diff  = nw_regressor.predict(X_test, X0_test)

    # Reconstruct predictions if differencing was used
    if delay > 0:
        y_past_train = y[train_start+max_window-delay : train_start+max_window-delay+len(y_pred_train_diff)]
        y_pred_train = y_pred_train_diff + y_past_train
    else:
        y_pred_train = y_pred_train_diff

    if delay > 0:
        y_past_test = y[-y_pred_test_diff.shape[0]-delay : -delay]
        y_pred_test = y_pred_test_diff + y_past_test
    else:
        y_pred_test = y_pred_test_diff

    for metric in METRICS:
        error_train[metric] = METRICS[metric](y_train, y_pred_train)
        error_test[metric]  = METRICS[metric](y_test, y_pred_test)

    return {
        "error_train": error_train,
        "error_test": error_test,
        "y_pred_test": y_pred_test,
        "y_pred_train": y_pred_train,
        "y_test": y_test,
        "y_train": y_train,
        "X_test": X_test,
        "X_train": X_train,
        "best_bandwith": bandwidth
    }
```

**Key steps:**
- Tune bandwidth on validation set using `NWSignatureRegressor`
- Fit final model with best bandwidth
- Reconstruct predictions if differencing was applied to target
- Compute train/test metrics

### 2.6 Transfer Learning with Path Signatures

**File:** `Transfer learning/run_efm_benchmark.py:146-214`

```python
def run_path_signatures(all_data, source_tickers, target_ticker):
    dates_t, lp_t, lv_t, lr_t = prepare_stock_data(all_data[target_ticker])

    source_data = {}
    for ticker in source_tickers:
        dates_s, lp_s, lv_s, lr_s = prepare_stock_data(all_data[ticker])
        source_data[ticker] = (dates_s, lp_s, lv_s, lr_s)

    results = {}
    for lag in LAGS:
        for order in ORDERS:
            source_X_list, source_y_list = [], []
            for ticker in source_tickers:
                dates_s, lp_s, lv_s, lr_s = source_data[ticker]
                feats = compute_signature_features(lp_s, lv_s, lag, order)
                fd = dates_s[lag - 1:-1]
                tgts = lr_s[lag - 1:]
                X_tr, y_tr, _, _ = split_by_date(fd, feats, tgts, TRAIN_END)
                source_X_list.append(X_tr)
                source_y_list.append(y_tr)

            X_source = np.vstack(source_X_list)
            y_source = np.concatenate(source_y_list)

            # Standardize source
            mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
            my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
            X_source_n = (X_source - mu_s) / sd_s
            y_source_n = (y_source - my_s) / sy_s

            # Compute target features
            feats_t = compute_signature_features(lp_t, lv_t, lag, order)
            fd_t = dates_t[lag - 1:-1]
            tgts_t = lr_t[lag - 1:]
            X_tr, y_tr, X_te, y_te = split_by_date(fd_t, feats_t, tgts_t, TRAIN_END)

            # Direct: standardize with target stats
            mu_t, sd_t = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
            my_t, sy_t = np.mean(y_tr), np.std(y_tr) + 1e-10
            X_tr_n = (X_tr - mu_t) / sd_t
            y_tr_n = (y_tr - my_t) / sy_t
            X_te_n = (X_te - mu_t) / sd_t
            y_te_n = (y_te - my_t) / sy_t

            theta_d = ridge_solve(X_tr_n, y_tr_n, LAMBDA_SOURCE)
            y_pred_d = predict(X_te_n, theta_d)
            m_direct = compute_all_metrics(y_te_n, y_pred_d)

            # Transfer: standardize target with SOURCE stats
            X_tr_n2 = (X_tr - mu_s) / sd_s
            y_tr_n2 = (y_tr - my_s) / sy_s
            X_te_n2 = (X_te - mu_s) / sd_s
            y_te_n2 = (y_te - my_s) / sy_s

            theta_S = ridge_solve(X_source_n, y_source_n, LAMBDA_SOURCE)
            theta_T = transfer_ridge_solve(X_tr_n2, y_tr_n2, theta_S, LAMBDA_TRANSFER)
            y_pred_t = predict(X_te_n2, theta_T)
            m_transfer = compute_all_metrics(y_te_n2, y_pred_t)

            # Transfer risk on unstandardized data
            risk = compute_transfer_risk(X_source, y_source, X_tr, y_tr)

            results[(lag, order)] = {
                "direct": m_direct, "transfer": m_transfer,
                "transfer_risk": risk,
            }

    return results
```

**Key steps:**
1. Prepare source and target stock data
2. For each `(lag, order)` combination:
   - Compute sliding window path signatures for all source stocks
   - Pool source data and standardize
   - Fit source ridge model: `theta_S`
   - Compute target path signatures with same `(lag, order)`
   - **Direct learning**: Standardize target with target stats, fit ridge
   - **Transfer learning**: Standardize target with SOURCE stats, fit transfer ridge toward `theta_S`
   - Compute metrics and transfer risk

### 2.7 Evaluation

Same evaluation metrics as EFM approach:

**File:** `Transfer learning/src/evaluation.py`

```python
def compute_all_metrics(y_true, y_pred):
    return {
        "mse": compute_mse(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "corr": compute_correlation(y_true, y_pred),
    }
```

---

## Pipeline Classes (Experiments Module)

The `experiments/src/pipeline.py` module provides high-level pipeline classes for both approaches.

### EFM Pipeline

**File:** `experiments/src/pipeline.py:53-181`

```python
class EFM_Pipeline:
    def __init__(self, alphas=None, lambda_bounds=None, bandwidth_grid=None,
                 trunc=None, val_end=None, in_sample_T=None, path=None, y=None,
                 time_grid=None, train_start=0, train_end=None, delay=0, max_window=0):
        self.alphas = alphas
        self.lambda_bounds = lambda_bounds
        self.bandwidth_grid = bandwidth_grid
        self.trunc = trunc
        self.path = path
        self.y = y
        self.time_grid = time_grid
        self.train_start = train_start
        self.train_end = train_end
        self.val_end = val_end
        self.in_sample_T = in_sample_T
        self.delay = delay
        self.max_window = max_window
        self.val_results = {}

    def EFM_ridge(self):
        best_alpha, best_lambda, best_score, _ = ridge_lambda_grid_search(
            alphas=self.alphas, path=self.path, y=self.y,
            time_grid=self.time_grid, trunc=self.trunc,
            lambda_bounds=self.lambda_bounds,
            train_start=self.train_start, train_end=self.train_end,
            val_end=self.val_end
        )
        self.val_results['efm+ridge'] = best_score

        StatSig = path_to_fm_signature(
            path=self.path, trunc=self.trunc,
            t_grid=self.time_grid, lam=best_lambda
        )
        X = StatSig.array.T

        results = ridge_regression_diff_and_reconstruct(
            X=X, y=self.y, tune=False,
            train_start=self.train_start, train_end=self.train_end,
            alpha=best_alpha, verbose=True, fit_intercept=True
        )
        return results, self.val_results

    def EFM_local_reg(self, X0):
        best_bandwidth, best_lambda, best_score, nw_results = (
            NW_bandwith_lambda_grid_search(
                path=self.path, y=self.y, X0=X0,
                time_grid=self.time_grid, trunc=self.trunc,
                lambda_bounds=self.lambda_bounds,
                train_end=self.train_end, val_end=self.val_end,
                bandwidth_grid=self.bandwidth_grid, verbose=True
            )
        )
        self.val_results['efm+local_reg'] = best_score

        StatSig = path_to_fm_signature(
            path=self.path, trunc=self.trunc,
            t_grid=self.time_grid, lam=best_lambda
        )
        X = StatSig.array.T

        results = local_reg_nw_signature(
            X=X, y=self.y, X0=X0,
            train_start=self.train_start, train_end=self.train_end,
            bandwidth=best_bandwidth
        )
        return results, self.val_results
```

### Sliding Signature Pipeline

**File:** `experiments/src/pipeline.py:187-327`

```python
class SlidingSig_Pipeline:
    def __init__(self, alphas=None, bandwidth_grid=None, windows=None,
                 trunc=None, val_end=None, in_sample_T=None, path=None, y=None,
                 time_grid=None, train_start=None, train_end=None, delay=None,
                 max_window=None, standard_scale=False):
        self.alphas = alphas
        self.bandwidth_grid = bandwidth_grid
        self.windows = windows
        self.trunc = trunc
        self.delay = delay
        self.max_window = max_window
        self.standard_scale = standard_scale
        self.path = path
        self.y = y
        self.time_grid = time_grid
        self.train_start = train_start
        self.train_end = train_end
        self.val_end = val_end
        self.in_sample_T = in_sample_T
        self.val_results = {}

    def SlidingSig_ridge(self):
        best_window, best_alpha, best_score = ridge_window_grid_search(
            windows=self.windows, alphas=self.alphas, path=self.path, y=self.y,
            max_window=self.max_window, train_start=self.train_start,
            train_end=self.train_end, val_end=self.val_end,
            trunc_order=self.trunc, delay=self.delay, verbose=True,
            standard_scale=self.standard_scale
        )
        self.val_results['sliding_sig_ridge'] = best_score

        _, SlideSig_raw = compute_sliding_window_signatures_optimized(
            covariates=self.path, window=best_window, trunc_order=self.trunc
        )
        SlideSig = SlideSig_raw[np.any(SlideSig_raw != 0, axis=1)]

        results = ridge_regression_diff_and_reconstruct(
            X=SlideSig, y=self.y, max_window=self.max_window, tune=False,
            validation_type='val', train_start=self.train_start,
            train_end=self.train_end, val_end=None, delay=self.delay,
            alpha=best_alpha, verbose=True, fit_intercept=True
        )
        return results, self.val_results

    def SlidingSig_local_reg(self, X0=None):
        best_window, best_bandwidth, best_score = NW_bandwidth_window_grid_search(
            windows=np.arange(10, 100, 10), path=self.path, y=self.y, X0=X0,
            trunc_order=self.trunc, max_window=self.max_window, delay=self.delay,
            train_start=self.train_start, train_end=self.train_end,
            val_end=self.val_end, bandwidth_grid=self.bandwidth_grid,
            verbose=True, standard_scale=self.standard_scale,
        )
        self.val_results['sliding_sig_local_reg'] = best_score

        _, SlideSig_raw = compute_sliding_window_signatures_optimized(
            covariates=self.path, window=best_window, trunc_order=self.trunc
        )
        SlideSig = SlideSig_raw[np.any(SlideSig_raw != 0, axis=1)]

        results = local_reg_nw_signature(
            X=SlideSig, y=self.y, X0=X0, delay=self.delay,
            max_window=self.max_window, trunc_order=self.trunc,
            train_start=self.train_start, train_end=self.train_end,
            bandwidth=best_bandwidth, tune=False, verbose=True,
        )
        return results, self.val_results
```

---

## Summary of Key Differences

| Aspect | EFM Signature | Path Signature |
|--------|--------------|----------------|
| **Library** | `signature.path_signature.path_to_fm_signature` | `iisignature.sig` |
| **History** | Full history with exponential decay (`lam`) | Fixed sliding window (`lag`) |
| **Time handling** | Real calendar time in years | Normalized `[0, 1]` per window |
| **Hyperparameters** | `trunc`, `lam`, `alpha` | `lag`, `order`, `alpha` |
| **Optimization** | Grid search over `lam` and `alpha` | Grid search over `window` and `alpha` |
| **Feature dim** | `(d^(trunc+1) - 1) / (d - 1)` | `iisignature.siglength(d, order)` |
| **Warmup** | 252 days required | `lag` observations required |
| **Transfer** | Basket optimization of `lam` | Fixed `(lag, order)` across stocks |

---

## File Reference Map

| Component | EFM Files | Path Signature Files |
|-----------|-----------|---------------------|
| Data prep | `Transfer learning/src/data_utils.py` | `Transfer learning/src/data_utils.py` |
| Feature computation | `Transfer learning/src/efm_signature_utils.py`<br>`experiments/src/signature_functions.py` | `Transfer learning/src/signature_utils.py`<br>`experiments/src/signature_functions.py` |
| Optimization | `Transfer learning/src/efm_signature_utils.py`<br>`experiments/src/optimisation_functions.py` | `experiments/src/optimisation_functions.py` |
| Models | `Transfer learning/src/models.py` | `Transfer learning/src/models.py` |
| Regression | `experiments/src/regression_functions.py` | `experiments/src/regression_functions.py` |
| Pipelines | `experiments/src/pipeline.py` (EFM_Pipeline) | `experiments/src/pipeline.py` (SlidingSig_Pipeline) |
| Evaluation | `Transfer learning/src/evaluation.py` | `Transfer learning/src/evaluation.py` |
| Transfer risk | `Transfer learning/src/transfer_risk.py` | `Transfer learning/src/transfer_risk.py` |
| Benchmark | `Transfer learning/run_efm_benchmark.py` | `Transfer learning/run_efm_benchmark.py` |
| Main experiment | `Transfer learning/run_experiment.py` | `Transfer learning/run_experiment.py` |
