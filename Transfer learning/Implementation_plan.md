# Implementation Plan: Reproducing Section 4 — Transfer Learning for Stock Return Prediction

**Paper:** "Risk of Transfer Learning and its Applications in Finance" (arXiv:2311.03283)
**Goal:** 100% faithful reproduction of Section 4 results (Table 1 + Table 2)

---

## 1. Experimental Setup Summary (from Paper)

### Data
- **Source:** Historical daily stock price (close) and trading volume from the **Information Technology sector** of the US equity market
- **Period:** February 2010 to September 2022
- **Stock selection:** For each experiment, randomly sample **11 stocks** from IT sector; first 10 = source, last 1 = target
- **Splits:**
  - Source training: 10 source stocks, Feb 2010 – Sep 2021
  - Target training: 1 target stock, Feb 2010 – Sep 2021
  - Target testing: 1 target stock, Sep 2021 – Sep 2022
- **Averaging:** Results averaged over **200 random selections** of 11 stocks

### Feature Construction (Signature Transform)
1. Log-transform price and volume: `log(s_tau)`, `log(v_tau)`
2. Target: `y_t = log(s_{t+1}) - log(s_t)` (next-day log return)
3. For each time step t, build 3D path: `z_tau = (tau, log(s_tau), log(v_tau))` for `t-L+1 <= tau <= t`
4. Compute truncated signature of order M: `x_t = S_M(z)`
5. **Standardize** each feature and target variable (z-score: subtract mean, divide by std)
6. Hyperparameters: Lag L ∈ {2, 3, 5, 10}, Order M ∈ {2, 3, 4}

### Models
- **Direct Learning Ridge:** `theta_hat = argmin (1/T) Σ(x_t·θ - y_t)² + λ‖θ‖²`
  - Trained on target training data
- **Transfer Learning Ridge:**
  - Step 1 (pre-train): `θ_S = argmin (1/T_S) Σ(x_{S,t}·θ - y_{S,t})² + λ_S‖θ‖²`
  - Step 2 (fine-tune): `θ_T = argmin (1/T_T) Σ(x_{T,t}·θ - y_{T,t})² + λ_T‖θ - θ_S‖²`
- **Regularization:** λ_S = 1.0, λ_T = 5.0 (for both Direct and Transfer)

### Metrics
- **MSE** (mean squared error on test set)
- **R²** (coefficient of determination)
- **Correlation** (Pearson between predicted and actual returns)

### Transfer Risk (Table 2)
- Computed as Wasserstein-2 output transport risk: `C(S,T) = E^O_W(θ̂_S)` with p=2
- Closed-form for Gaussian model:
  ```
  C_W(S,T) = [μ_{T,Y} - μ_{S,Y} - Σ_{S,YX}Σ_{S,X}^{-1}(μ_{T,X} - μ_{S,X})]²
           + (√(Σ_{S,YX}Σ_{S,X}^{-1}Σ_{T,X}Σ_{S,X}^{-1}Σ_{S,XY}) - √(Σ_{T,YX}Σ_{T,X}^{-1}Σ_{T,XY}))²
  ```
- Table 2: Correlation matrix of {MSE, -R², -Corr, Transfer Risk} across 200 experiments (L=5, M=2)

---

## 2. Folder Structure

```
Sig-Kernel/Transfer learning/
├── plan.md                    # This file
├── requirements.txt           # Dependencies (path sig only, EFM-ready)
├── src/
│   ├── __init__.py
│   ├── config.py              # Constants, hyperparameters, paths
│   ├── data_utils.py          # Stock data download (yfinance), sampling, preprocessing
│   ├── signature_utils.py     # Path construction + iisignature computation
│   ├── models.py              # Direct Ridge + Transfer Ridge (closed-form)
│   ├── transfer_risk.py       # Wasserstein-2 transfer risk computation
│   ├── evaluation.py          # MSE, R², Correlation metrics
│   └── experiment.py          # Main experiment runner (200 trials)
├── run_experiment.py          # Entry point script
├── notebooks/
│   └── section4_reproduction.ipynb  # Interactive reproduction notebook
└── data/                      # Cached downloaded stock data (gitignored)
```

---

## 3. Module-by-Module Specification

### 3.1 `requirements.txt`
```
numpy>=1.21
pandas>=1.3
yfinance>=0.2.18
iisignature>=0.24
scikit-learn>=1.0
scipy>=1.7
matplotlib>=3.4
seaborn>=0.11
tqdm>=4.62
tabulate>=0.9
# EFM preparation (not used yet, but ready for future benchmarking):
# signature>=0.4.0
# jax>=0.3.0
# jaxlib>=0.3.0
```

### 3.2 `src/config.py`
- IT sector ticker universe (all current & historical IT sector constituents of S&P 500 / NASDAQ-100, ~70-80 tickers)
- Date ranges: `TRAIN_START = "2010-02-01"`, `TRAIN_END = "2021-09-30"`, `TEST_END = "2022-09-30"`
- Lag values: `LAGS = [2, 3, 5, 10]`
- Order values: `ORDERS = [2, 3, 4]`
- Regularization: `LAMBDA_S = 1.0`, `LAMBDA_T = 5.0`, `LAMBDA_DIRECT = 1.0`
- `N_EXPERIMENTS = 200`, `N_SOURCE = 10`, `RANDOM_SEED = 42`

### 3.3 `src/data_utils.py`

**Functions:**
- `download_it_sector_data(tickers, start, end)` → `Dict[str, pd.DataFrame]`
  - Uses `yfinance.download(ticker, start, end)` for each ticker
  - Returns dict mapping ticker → DataFrame with columns: Close, Volume
  - Cache to `data/` folder as CSV to avoid re-downloading
- `get_it_tickers()` → `List[str]`
  - Returns comprehensive list of IT sector tickers (hardcoded from S&P 500 IT sector constituents circa 2022: AAPL, ACN, ADP, ADSK, ADBE, AMD, AVGO, CDW, CDNS, CTSH, CRM, CSCO, DXC, FIS, FISV, FTNT, GOOG, GOOGL, HPE, HPQ, IBM, INTU, INTC, IT, JKHY, KEYS, KLAC, LRCX, MA, MCHP, MSFT, MSI, NTAP, NVDA, NXPI, ORCL, PAYX, PYPL, QCOM, SNPS, SWKS, T, TMUS, TRMB, TSM, TXN, V, VRSN, WDC, WIT, XLNX, etc.)
- `sample_stocks(tickers, n_source=10, random_state=None)` → `(source_tickers, target_ticker)`
  - Randomly selects 11 tickers, returns first 10 as source, last 1 as target
- `prepare_stock_data(df)` → `(dates, log_prices, log_volumes, log_returns)`
  - Computes `log(close)`, `log(volume)`, `log(close[t+1]) - log(close[t])`
  - Aligns dates and handles NaN/inf

### 3.4 `src/signature_utils.py`

**Functions:**
- `build_path(log_prices, log_volumes, t, lag)` → `np.ndarray` of shape `(lag+1, 3)`
  - Constructs `z_tau = (tau, log(s_tau), log(v_tau))` for `t-lag+1 <= tau <= t`
  - Time component: `tau` normalized to `[0, 1]` range (i.e., `np.arange(lag+1) / lag`)
- `compute_signature_features(log_prices, log_volumes, lag, order)` → `np.ndarray` of shape `(n_samples, sig_dim)`
  - For each valid time step t (from t=lag-1 onward), build path and compute truncated signature
  - Uses `iisignature.sig(path, order)` for truncated signature computation
  - Returns matrix of signature features
- `get_signature_dimension(path_dim=3, order=2)` → `int`
  - Computes expected signature dimension: `sum_{k=0}^{M} path_dim^k`

**Key implementation detail:** The time augmentation uses `tau` normalized to `[0,1]` within each lag window, matching the paper's 3D path definition. The `iisignature` library is used (already in the repo's `requirements.txt`), consistent with how the existing codebase computes sliding window signatures.

### 3.5 `src/models.py`

**Functions:**

- `direct_ridge(X_train, y_train, lam)` → `theta`
  - Closed-form Ridge: `θ = (X^T X + λ·T·I)^{-1} X^T y`
  - Where T = number of training samples (matching paper's `(1/T)` scaling)
  - Implementation: `theta = np.linalg.solve(X.T @ X + lam * T * I, X.T @ y)`

- `pretrain_source(X_source, y_source, lam_S)` → `theta_S`
  - Same closed-form as direct_ridge but on source data
  - Concatenates signature features from all 10 source stocks

- `transfer_ridge(X_target, y_target, theta_S, lam_T)` → `theta_T`
  - Closed-form: `θ_T = (X^T X + λ_T·T·I)^{-1} (X^T y + λ_T·T·θ_S)`
  - This is the key transfer equation: regularize toward θ_S instead of zero

- `predict(X_test, theta)` → `y_pred`
  - Simple: `X_test @ theta`

**Important:** The paper uses the formulation `(1/T) Σ(...)² + λ‖θ‖²`, so the closed-form becomes `θ = (X^TX + λ·T·I)^{-1}(X^Ty)`. For transfer: `θ = (X^TX + λ_T·T·I)^{-1}(X^Ty + λ_T·T·θ_S)`.

**Standardization:** Features and targets are z-score standardized (mean=0, std=1) using **training set statistics only**. Test set is transformed using the same training statistics. This is done per-experiment, per-stock/target.

### 3.6 `src/transfer_risk.py`

**Functions:**

- `compute_transfer_risk(X_source, y_source, X_target, y_target)` → `float`
  - Computes Wasserstein-2 transfer risk `C_W(S,T)` using the closed-form Gaussian formula:
  1. Estimate `mu_S,X`, `mu_S,Y`, `Sigma_S,X`, `Sigma_S,XY`, `Sigma_S,YX`, `Sigma_S,Y` from source data
  2. Estimate `mu_T,X`, `mu_T,Y`, `Sigma_T,X`, `Sigma_T,XY`, `Sigma_T,YX`, `Sigma_T,Y` from target training data
  3. Compute:
     ```
     bias = (mu_TY - mu_SY - Sigma_SYX @ inv(Sigma_SX) @ (mu_TX - mu_SX))**2
     var_term = (sqrt(Sigma_SYX @ inv(Sigma_SX) @ Sigma_TX @ inv(Sigma_SX) @ Sigma_SXY) 
               - sqrt(Sigma_TYX @ inv(Sigma_TX) @ Sigma_TXY))**2
     transfer_risk = bias + var_term
     ```
  - For multivariate Y (but here Y is scalar), the formulas simplify accordingly
  - Uses `np.linalg.solve` for matrix inversion (numerically stable)

**Note:** The paper computes this on the **signature features** (which are the standardized x_t vectors), treating them as approximately Gaussian. The covariance matrices are sample covariances from training data.

### 3.7 `src/evaluation.py`

**Functions:**

- `compute_mse(y_true, y_pred)` → `float`
  - `np.mean((y_true - y_pred)**2)`

- `compute_r2(y_true, y_pred)` → `float`
  - `1 - SS_res / SS_tot`

- `compute_correlation(y_true, y_pred)` → `float`
  - Pearson correlation via `np.corrcoef`

- `compute_all_metrics(y_true, y_pred)` → `Dict[str, float]`
  - Returns `{"mse": ..., "r2": ..., "corr": ...}`

### 3.8 `src/experiment.py`

**Main experiment runner:**

```python
def run_single_experiment(tickers, lag, order, random_state):
    # 1. Sample 11 stocks (10 source + 1 target)
    source_tickers, target_ticker = sample_stocks(tickers, random_state=random_state)
    
    # 2. Download/Load data
    # 3. Compute signature features for each stock
    # 4. Standardize using training set statistics
    
    # 5. Build source training data (concatenate all 10 source stocks)
    # 6. Build target training data (1 target stock)
    # 7. Build target testing data (1 target stock, Sep 2021 - Sep 2022)
    
    # 8. Direct Learning: fit Ridge on target training, predict on target test
    theta_direct = direct_ridge(X_target_train, y_target_train, LAMBDA_DIRECT)
    y_pred_direct = predict(X_target_test, theta_direct)
    
    # 9. Transfer Learning: pretrain on source, fine-tune on target
    theta_S = pretrain_source(X_source_train, y_source_train, LAMBDA_S)
    theta_T = transfer_ridge(X_target_train, y_target_train, theta_S, LAMBDA_T)
    y_pred_transfer = predict(X_target_test, theta_T)
    
    # 10. Compute metrics
    metrics_direct = compute_all_metrics(y_target_test, y_pred_direct)
    metrics_transfer = compute_all_metrics(y_target_test, y_pred_transfer)
    
    # 11. Compute transfer risk
    risk = compute_transfer_risk(X_source_train, y_source_train, 
                                  X_target_train, y_target_train)
    
    return metrics_direct, metrics_transfer, risk

def run_full_reproduction():
    # Table 1: All (lag, order) combinations
    results_table1 = {}
    for lag in LAGS:
        for order in ORDERS:
            results = [run_single_experiment(tickers, lag, order, rs) 
                       for rs in range(N_EXPERIMENTS)]
            # Average MSE, R2, Corr over 200 experiments
            results_table1[(lag, order)] = aggregate(results)
    
    # Table 2: Correlation matrix at L=5, M=2
    risks, mses, r2s, corrs = [], [], [], []
    for rs in range(N_EXPERIMENTS):
        metrics_d, metrics_t, risk = run_single_experiment(tickers, lag=5, order=2, rs)
        mses.append(metrics_t['mse'])
        r2s.append(-metrics_t['r2'])     # negate for table
        corrs.append(-metrics_t['corr'])  # negate for table
        risks.append(risk)
    # Compute Pearson correlation matrix of [MSE, -R2, -Corr, Transfer Risk]
    corr_matrix = np.corrcoef([mses, r2s, corrs, risks])
```

---

## 4. Critical Implementation Details

### 4.1 Source Data Concatenation
The source training data concatenates features from **all 10 source stocks** into a single training set. Each stock contributes ~2900 training samples (Feb 2010 – Sep 2021, ~11.5 years × ~252 trading days). Total source training size: ~29,000 samples.

### 4.2 Time Augmentation in Path
Within each lag window of length L, time is indexed as `tau = 0, 1/L, 2/L, ..., 1`. This normalization ensures the time component is bounded regardless of lag length.

### 4.3 Standardization
- Compute mean and std of **each feature column** from training data (source training for source, target training for target)
- Apply same transformation to test data
- Standardize **both features and targets**
- For direct learning: use target training statistics
- For transfer learning: use respective training statistics for source/target; **important**: the target training features must be standardized consistently for both the pre-training evaluation and the fine-tuning step

### 4.4 Transfer Ridge Closed-Form Derivation
The transfer Ridge objective: `(1/T_T) Σ(x_i·θ - y_i)² + λ_T‖θ - θ_S‖²`

Expanding: `(1/T_T)(Xθ - y)^T(Xθ - y) + λ_T(θ - θ_S)^T(θ - θ_S)`

Taking derivative = 0: `(2/T_T)X^T(Xθ - y) + 2λ_T(θ - θ_S) = 0`

Solving: `θ = (X^TX/T_T + λ_T·I)^{-1}(X^Ty/T_T + λ_T·θ_S)`

### 4.5 Transfer Risk with Signature Features
The signature features are treated as approximately Gaussian (justified by central limit arguments for high-dimensional features). Sample means and covariances are estimated from the training data. The Wasserstein-2 risk formula is applied using these sample statistics.

### 4.6 Signature Dimension
For path dimension d=3 (time, log-price, log-volume) and truncation order M:
- M=2: dimension = 1 + 3 + 9 = 13
- M=3: dimension = 1 + 3 + 9 + 27 = 40
- M=4: dimension = 1 + 3 + 9 + 27 + 81 = 121

The empty (level-0) constant 1 may or may not be included by iisignature. Need to verify: `iisignature.sig` returns iterated integrals starting from level 1 (the level-0 constant 1 is omitted). So actual dimensions:
- M=2: 3 + 9 = 12 features
- M=3: 3 + 9 + 27 = 39 features  
- M=4: 3 + 9 + 27 + 81 = 120 features

---

## 5. Execution Plan (Step-by-Step)

1. **Create folder structure** and `requirements.txt`
2. **Implement `config.py`** with all constants and IT ticker list
3. **Implement `data_utils.py`** with yfinance download and caching
4. **Implement `signature_utils.py`** with iisignature-based feature computation
5. **Implement `models.py`** with direct and transfer Ridge (closed-form)
6. **Implement `evaluation.py`** with MSE, R², Corr
7. **Implement `transfer_risk.py`** with Wasserstein-2 closed-form
8. **Implement `experiment.py`** with full experiment runner
9. **Create `run_experiment.py`** entry point
10. **Create notebook** `section4_reproduction.ipynb`
11. **Test with 5 experiments** to verify pipeline works
12. **Run full 200 experiments** to reproduce Table 1 and Table 2
13. **Compare results** with paper's Table 1 and Table 2 values

---

## 6. Expected Results (Paper Table 1)

| Lag | Order | Direct MSE | Transfer MSE | Direct R² | Transfer R² | Direct Corr | Transfer Corr |
|-----|-------|-----------|-------------|-----------|-------------|------------|--------------|
| 2   | 2     | 1.21      | 1.66        | 0.07      | 0.08        | 0.03       | 0.04         |
| 2   | 3     | 1.41      | 1.15        | 0.06      | 0.07        | 0.03       | 0.03         |
| 2   | 4     | 1.88      | 1.91        | -0.06     | 0.06        | 0.03       | 0.02         |
| 3   | 2     | 1.75      | 1.55        | 0.06      | 0.07        | -0.01      | -0.01        |
| 3   | 3     | 1.58      | 1.27        | 0.04      | 0.05        | 0.02       | 0.03         |
| 3   | 4     | 1.72      | 1.70        | -0.14     | -0.07       | 0.00       | 0.01         |
| 5   | 2     | 1.62      | 1.54        | 0.08      | 0.08        | 0.04       | 0.06         |
| 5   | 3     | 1.74      | 2.28        | 0.05      | 0.07        | 0.04       | 0.04         |
| 5   | 4     | 1.63      | 2.05        | -0.13     | 0.05        | 0.02       | 0.06         |
| 10  | 2     | 1.31      | 1.75        | 0.08      | 0.08        | 0.06       | 0.07         |
| 10  | 3     | 1.45      | 1.53        | 0.04      | 0.07        | -0.01      | 0.04         |
| 10  | 4     | 2.48      | 2.84        | -0.14     | -0.01       | 0.07       | 0.06         |

---

## 7. EFM Preparation (Future-Proofing, NO EFM code now)

The architecture is designed so that swapping path signatures for EFM signatures requires only changes in `signature_utils.py`:
- Add a `signature_type` parameter: `'path'` (default) or `'efm'`
- The EFM path would use `signature.path_signature.path_to_fm_signature` with optimized lambda
- All other modules (models, evaluation, experiment) remain unchanged
- `requirements.txt` already has EFM dependencies commented out and ready to uncomment

---

## 8. Dependencies

**Runtime (path signatures only):**
- `numpy`, `pandas` — data handling
- `yfinance` — stock data download
- `iisignature` — truncated signature computation (C++ backend, fast)
- `scikit-learn` — optional, for StandardScaler
- `scipy` — matrix operations in transfer risk
- `matplotlib`, `seaborn` — visualization
- `tqdm` — progress bars for 200 experiments
- `tabulate` — table formatting

**EFM-ready (commented out in requirements.txt):**
- `signature` — EFM signature library (JAX-based)
- `jax`, `jaxlib` — JAX runtime
