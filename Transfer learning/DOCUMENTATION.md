# Transfer Learning for Stock Return Prediction with Path Signatures

> Reproduction and extension of Section 4 from ["Risk of Transfer Learning and its Applications in Finance"](https://arxiv.org/abs/2311.03283) (Cao, Gu, Guo, Rosenbaum, 2023).

---

## Table of Contents

1. [Overview](#overview)
2. [Paper Background](#paper-background)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [Detailed Module Reference](#detailed-module-reference)
   - [config.py](#configpy)
   - [data_utils.py](#data_utilspy)
   - [signature_utils.py](#signature_utilspy)
   - [efm_signature_utils.py](#efm_signature_utilspy)
   - [models.py](#modelspy)
   - [evaluation.py](#evaluationpy)
   - [transfer_risk.py](#transfer_riskpy)
   - [experiment.py](#experimentpy)
   - [run_experiment.py](#run_experimentpy)
   - [run_efm_benchmark.py](#run_efm_benchmarkpy)
   - [download_data.py](#download_datapy)
6. [Pipeline Walkthrough: All 4 Approaches Step-by-Step](#pipeline-walkthrough-all-4-approaches-step-by-step)
   - [Common Preprocessing](#common-preprocessing-all-approaches)
   - [Pipeline 1: Path Signatures + Direct Learning](#pipeline-1-path-signatures--direct-learning)
   - [Pipeline 2: Path Signatures + Transfer Learning](#pipeline-2-path-signatures--transfer-learning)
   - [Pipeline 3: EFM Signatures + Direct Learning](#pipeline-3-efm-signatures--direct-learning)
   - [Pipeline 4: EFM Signatures + Transfer Learning](#pipeline-4-efm-signatures--transfer-learning)
   - [Summary Comparison Table](#summary-4-pipelines-comparison)
7. [How the Two Signatures Differ](#how-the-two-signatures-differ)
8. [Output Files](#output-files)

---

## Overview

This repository implements the stock return prediction experiments from **Section 4** of the paper. The core idea:

- Convert daily price/volume data into **signature features** (truncated iterated-integrals of a 3-dimensional path)
- Predict next-day log returns using **Ridge regression**
- Apply **transfer learning**: pre-train on 10 source stocks, then fine-tune on 1 target stock
- Compute **transfer risk** — a theoretically-grounded quantity that predicts how well transfer learning will work *before* running the full experiment

Two signature methods are implemented:

| Method | Module | In Paper? |
|--------|--------|-----------|
| **Path Signatures** (truncated) | `signature_utils.py` | Yes — Section 4.1, Definition 4.1 |
| **EFM Signatures** (Extended Feynman-Mueller) | `efm_signature_utils.py` | No — experimental extension |

---

## Paper Background

### Section 4.1 — Signature Transform

The paper (Definition 4.1) defines the **signature** of a continuous path `x : [0, T] → R^d` as the infinite collection of iterated integrals:

```
S(x) = (1, S(x)^(1), ..., S(x)^(d), S(x)^(1,1), ..., S(x)^(i,j), ..., S(x)^(d,d), ...)
```

where `S(x)^(i₁,...,iₘ) = ∫∫...∫ dx^(i₁)_t₁ ... dx^(iₘ)_tₘ`.

For prediction, we:
1. Fix a **lag L** (sliding window size) and **order M** (truncation degree)
2. At each time `t`, take the window `[t-L+1, t]` of historical prices and volumes
3. Build a 3D path: `z_τ = (τ, log(s_τ), log(v_τ))`, time-normalized to `[0, 1]`
4. Compute the M-th order truncated signature: `x_t = S_M(z)`
5. Standardize features and targets before model fitting

### Section 4.1 — Ridge Regression

**Direct learning** (Eq 29):
```
θ̂ = argmin (1/T) Σ (x_t·θ - y_t)² + λ‖θ‖²₂
```

**Source pre-training** (Eq 30) — fit on pooled source stock data:
```
θ̂_S = argmin (1/T_S) Σ (x_{S,t}·θ - y_{S,t})² + λ_S‖θ‖²₂
```

**Transfer learning** (Eq 31) — fine-tune on target with source prior:
```
θ̂_T = argmin (1/T_T) Σ (x_{T,t}·θ - y_{T,t})² + λ_T‖θ - θ̂_S‖²₂
```

The regularization constants are `λ_S = 1.0` and `λ_T = 5.0` (paper Section 4.2).

**Key insight**: Eq 31 penalizes the *distance* between the target model `θ` and the pre-trained source model `θ̂_S`. This means the target model is pulled toward the source estimate, with strength controlled by `λ_T`.

### Section 3.3 — Transfer Risk (Theorem 3.1)

Under the Gaussian model where `(X, y) ~ N(μ, Σ)`, the **Wasserstein-based transfer risk** is:

```
C_W(S, T) = bias² + (√var_S - √var_T)²
```

where:
- `bias = μ_TY - μ_SY - w_S·(μ_TX - μ_SX)` — output distribution mismatch
- `var_S = w_S·Σ_TX·w_S`, `var_T = w_T·Σ_TX·w_T` — variance terms
- `w_S = Σ_SX^{-1} Σ_SXY`, `w_T = Σ_TX^{-1} Σ_TXY` — optimal linear predictors

**Theorem 3.1** proves: `C_W(S, T) ≤ R(S, T)` — the transfer risk lower-bounds the regret (the gap between transfer and direct learning). This means **low transfer risk implies good transfer performance**.

### Section 4.2 — Experimental Design

- **Universe**: 59 Information Technology sector stocks, US equity market
- **Period**: February 2010 – September 2022
- **Split**: Train: Feb 2010 – Sep 2021 / Test: Oct 2021 – Sep 2022
- **Per experiment**: Randomly select 11 stocks → 10 source + 1 target
- **Repetitions**: 200 random selections (results averaged)
- **Metrics**: MSE, R², Pearson correlation

---

## Repository Structure

```
Transfer learning/
├── DOCUMENTATION.md          # This file
├── requirements.txt          # Python dependencies
├── download_data.py          # Standalone data downloader
├── run_experiment.py         # Path signature pipeline (paper Section 4)
├── run_efm_benchmark.py      # Path + EFM signature benchmark
├── src/
│   ├── __init__.py           # Empty package marker
│   ├── config.py             # All configuration constants
│   ├── data_utils.py         # Yahoo Finance download, preprocessing
│   ├── signature_utils.py    # Path signature computation (iisignature)
│   ├── efm_signature_utils.py # EFM signature computation (JAX)
│   ├── models.py             # Ridge / Transfer Ridge solvers
│   ├── evaluation.py         # MSE, R², correlation metrics
│   ├── transfer_risk.py      # Wasserstein transfer risk (Section 3.3)
│   └── experiment.py         # Full pipeline: experiments, tables, plots
└── data/
    ├── cache/                # Cached CSV files from Yahoo Finance
    ├── benchmark_results.csv # Raw results from EFM benchmark
    ├── efm_benchmark_results.csv # Averaged summary
    └── *.png                 # Generated plots
```

---

## Quick Start

### Installation

```bash
# Using uv (recommended):
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Install the EFM signature package (external dependency):
uv pip install -e /path/to/Sig/signature
```

### Download Data

```bash
python download_data.py
# Downloads IT sector stocks from Yahoo Finance to data/cache/
# Stops after 30 valid tickers are cached
```

### Run Path Signature Experiment (Paper Section 4)

```bash
python run_experiment.py --n-experiments 200 --n-source 10
# Runs Tables 1 & 2, generates all plots

# Plot-only mode (from saved results):
python run_experiment.py --plot-only data/results.pkl
```

### Run EFM Benchmark (Path + EFM Signatures)

```bash
PYTHONPATH=/path/to/Sig/signature:$PYTHONPATH python run_efm_benchmark.py --n-experiments 5 --n-source 10
# Runs both Path Signature and EFM Signature experiments
# Outputs: benchmark_results.csv, efm_benchmark_results.csv, plots
```

---

## Detailed Module Reference

### `config.py`

Central configuration matching the paper's experimental setup.

| Variable | Value | Paper Reference |
|----------|-------|-----------------|
| `TRAIN_START` | 2010-02-01 | Section 4.2 |
| `TRAIN_END` | 2021-09-30 | Section 4.2 |
| `TEST_START` | 2021-10-01 | Section 4.2 |
| `TEST_END` | 2022-09-30 | Section 4.2 |
| `LAGS` | [2, 3, 5, 10] | Section 4.1 (hyper-parameter L) |
| `ORDERS` | [2, 3, 4] | Section 4.1 (hyper-parameter M) |
| `ALPHA_SOURCE` | 1.0 | Eq 30 (λ_S) |
| `ALPHA_TRANSFER` | 5.0 | Eq 31 (λ_T) |
| `N_EXPERIMENTS` | 200 | Section 4.2 |
| `N_SOURCE` | 10 | Section 4.2 |
| `RANDOM_SEED` | 42 | Fixed for reproducibility |
| `PATH_DIM` | 3 | (time, log_price, log_volume) |

### `data_utils.py`

Handles data acquisition from Yahoo Finance and preprocessing.

**Key functions:**

- **`download_all_data(tickers, start, end)`** — Downloads stocks from Yahoo Finance via `yfinance`. Checks `data/cache/` first; only downloads missing tickers. Uses `curl_cffi` for browser impersonation if available. Returns `dict[ticker → DataFrame]`.

- **`get_valid_tickers(all_data, min_coverage=0.90)`** — Filters tickers covering at least 90% of business days in the date range. Ensures all stocks have sufficient data for the experiment.

- **`sample_stocks(valid_tickers, n_source, random_state)`** — Randomly selects `n_source + 1` stocks without replacement. Returns `(source_list, target)`.

- **`prepare_stock_data(df)`** — Core preprocessing pipeline:
  1. Deduplicate and sort index
  2. Extract Close (or Adj Close) and Volume columns
  3. Remove rows with non-positive prices/volumes
  4. Apply log transform: `log_prices = ln(close)`, `log_volumes = ln(volume + 1)` (the `+1` prevents `ln(0)`)
  5. Compute `log_returns = diff(log_prices)`
  6. Returns `(dates, log_prices, log_volumes, log_returns)`

### `signature_utils.py`

Implements the **path signature** transform exactly as described in paper Section 4.1, Definition 4.1, using the `iisignature` library.

**Key functions:**

- **`build_path(log_prices_window, log_volumes_window)`** — Constructs a 3D path `(τ, log_prices, log_volumes)` with time `τ ∈ [0, 1]` (linearly spaced). This matches the paper's `z_τ = (τ, log(s_τ), log(v_τ))`.

- **`compute_signature_features(log_prices, log_volumes, lag, order)`** — Generates signature features via sliding windows:
  ```
  For each time t (from lag-1 to n-2):
      window_prices = log_prices[t-lag+1 : t+1]
      window_volumes = log_volumes[t-lag+1 : t+1]
      path = build_path(window_prices, window_volumes)
      features[t - (lag-1)] = sig(path, order)
  ```
  Returns `(n - lag, sig_dim)` feature matrix where `sig_dim = siglength(3, order)`.

### `efm_signature_utils.py`

Implements the **Extended Feynman-Mueller (EFM) signature** — an experimental extension not in the original paper.

**Key differences from Path Signatures:**

| Aspect | Path Signature | EFM Signature |
|--------|---------------|---------------|
| Time encoding | Normalized to [0,1] within each window | Measured in years from base date |
| Lambda parameters | None | 3 lambdas: λ_time, λ_price, λ_volume |
| Computation | `iisignature.sig()` | `path_to_fm_signature()` (JAX) |
| Warmup | None | 252 trading days (1 year) |

**`WARMUP_DAYS = 252`** — One trading year of data is used as warmup to stabilize the EFM computation. Features before this period are dropped.

**`compute_efm_features(log_prices, log_volumes, dates, lam, trunc)`** — Core EFM computation:
1. Builds the full path with time in years: `path = (time_years, log_prices, log_volumes)`
2. Converts `lam = [λ_time, λ_price, λ_volume]` to JAX array
3. Calls `path_to_fm_signature(path, trunc, t_grid, lam)` — the EFM kernel operates over the entire time grid, modulated by the lambda parameters
4. Returns `features[:-1]` paired with `targets[warmup:]` — the `[t]` feature predicts the `[t]` return (one-step-ahead)

**Lambda grid** (`run_efm_benchmark.py:_make_lambda_grid`):
- Each λ component is searched over `{0.01, 0.1, 1.0, 10.0}`
- Combinations where all three λ values are equal (e.g., `[0.01, 0.01, 0.01]`) are **excluded**
- Total grid size: 4³ − 4 = **60 combinations**
- Lambda optimization uses in-sample training MSE on the **full training set** (no train/validation split)

### `models.py`

Closed-form Ridge regression solvers with `/n` normalization matching the paper.

- **`ridge_solve(X, y, alpha)`** — Eq 29/30: `θ = (XᵀX/n + αI)⁻¹(Xᵀy/n)`
- **`transfer_ridge_solve(X, y, theta_S, alpha)`** — Eq 31: `θ = (XᵀX/n + αI)⁻¹(Xᵀy/n + α·θ_S)`. The term `α·θ_S` in the RHS pulls the solution toward the source estimate.
- **`predict(X, theta)`** — `ŷ = X·θ`

**Why `/n` normalization?** The paper's objective function (Eq 29) divides the sum of squared errors by `T` (the number of samples). This makes the regularization parameter `λ` independent of the sample size. Solver form: `(XᵀX)/n + αI` instead of `XᵀX + αI`.

### `evaluation.py`

Standard regression metrics:

- **`compute_mse(y_true, y_pred)`** — Mean squared error: `mean((y_true - y_pred)²)`
- **`compute_r2(y_true, y_pred)`** — R² score: `1 - SS_res / SS_tot`
- **`compute_correlation(y_true, y_pred)`** — Pearson correlation coefficient
- **`compute_all_metrics(y_true, y_pred)`** — Returns `dict` with all three metrics

### `transfer_risk.py`

Implements the Wasserstein-based transfer risk from paper Section 3.3, Theorem 3.1, Eq 27.

**`compute_transfer_risk(X_source, y_source, X_target, y_target)`** — Algorithm:

1. **Estimate Gaussian parameters** for source and target: `μ_X, μ_Y, Σ_X, Σ_XY` via maximum likelihood from data (Eq 15)
2. **Solve for optimal linear predictors**: `w_S = Σ_SX⁻¹ Σ_SXY`, `w_T = Σ_TX⁻¹ Σ_TXY` (Eq 16)
3. **Compute bias term**: `bias = μ_TY − μ_SY − w_S·(μ_TX − μ_SX)`
4. **Compute variance term**: `var_S = w_S·Σ_TX·w_S`, `var_T = w_T·Σ_TX·w_T`
5. **Return**: `bias² + (√var_S − √var_T)²`

The bias term measures **output distribution mismatch** (how different the prediction targets are). The variance term measures **input distribution mismatch** (how different the feature distributions are).

**Theorem 3.1 guarantee**: Transfer risk ≤ Regret (gap between transfer and direct learning). So low risk → good transfer.

### `experiment.py`

Orchestrates the full path signature pipeline, reproduces Tables 1 and 2.

**Key functions:**

- **`precompute_features(all_data, lags, orders)`** — Precomputes path signatures for all (ticker, lag, order) combinations. Stores in nested cache: `cache[ticker]["features"][(lag, order)]`. This avoids redundant computation across multiple experiments.

- **`run_single_experiment(cache, valid_tickers, lag, order, exp_id, standardization)`** — Single experiment:
  1. Sample 10 source + 1 target stocks (seeded by `RANDOM_SEED + exp_id`)
  2. Extract signature features and targets, split by `TRAIN_END`
  3. Standardize (two modes: `"dataset"` — after pooling sources; `"per_stock"` — per-stock before pooling)
  4. Fit direct ridge on target training data (Eq 29)
  5. Fit source ridge on pooled source data (Eq 30), then transfer ridge on target (Eq 31)
  6. Compute transfer risk using `compute_transfer_risk()`
  7. Return dict with `direct`, `transfer`, `transfer_risk`, and ticker lists

- **`run_table1(cache, valid_tickers, n_experiments, standardization)`** — Runs `N_EXPERIMENTS` per each (lag, order) combination. Averages results. Returns nested dict keyed by `(lag, order)`.

- **`run_table2(cache, valid_tickers, lag=5, order=2, n_experiments, standardization)`** — Runs for fixed L=5, M=2 (paper Table 2 configuration). Computes the 4×4 correlation matrix between MSE, −R², −Corr, and Transfer Risk. Uses negations of R² and Corr so that "lower = better" for all metrics.

- **`run_full_reproduction(cache, valid_tickers, ...)`** — Top-level orchestration: runs Tables 1 & 2, saves results as `results.pkl`, generates all 6 plot types.

**Plot functions** (output to `data/` directory):
| Function | Output | Description |
|----------|--------|-------------|
| `plot_table1_table` | stdout | ASCII table of all (lag,order) results |
| `plot_table1_heatmap` | `table1_heatmap.png` | 2×3 heatmap grid (Direct/Transfer × MSE/R²/Corr) |
| `plot_direct_vs_transfer_bars` | `direct_vs_transfer_bars.png` | Grouped bar chart comparing Direct vs Transfer |
| `plot_table2_heatmap` | `table2_correlation.png` | Correlation heatmap of 4 metrics |
| `plot_transfer_risk_scatter` | `transfer_risk_scatter.png` | Transfer Risk vs MSE/R²/Corr scatter plots |
| `plot_metric_distributions` | `metric_distributions.png` | Histogram distributions for L=5, M=2 |

### `run_experiment.py`

Entry point script for the paper reproduction.

**Pipeline:**
```
Step 1: Download all IT ticker data (--skip-download to use cache)
Step 2: Filter valid tickers (≥90% coverage)
Step 3: Precompute path signature features for all (lag,order) combos
Step 4: Run experiments → Tables 1 & 2 → plots
```

**CLI arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--n-experiments` | 200 | Number of random stock selections |
| `--n-source` | 10 | Source stocks per experiment |
| `--standardization` | dataset | `dataset` (after pooling) or `per_stock` (before pooling) |
| `--skip-download` | False | Skip Yahoo Finance download |
| `--plot-only` | None | Load pickle file and only generate plots |

### `run_efm_benchmark.py`

Benchmark script comparing both **Path Signatures** and **EFM Signatures**.

**Two-phase execution:**
- **Phase 1**: Precompute all features (EFM features for all lambda/trunc/ticker combos; Path features for all lag/order/ticker combos)
- **Phase 2**: Run experiments for each (lag,order) for Path method, and each trunc level for EFM method

**Key EFM-specific logic:**

- **`_make_lambda_grid()`** — Generates 60 lambda combinations from `{0.01, 0.1, 1.0, 10.0}³`, excluding cases where `λ_time = λ_price = λ_volume`. This ensures the EFM signature modulates time, price, and volume components differently.

- **`_efm_optimize_lambda(efm_cache, tickers, trunc, lambda_grid)`** — Searches the lambda grid by computing EFM features for each lambda, fitting ridge regression on the **full training set**, and selecting the lambda with lowest **in-sample training MSE**. No train/validation split is used. Returns the best `lam` array.

- **`_run_single_efm_experiment()`** — Optimizes lambda separately for direct (target-only) and transfer (source-pooled) settings. The direct model uses the target-optimized lambda; the transfer model uses the source-optimized lambda for feature computation across all stocks.

**Output files:**
| File | Description |
|------|-------------|
| `data/benchmark_results.csv` | Per-experiment raw results |
| `data/efm_benchmark_results.csv` | Averaged summary by method/parameter |
| `data/distributions_mse.png` | Histograms of Direct vs Transfer MSE |

### `download_data.py`

Standalone script for initial data acquisition.

- Downloads from Yahoo Finance using `yfinance` with `curl_cffi` session
- Caches results as CSV in `data/cache/`
- Stops after 30 valid tickers are cached (enough for experiments)
- Delay of 10 seconds between downloads to respect rate limits
- 5 retries per ticker with exponential backoff

---

## Pipeline Walkthrough: All 4 Approaches Step-by-Step

This section traces each pipeline from **raw data to final metrics**, covering every transformation and intermediate result.

### Common Preprocessing (All Approaches)

Every pipeline begins identically:

```
Raw stock data (Yahoo Finance)
  │
  ▼
prepare_stock_data(df)                          # data_utils.py
  ├─ Index: deduplicate, sort, timezone-naive
  ├─ Columns: "Adj Close" → "Close" (fallback) + "Volume"
  ├─ Filter: keep only close > 0 and volume > 0
  ├─ Transform: log_prices = ln(close)
  │              log_volumes = ln(volume + 1)      # +1 handles zero-vol days
  │              log_returns = diff(log_prices)    # one-step-ahead target
  └─ Output: (dates, log_prices[N], log_volumes[N], log_returns[N-1])
```

**Key data constraint**: `len(log_returns) = len(log_prices) - 1`. The return at index `t` is the return from `t` to `t+1`. Every feature-to-target alignment respects this one-step-ahead structure.

**Date split**: All pipelines split at `TRAIN_END = "2021-09-30"`. Dates ≤ this are training; dates > are testing.

---

### Pipeline 1: Path Signatures + Direct Learning

*Reproduces paper Eq (29). Scripts: `run_experiment.py`, `run_efm_benchmark.py` (Path method).*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Feature Computation (for one stock, fixed lag L, order M)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: log_prices[N], log_volumes[N]                                       │
│                                                                             │
│  For each time t from (L-1) to (N-2):    # window centers at t              │
│    idx_start = t - L + 1                 # L points total                   │
│    idx_end   = t + 1                                                       │
│    window_p = log_prices[idx_start : idx_end]   # shape (L,)                │
│    window_v = log_volumes[idx_start : idx_end]   # shape (L,)               │
│                                                                             │
│    path = column_stack([                                                      │
│        linspace(0, 1, L),     # normalized time ∈ [0,1]                    │
│        window_p,               # log-prices                                 │
│        window_v                # log-volumes                                │
│    ])                          # shape (L, 3)                               │
│                                                                             │
│    features[t - (L-1)] = iisignature.sig(path, order=M)                     │
│                          # output shape: (sig_dim,)                          │
│                                                                             │
│  Output: features[N-L, sig_dim]                                             │
│                                                                             │
│  sig_dim = siglength(3, M)   # truncated signature dimension:               │
│    M=2 → 13    M=3 → 40    M=4 → 121                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Feature-Target Alignment                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  feat_dates = dates[ L-1 : -1 ]     # shape (N-L,)                         │
│  targets    = log_returns[ L-1 : ]  # shape (N-L,)                          │
│                                                                             │
│  Alignment: features[t] computed from window ending at time t               │
│             predicts target[t] = log_return from t to t+1                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Train/Test Split                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  mask = feat_dates ≤ pd.Timestamp("2021-09-30")                             │
│  X_train = features[mask]     y_train = targets[mask]                       │
│  X_test  = features[~mask]    y_test  = targets[~mask]                      │
│                                                                             │
│  Sizes (for 2010-2021 train, ~2686 train / ~250 test per stock)            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Standardization                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  μ_X = mean(X_train, axis=0)    σ_X = std(X_train, axis=0) + 1e-10         │
│  μ_y = mean(y_train)            σ_y = std(y_train) + 1e-10                  │
│                                                                             │
│  X_train_n = (X_train - μ_X) / σ_X     y_train_n = (y_train - μ_y) / σ_y    │
│  X_test_n  = (X_test  - μ_X) / σ_X     y_test_n  = (y_test  - μ_y) / σ_y    │
│                                                                             │
│  Statistics computed from TRAIN only; applied identically to TEST.          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Model Fitting — Direct Ridge (paper Eq 29)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  θ_direct = ridge_solve(X_train_n, y_train_n, alpha=λ_S)                    │
│                                                                             │
│  Closed form:  A = (X_train_nᵀ · X_train_n) / n + λ_S · I                   │
│                b = (X_train_nᵀ · y_train_n) / n                             │
│                θ_direct = solve(A, b)                                       │
│                                                                             │
│  λ_S = ALPHA_SOURCE = 1.0  (from paper Section 4.2)                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Prediction & Metrics                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  y_pred = X_test_n @ θ_direct                                               │
│                                                                             │
│  metrics = {                                                                │
│      "mse":  mean((y_test_n - y_pred)²)                                     │
│      "r2":   1 - SS_res / SS_tot                                            │
│      "corr": corrcoef(y_test_n, y_pred)                                     │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key configuration**: `ALPHA_SOURCE = 1.0`. Only the target stock's training data is used. No source stocks involved.

---

### Pipeline 2: Path Signatures + Transfer Learning

*Reproduces paper Eqs (30)-(31). Scripts: same as Pipeline 1 but with transfer path.*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEPS 1-4: Same as Pipeline 1 but for BOTH source stocks AND target stock   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Source stocks (10 tickers): for each, compute features → split → (train)   │
│  Target stock (1 ticker):     compute features → split → train + test       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5A: Pool Source Data                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each source ticker i:                                                  │
│      X_src_i = training features of ticker i  (n_i × d)                     │
│      y_src_i = training targets  of ticker i  (n_i,)                        │
│                                                                             │
│  X_source = vstack([X_src_1, ..., X_src_10])   # shape (N_total, d)        │
│  y_source = concat([y_src_1, ..., y_src_10])   # shape (N_total,)           │
│                                                                             │
│  Where N_total ≈ 10 × 2686 = 26,860 rows (10 source stocks pooled)          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5B: Standardization (dataset mode — paper's default)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  μ_S = mean(X_source, axis=0)    σ_S = std(X_source, axis=0) + 1e-10        │
│  μ_sy = mean(y_source)           σ_sy = std(y_source) + 1e-10               │
│                                                                             │
│  X_source_n = (X_source - μ_S) / σ_S                                        │
│  y_source_n = (y_source - μ_sy) / σ_sy                                      │
│                                                                             │
│  Alternately "per_stock" mode: standardize each source individually         │
│  before pooling (available in run_experiment.py only).                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5C: Target Standardization                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Same as Pipeline 1 Step 4: standardize target using its OWN train stats.   │
│                                                                             │
│  X_t_tr_n, y_t_tr_n = standardized target TRAIN                             │
│  X_t_te_n, y_t_te_n = standardized target TEST                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6A: Source Pre-training (paper Eq 30)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  θ_S = ridge_solve(X_source_n, y_source_n, alpha=λ_S)                       │
│                                                                             │
│  Closed form:  A = (X_source_nᵀ · X_source_n) / N + λ_S · I                │
│                b = (X_source_nᵀ · y_source_n) / N                           │
│                θ_S = solve(A, b)                                            │
│                                                                             │
│  λ_S = ALPHA_SOURCE = 1.0                                                   │
│                                                                             │
│  θ_S is the "pre-trained model" — it captures patterns across 10 stocks.    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6B: Transfer Ridge on Target (paper Eq 31)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  θ_T = transfer_ridge_solve(X_t_tr_n, y_t_tr_n, θ_S, alpha=λ_T)             │
│                                                                             │
│  Closed form:  A = (X_t_tr_nᵀ · X_t_tr_n) / n + λ_T · I                     │
│                b = (X_t_tr_nᵀ · y_t_tr_n) / n + λ_T · θ_S                   │
│                             ^^^^^^^^^^^^^^^^^^^                              │
│                             This TERM pulls θ_T toward θ_S                   │
│                θ_T = solve(A, b)                                            │
│                                                                             │
│  λ_T = ALPHA_TRANSFER = 5.0                                                 │
│                                                                             │
│  Interpretation: λ_T controls how much the target model can deviate from    │
│  the source model. Higher λ_T = more trust in source knowledge.             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Prediction & Metrics                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  y_pred = X_t_te_n @ θ_T                                                    │
│                                                                             │
│  metrics = { "mse", "r2", "corr" }  (same as Pipeline 1)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: Transfer Risk (paper Theorem 3.1, Eq 27)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  compute_transfer_risk(X_source_n, y_source_n, X_t_tr_n, y_t_tr_n)          │
│                                                                             │
│  Input: standardized source training data + standardized target training    │
│                                                                             │
│  1. Gaussian estimation (paper Eq 15):                                      │
│     For source: μ_SX, μ_SY, Σ_SX, Σ_SXY ← estimate_gaussian_params()       │
│     For target: μ_TX, μ_TY, Σ_TX, Σ_TXY ← estimate_gaussian_params()       │
│                                                                             │
│  2. Optimal linear predictors (paper Eq 16):                                │
│     w_S = Σ_SX⁻¹ · Σ_SXY      (source optimal model)                        │
│     w_T = Σ_TX⁻¹ · Σ_TXY      (target optimal model)                        │
│                                                                             │
│  3. Decomposition (paper Eq 27):                                            │
│     bias  = μ_TY - μ_SY - w_S · (μ_TX - μ_SX)                              │
│     bias² = output distribution mismatch                                    │
│                                                                             │
│     var_S = w_S · Σ_TX · w_S   (source variance on target distribution)      │
│     var_T = w_T · Σ_TX · w_T   (target variance on target distribution)      │
│     var_term = (√var_S - √var_T)²   (input distribution mismatch)            │
│                                                                             │
│  4. Transfer risk = bias² + var_term                                        │
│                                                                             │
│  Theorem 3.1: Transfer Risk ≤ Regret. Lower risk = better transferability.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key difference from Direct**: θ_T is regularized toward θ_S, not toward zero. The source model acts as a "prior" that the target model can lean on when target data is limited.

---

### Pipeline 3: EFM Signatures + Direct Learning

*Experimental extension — not in the paper. Script: `run_efm_benchmark.py` (EFM method, direct path).*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: EFM Feature Computation (for one stock, fixed trunc, fixed lambda)  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: log_prices[N], log_volumes[N], dates[N]                             │
│  Param: lam = [λ_time, λ_price, λ_volume]    # 3-element float array       │
│         trunc = M   # 2, 3, or 4                                            │
│                                                                             │
│  (a) Build full path (all dates, no sliding window):                        │
│      base_date = dates[252]                   # after 1yr warmup            │
│      time_years = [(d - base_date).days / 365.0 for d in dates]             │
│      path = column_stack([time_years, log_prices, log_volumes])             │
│             # shape (N, 3)                                                   │
│             # Time dimension = actual calendar years, not [0,1]             │
│                                                                             │
│  (b) Compute EFM signature:                                                 │
│      lam_jax = jnp.array(lam)                                                │
│      SigS = path_to_fm_signature(                                           │
│          path=jnp.array(path),                                              │
│          trunc=trunc,                                                       │
│          t_grid=jnp.array(time_years),                                     │
│          lam=lam_jax                                                        │
│      )                                                                      │
│      features = np.array(SigS.array.T)     # shape (N, sig_dim)            │
│                                                                             │
│  (c) Align features with one-step-ahead targets:                            │
│      features = features[:-1]                # drop last (no target)        │
│      targets  = log_returns[warmup_days:]    # from day 253 onward           │
│      feat_dates = dates[warmup_days:-1]      # corresponding dates           │
│                                                                             │
│  Output: features[N-warmup-1, sig_dim], targets[N-warmup-1], dates          │
│                                                                             │
│  sig_dim: M=2→13   M=3→40   M=4→121   (same geometric formula as path sig) │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Lambda Optimization (EFM-specific)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  This step searches the 60-combo lambda grid to find the best λ for the     │
│  TARGET stock. Optimization uses FULL TRAINING SET (no train/val split).    │
│                                                                             │
│  best_lam = None   best_mse = inf                                           │
│                                                                             │
│  For each lam in lambda_grid:                                               │
│    │  Compute EFM features for target ticker with this lam                  │
│    │  Split by TRAIN_END → X_train, y_train                                 │
│    │  Standardize X_train, y_train                                          │
│    │  Fit ridge_solve(X_train_n, y_train_n, α=λ_S)                          │
│    │  y_pred = X_train_n @ θ                                                │
│    │  mse = mean((y_train_n - y_pred)²)  ← IN-SAMPLE training MSE           │
│    │  If mse < best_mse: best_lam = lam                                     │
│                                                                             │
│  Output: best_lam (np.ndarray of shape (3,))                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Final Model with Best Lambda                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Re-compute EFM features using best_lam for the target stock.               │
│                                                                             │
│  X_tr, y_tr, X_te, y_te = split_by_date(feat_dates, features, targets,     │
│                                          split_date=TRAIN_END)              │
│                                                                             │
│  Standardize using TRAIN stats (same as Path Sig Pipeline 1 Step 4).        │
│                                                                             │
│  θ = ridge_solve(X_tr_n, y_tr_n, α=λ_S)  # Eq 29                            │
│                                                                             │
│  y_pred = X_te_n @ θ                                                        │
│  metrics = compute_all_metrics(y_te_n, y_pred)                              │
│                                                                             │
│  Output: best_lam, {mse, r2, corr}                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key differences from Path Sig Direct**:
1. EFM uses a **single full-history path** instead of sliding windows
2. A **lambda grid search** optimizes per-channel exponential weighting
3. **252-day warmup** period is dropped (1 trading year)
4. Time is measured in **calendar years** from the base date, not [0,1]

---

### Pipeline 4: EFM Signatures + Transfer Learning

*Experimental extension — not in the paper. Script: `run_efm_benchmark.py` (EFM method, transfer path).*

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Lambda Optimization — Two Separate Searches                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  (a) Direct lambda (on TARGET stock only):                                  │
│      best_lam_d = _efm_optimize_lambda(efm_cache, [target_ticker],          │
│                                         trunc, lambda_grid)                 │
│      → Searches 60 lambdas, picks best in-sample MSE on target train data   │
│                                                                             │
│  (b) Transfer lambda (on SOURCE stocks pooled):                             │
│      best_lam_t = _efm_optimize_lambda(efm_cache, source_tickers,           │
│                                         trunc, lambda_grid)                 │
│      → For each lambda: pools training data from all 10 source stocks,      │
│        fits ridge, computes in-sample MSE, picks best lambda                │
│                                                                             │
│  best_lam_d ≠ best_lam_t in general. The direct model is tuned for the      │
│  target stock's own dynamics; the transfer model is tuned for the pooled    │
│  source dynamics.                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Direct Model Evaluation (using best_lam_d)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Compute EFM features for target ticker with best_lam_d at trunc M.         │
│  Split → standardize → fit ridge → predict → metrics.                       │
│  (Same as Pipeline 3 Steps 3).                                              │
│                                                                             │
│  Output: m_direct = {mse, r2, corr}                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Source Model Pre-training (using best_lam_t)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  For each SOURCE ticker i:                                                  │
│      Compute EFM features with best_lam_t at trunc M                        │
│      Split → take training portion → X_src_i, y_src_i                       │
│                                                                             │
│  X_source = vstack([X_src_1, ..., X_src_10])                               │
│  y_source = concat([y_src_1, ..., y_src_10])                               │
│                                                                             │
│  Standardize pooled source data.                                            │
│                                                                             │
│  θ_S = ridge_solve(X_source_n, y_source_n, α=λ_S)   # paper Eq 30          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Transfer Ridge on Target (using best_lam_t)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Compute EFM features for TARGET ticker with best_lam_t at trunc M.         │
│  Split → standardize target using its own train stats.                      │
│                                                                             │
│  θ_T = transfer_ridge_solve(X_t_tr_n, y_t_tr_n, θ_S, α=λ_T)                 │
│       # paper Eq 31: pulls θ_T toward source model θ_S                      │
│                                                                             │
│  y_pred = X_t_te_n @ θ_T                                                    │
│  m_transfer = compute_all_metrics(y_t_te_n, y_pred)                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Transfer Risk                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  risk = compute_transfer_risk(X_source_n, y_source_n, X_t_tr_n, y_t_tr_n)   │
│       # Same computation as Pipeline 2 Step 8                               │
│                                                                             │
│  Output: risk (float)                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key difference from Path Sig Transfer**: The EFM transfer pipeline performs **two independent lambda grid searches** — one for the direct model (on target data) and one for the transfer model (on source data). The transfer model uses `best_lam_t` (optimized on sources) for feature computation on BOTH source and target stocks during the transfer step, ensuring feature-space consistency.

---

### Summary: 4 Pipelines Comparison

| # | Method | Features | Lambda | Model | Risk |
|---|--------|----------|--------|-------|------|
| 1 | Path + Direct | Sliding window truncated sig, (L,M) | None | `ridge_solve(target_train, α=λ_S)` | N/A |
| 2 | Path + Transfer | Sliding window truncated sig, (L,M) | None | `ridge_solve(source_pool, α=λ_S)` → `transfer_ridge(target_train, θ_S, α=λ_T)` | `bias² + (√var_S - √var_T)²` |
| 3 | EFM + Direct | Full-path EFM sig, (lam, trunc) | Grid search (60 combos) on target train MSE | `ridge_solve(target_train, α=λ_S)` with best λ | N/A |
| 4 | EFM + Transfer | Full-path EFM sig, (lam, trunc) | Two grid searches: best_lam_d on target, best_lam_t on sources | `ridge_solve(source_pool, α=λ_S)` → `transfer_ridge(target_train, θ_S, α=λ_T)` with best_lam_t | `bias² + (√var_S - √var_T)²` |

| Aspect | Path Signature (paper) | EFM Signature (experimental) |
|--------|----------------------|------------------------------|
| **Window type** | Sliding, fixed length L | Single path, full history |
| **Time encoding** | `linspace(0, 1, L)` per window | Calendar years from base date |
| **Parameters** | `(lag, order)` — L=2,3,5,10; M=2,3,4 | `(lam, trunc)` — 60 λ combos × trunc=2,3,4 |
| **Library** | `iisignature.sig()` (C++) | `path_to_fm_signature()` (JAX) |
| **Warmup** | None (minimal window needed) | 252 trading days (1 year) |
| **Std. modes** | `dataset` or `per_stock` | `dataset` only (pool-then-standardize) |
| **Ridge α** | `α_S = 1.0`, `α_T = 5.0` | `α_S = 1.0`, `α_T = 5.0` (same) |
| **Train/test split** | ≤ vs > `TRAIN_END` | ≤ vs > `TRAIN_END` (same) |
| **Metrics** | MSE, R², Corr | MSE, R², Corr (same) |
| **Script** | `run_experiment.py` | `run_efm_benchmark.py` |

---

## How the Two Signatures Differ

### Path Signature (paper approach)

The standard truncated signature operates on **sliding windows** of fixed length `L`. Each window forms an independent path, time-normalized to `[0, 1]`:

```
Window: [t-L+1, t]
Path:   z_τ = (τ, log(s_τ), log(v_τ)),  τ ∈ [0, 1]
Output: S_M(z) — truncated signature of order M
```

This is the approach described in paper Section 4.1, Definition 4.1, and used throughout Section 4.2 (Tables 1 and 2).

- Library: `iisignature`
- Parameterization: `(lag, order)` → sliding window controls the memory length
- Computational: O(`n` windows × `M`-order signature)

### EFM Signature (experimental extension)

The Extended Feynman-Mueller signature operates on a **single long path** covering the full history, with exponential weighting controlled by lambda parameters:

```
Full path:     z_t = (t, log(s_t), log(v_t)),  t in years
Lambda:        [λ_time, λ_price, λ_volume]
Output:        EFM signature modulated by lambdas
```

Each lambda controls the exponential decay rate for that dimension. The signature captures the full history but with varying time/price/volume sensitivities. This is an experimental extension not present in the paper.

- Library: `signature.path_signature` (external JAX-based module)
- Parameterization: `(lam, trunc)` → lambdas control weighting per dimension
- Precomputation required: 252-day warmup period, full path computation
- Lambda grid search: 60 combos, optimized on full training set (in-sample MSE)

### Why the Lambda Grid Excludes Equal Parameters

The EFM signature is designed to modulate time, price, and volume with different sensitivities. When `λ_time = λ_price = λ_volume`, the EFM degenerates to a uniform exponential weighting, losing the ability to capture different dynamics per dimension. The 4 equal-parameter combinations are excluded to ensure a meaningful search over differential temporal and financial dynamics.

---

## Output Files

All output is written to `data/`:

| File | Generated By | Contents |
|------|-------------|----------|
| `cache/*.csv` | `download_data.py` | Raw stock data from Yahoo Finance |
| `results.pkl` | `run_experiment.py` | Pickled Table 1 & 2 results |
| `table1_heatmap.png` | `run_experiment.py` | Heatmap of Direct vs Transfer performance |
| `table2_correlation.png` | `run_experiment.py` | Correlation matrix heatmap (Table 2) |
| `direct_vs_transfer_bars.png` | `run_experiment.py` | Bar chart comparison |
| `transfer_risk_scatter.png` | `run_experiment.py` | Transfer Risk vs metrics scatter |
| `metric_distributions.png` | `run_experiment.py` | Histogram distributions |
| `benchmark_results.csv` | `run_efm_benchmark.py` | Raw per-experiment results |
| `efm_benchmark_results.csv` | `run_efm_benchmark.py` | Averaged summary |
| `distributions_mse.png` | `run_efm_benchmark.py` | Path vs EFM MSE distributions |

---

## References

- Cao, H., Gu, H., Guo, X., & Rosenbaum, M. (2023). *Risk of Transfer Learning and its Applications in Finance.* arXiv:2311.03283v1.
- Cao, H., Gu, H., Guo, X., & Rosenbaum, M. (2023). *Feasibility and Transferability of Transfer Learning: A Mathematical Framework.* arXiv:2301.11542.
- Chen, K.-T. (1954, 1957, 1958). Foundational works on iterated integrals and path signatures.
- `iisignature` library: https://github.com/bottler/iisignature
- `signature` library (EFM): External package referenced by `efm_signature_utils.py`
