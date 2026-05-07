# Implementation Plan: Reproducing Section 4 — Transfer Learning for Stock Return Prediction

**Paper:** "Risk of Transfer Learning and its Applications in Finance" (arXiv:2311.03283)  
**Authors:** Haoyang Cao, Haotian Gu, Xin Guo, Mathieu Rosenbaum  
**Goal:** 100% faithful reproduction of Section 4 (Table 1 + Table 2)

---

## 1. Experimental Setup (from Paper)

### Data
- **Source:** Daily close price + trading volume from the **IT sector of the US equity market**
- **Period:** February 2010 to September 2022
- **Stock selection:** Randomly sample **11 stocks**; first 10 = source tasks, last 1 = target task
- **Splits:**
  - Source training: 10 source stocks, Feb 2010 – Sep 2021
  - Target training: 1 target stock, Feb 2010 – Sep 2021
  - Target testing: 1 target stock, Sep 2021 – Sep 2022
- **Averaging:** 200 random selections of stocks
- **Data source:** Yahoo Finance (`yfinance`)

### Feature Construction (Signature Transform)
1. Log-transform: `log(s_tau)`, `log(v_tau)`
2. Target: `y_t = log(s_{t+1}) - log(s_t)` (next-day log return)
3. 3D path: `z_tau = (tau, log(s_tau), log(v_tau))` for `t-L+1 <= tau <= t`
4. Truncated signature of order M: `x_t = S_M(z)`
5. Standardize features and targets (z-score)
6. Hyperparameters: Lag L ∈ {2, 3, 5, 10}, Order M ∈ {2, 3, 4}

### Models
- **Direct Ridge:** `theta = argmin (1/T) Σ(x_t·θ - y_t)² + λ‖θ‖²`  (λ=1.0, on target training)
- **Transfer Ridge:**
  - Pre-train: `θ_S = argmin (1/T_S) Σ(x_{S,t}·θ - y_{S,t})² + λ_S‖θ‖²`  (λ_S=1.0, on source)
  - Fine-tune: `θ_T = argmin (1/T_T) Σ(x_{T,t}·θ - y_{T,t})² + λ_T‖θ - θ_S‖²`  (λ_T=5.0, on target)

### Metrics
- MSE, R², Pearson Correlation (computed on standardized scale)

### Transfer Risk (Table 2)
- Wasserstein-2 output transport risk: `C_W(S,T) = E^O_W(θ̂_S)` with p=2
- Closed-form Gaussian formula computed from sample statistics of signature features
- Correlation matrix of {MSE, -R², -Corr, Transfer Risk} across 200 experiments at L=5, M=2

---

## 2. Folder Structure

```
Transfer learning/
├── plan.md
├── requirements.txt
├── run_experiment.py          # Entry point
├── src/
│   ├── __init__.py
│   ├── config.py              # Constants, tickers, hyperparameters
│   ├── data_utils.py          # yfinance download, caching, sampling
│   ├── signature_utils.py     # Path construction + iisignature
│   ├── models.py              # Direct + Transfer Ridge (closed-form)
│   ├── transfer_risk.py       # W2 transfer risk (Gaussian closed-form)
│   ├── evaluation.py          # MSE, R², Correlation
│   └── experiment.py          # Experiment runner + plots
├── notebooks/
│   └── (for interactive debugging)
└── data/                      # Cached stock data + results (gitignored)
```

---

## 3. Target Results (Paper Table 1)

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

## 4. Target Results (Paper Table 2 — Correlation Matrix at L=5, M=2)

|               | MSE   | -R²   | -Corr | Transfer Risk |
|---------------|-------|-------|-------|---------------|
| MSE           | 1.000 | 0.275 | 0.136 | 0.963         |
| -R²           | 0.275 | 1.000 | 0.588 | 0.401         |
| -Corr         | 0.136 | 0.588 | 1.000 | 0.173         |
| Transfer Risk | 0.963 | 0.401 | 0.173 | 1.000         |

---

## 5. Key Implementation Details

### 5.1 Ridge Closed-Form
Paper: `(1/T) Σ(x_t·θ - y_t)² + λ‖θ‖²`  
Solution: `θ = (XᵀX/T + λI)⁻¹ (Xᵀy/T)`

### 5.2 Transfer Ridge Closed-Form
Objective: `(1/T_T) Σ(x_{T,t}·θ - y_{T,t})² + λ_T‖θ - θ_S‖²`  
Solution: `θ = (XᵀX/T_T + λ_TI)⁻¹ (Xᵀy/T_T + λ_T·θ_S)`

### 5.3 Transfer Risk (Gaussian W2)
```
C_W(S,T) = [μ_TY - μ_SY - Σ_SYX Σ_SX⁻¹(μ_TX - μ_SX)]²
         + (√(Σ_SYX Σ_SX⁻¹ Σ_TX Σ_SX⁻¹ Σ_SXY) - √(Σ_TYX Σ_TX⁻¹ Σ_TXY))²
```

### 5.4 Standardization
- Default: z-score per dataset (source stats for source, target training stats for target + test)
- Option: per-stock (each stock's features standardized independently)

### 5.5 Signature Dimensions (d=3, iisignature)
- M=2: 3+9 = 12 features
- M=3: 3+9+27 = 39 features
- M=4: 3+9+27+81 = 120 features

---

## 6. EFM Future-Proofing

The architecture separates signature computation (`signature_utils.py`) from the rest of the pipeline. To benchmark EFM vs path signatures:
1. Add `signature_type` parameter to `compute_signature_features()`
2. Import `signature.path_signature.path_to_fm_signature` for EFM mode
3. Uncomment EFM dependencies in `requirements.txt`
4. All other modules (models, evaluation, experiment) remain unchanged

---

## 7. Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run full reproduction (200 experiments)
python run_experiment.py

# Quick test with fewer experiments
python run_experiment.py --n-experiments 5

# Regenerate plots from saved results
python run_experiment.py --plot-only results.pkl

# Per-stock standardization
python run_experiment.py --standardization per_stock
```
