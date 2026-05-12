"""
Benchmark comparing Path Signatures vs EFM Signatures for transfer learning.

EXPERIMENTAL EXTENSION — not part of the original paper.

Runs the same transfer-learning protocol with two feature families:
  1. **Path Signatures**: sliding-window truncated signatures (paper Section 4).
  2. **EFM Signatures**: Extended Feynman-Mueller signatures with learnable
     per-channel exponential weighting via lambda parameters.

Features:
  - Two-phase execution: precompute all features, then run experiments.
  - EFM lambda optimization via grid search (60 combinations).
  - Results saved as CSV + summary table + distribution histograms.

CLI arguments:
  --n-experiments    Number of random stock selections (default 5)
  --n-source         Number of source stocks per experiment (default 10)
"""

import sys
import os
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add the external signature package to the path (needed for EFM)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "signature"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate
from tqdm import tqdm

from src.config import (
    TRAIN_END, ALPHA_SOURCE, ALPHA_TRANSFER,
    RANDOM_SEED, BASE_DIR, LAGS, ORDERS,
)
from src.data_utils import (
    download_all_data, get_valid_tickers, prepare_stock_data,
    sample_stocks,
)
from src.models import ridge_solve, transfer_ridge_solve, predict
from src.evaluation import compute_all_metrics
from src.transfer_risk import compute_transfer_risk
from src.signature_utils import compute_signature_features
from src.efm_signature_utils import (
    compute_efm_features, WARMUP_DAYS,
)

# EFM truncation orders (maps to paper's M for comparison)
EFM_TRUNCS = [2, 3, 4]
# Default number of experiments (smaller than paper's 200 since EFM is expensive)
N_EXPERIMENTS = 5


def _make_lambda_grid():
    """Generate 60 lambda combinations for EFM hyperparameter tuning.

    Each lambda vector is [λ_time, λ_price, λ_volume] drawn from
    {0.01, 0.1, 1.0, 10.0}.  Combinations where all three are equal
    are excluded (yielding 4³ - 4 = 60 combos).

    Returns:
        list[np.ndarray]: List of 3-element lambda arrays.
    """
    lam_vals = np.array([0.01, 0.1, 1.0, 10.0])
    return [
        np.array([lt, lp, lv])
        for lt, lp, lv in itertools.product(lam_vals, lam_vals, lam_vals)
        if not (lt == lp == lv)
    ]


def split_by_date(feat_dates, features, targets, split_date):
    """Split features/targets into train and test sets by date.

    Args:
        feat_dates (array-like): Dates for each row.
        features (np.ndarray): Feature matrix.
        targets (np.ndarray): Target vector.
        split_date (str): Date boundary (YYYY-MM-DD format).

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    ts = pd.Timestamp(split_date)
    mask = np.array([d <= ts for d in feat_dates])
    return features[mask], targets[mask], features[~mask], targets[~mask]


# %%
def precompute_efm_features(all_data, valid_tickers, lambda_grid, truncs):
    """Precompute EFM signature features for all (ticker, lambda, trunc) combos.

    Args:
        all_data (dict): Ticker -> yfinance DataFrame.
        valid_tickers (list[str]): Tickers to process.
        lambda_grid (list[np.ndarray]): List of lambda vectors.
        truncs (list[int]): EFM truncation orders.

    Returns:
        tuple: (stock_data, efm_cache)
            stock_data: dict[ticker] -> (dates, log_prices, log_volumes, log_returns)
            efm_cache: dict[(ticker, tuple(lam), trunc)] -> (features, targets, feat_dates)
    """
    print(f"\nPrecomputing EFM features: {len(valid_tickers)} tickers × "
          f"{len(lambda_grid)} lambdas × {len(truncs)} truncs = "
          f"{len(valid_tickers) * len(lambda_grid) * len(truncs)} computations")

    stock_data = {}
    efm_cache = {}

    for ticker in tqdm(valid_tickers, desc="Stocks"):
        dates, lp, lv, lr = prepare_stock_data(all_data[ticker])
        stock_data[ticker] = (dates, lp, lv, lr)

        for lam in lambda_grid:
            for trunc in truncs:
                try:
                    feats, tgts, fd = compute_efm_features(lp, lv, dates, lam, trunc)
                    efm_cache[(ticker, tuple(lam.tolist()), trunc)] = (feats, tgts, fd)
                except Exception:
                    pass

    return stock_data, efm_cache


def precompute_path_features(all_data, valid_tickers):
    """Precompute path signature features for all (ticker, lag, order) combos.

    Args:
        all_data (dict): Ticker -> yfinance DataFrame.
        valid_tickers (list[str]): Tickers to process.

    Returns:
        dict: path_cache[(ticker, lag, order)] -> (features, dates, log_returns)
    """
    print(f"\nPrecomputing path signature features: {len(valid_tickers)} tickers")
    path_cache = {}

    for ticker in tqdm(valid_tickers, desc="Path features"):
        dates, lp, lv, lr = prepare_stock_data(all_data[ticker])
        for lag in LAGS:
            for order in ORDERS:
                feats = compute_signature_features(lp, lv, lag, order)
                path_cache[(ticker, lag, order)] = (feats, dates, lr)

    return path_cache


# %%
def _run_single_path_experiment(path_cache, source_tickers, target_ticker, lag, order):
    """Run one Path Signature experiment: pool sources, fit direct + transfer ridge.

    Args:
        path_cache (dict): Path feature cache.
        source_tickers (list[str]): Source stock tickers.
        target_ticker (str): Target stock ticker.
        lag (int): Window lag.
        order (int): Signature order.

    Returns:
        tuple: (metrics_direct, metrics_transfer, transfer_risk)
    """
    source_X_list, source_y_list = [], []
    for ticker in source_tickers:
        feats, dates, lr = path_cache[(ticker, lag, order)]
        fd = dates[lag - 1:-1]         # feature dates (aligned)
        tgts = lr[lag - 1:]             # targets (next-day returns)
        X_tr_s, y_tr_s, _, _ = split_by_date(fd, feats, tgts, TRAIN_END)
        source_X_list.append(X_tr_s)
        source_y_list.append(y_tr_s)

    # Pool all source data
    X_source = np.vstack(source_X_list)
    y_source = np.concatenate(source_y_list)

    # Standardize pooled sources
    mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
    my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
    X_source_n = (X_source - mu_s) / sd_s
    y_source_n = (y_source - my_s) / sy_s

    # Load and split target stock data
    feats_t, dates_t, lr_t = path_cache[(target_ticker, lag, order)]
    fd_t = dates_t[lag - 1:-1]
    tgts_t = lr_t[lag - 1:]
    X_tr, y_tr, X_te, y_te = split_by_date(fd_t, feats_t, tgts_t, TRAIN_END)

    # Standardize target using its own training stats
    mu_t, sd_t = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
    my_t, sy_t = np.mean(y_tr), np.std(y_tr) + 1e-10
    X_tr_n = (X_tr - mu_t) / sd_t
    y_tr_n = (y_tr - my_t) / sy_t
    X_te_n = (X_te - mu_t) / sd_t
    y_te_n = (y_te - my_t) / sy_t

    # Direct learning: ridge on target only
    theta_d = ridge_solve(X_tr_n, y_tr_n, ALPHA_SOURCE)
    y_pred_d = predict(X_te_n, theta_d)
    m_direct = compute_all_metrics(y_te_n, y_pred_d)

    # Transfer learning: source pre-training then transfer ridge
    theta_S = ridge_solve(X_source_n, y_source_n, ALPHA_SOURCE)
    theta_T = transfer_ridge_solve(X_tr_n, y_tr_n, theta_S, ALPHA_TRANSFER)
    y_pred_t = predict(X_te_n, theta_T)
    m_transfer = compute_all_metrics(y_te_n, y_pred_t)

    # Transfer risk between source and target distributions
    risk = compute_transfer_risk(X_source_n, y_source_n, X_tr_n, y_tr_n)

    return m_direct, m_transfer, risk


# %%
def _efm_optimize_lambda(efm_cache, tickers, trunc, lambda_grid):
    """Find the best EFM lambda for a set of tickers by minimizing in-sample MSE.

    For each lambda in the grid, pools training data from all given tickers,
    fits a direct ridge model, and computes the in-sample MSE.  Returns the
    lambda with the lowest MSE.

    Args:
        efm_cache (dict): EFM feature cache.
        tickers (list[str]): Tickers to pool (source stocks or target stock).
        trunc (int): EFM truncation order.
        lambda_grid (list[np.ndarray]): Candidate lambda vectors.

    Returns:
        np.ndarray or None: Best lambda vector, or None if no valid data found.
    """
    best_lam = None
    best_lam_mse = np.inf

    for lam in lambda_grid:
        lam_key = tuple(lam.tolist())
        all_X, all_y = [], []

        for ticker in tickers:
            entry = efm_cache.get((ticker, lam_key, trunc))
            if entry is None:
                continue
            feats, tgts, fd = entry
            X_tr, y_tr, _, _ = split_by_date(fd, feats, tgts, TRAIN_END)
            if len(X_tr) < 50:
                continue
            all_X.append(X_tr)
            all_y.append(y_tr)

        if not all_X:
            continue

        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)

        mu_x, sd_x = np.mean(X_train, axis=0), np.std(X_train, axis=0) + 1e-10
        mu_y, sd_y = np.mean(y_train, axis=0), np.std(y_train, axis=0) + 1e-10
        X_tr_n = (X_train - mu_x) / sd_x
        y_tr_n = (y_train - mu_y) / sd_y

        theta = ridge_solve(X_tr_n, y_tr_n, ALPHA_SOURCE)
        y_pred = predict(X_tr_n, theta)
        mse = float(np.mean((y_tr_n - y_pred) ** 2))

        if mse < best_lam_mse:
            best_lam_mse = mse
            best_lam = lam

    return best_lam


def _run_single_efm_experiment(efm_cache, source_tickers, target_ticker,
                                trunc, lambda_grid):
    """Run one EFM experiment with lambda optimization.

    Steps:
      1. Optimize lambda for direct learning on the target stock.
      2. Fit direct ridge with the best lambda and evaluate.
      3. Optimize lambda for transfer learning on the source stocks.
      4. Fit source ridge, then transfer ridge with source prior, and evaluate.
      5. Compute transfer risk.

    Args:
        efm_cache (dict): EFM feature cache.
        source_tickers (list[str]): Source stock tickers.
        target_ticker (str): Target stock ticker.
        trunc (int): EFM truncation order.
        lambda_grid (list[np.ndarray]): Candidate lambda vectors.

    Returns:
        tuple: (m_direct, best_lam_d, m_transfer, best_lam_t, risk)
            Any element may be None if computation failed.
    """
    # --- Phase 1: Direct learning with lambda optimization ---
    best_lam_d = _efm_optimize_lambda(
        efm_cache, [target_ticker], trunc, lambda_grid,
    )

    if best_lam_d is None:
        return None, None, None, None

    lam_key_d = tuple(best_lam_d.tolist())
    feats, tgts, fd = efm_cache[(target_ticker, lam_key_d, trunc)]
    X_tr, y_tr, X_te, y_te = split_by_date(fd, feats, tgts, TRAIN_END)

    # Standardize target
    mu_x, sd_x = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
    mu_y, sd_y = np.mean(y_tr), np.std(y_tr) + 1e-10
    X_tr_n = (X_tr - mu_x) / sd_x
    y_tr_n = (y_tr - mu_y) / sd_y
    X_te_n = (X_te - mu_x) / sd_x
    y_te_n = (y_te - mu_y) / sd_y

    # Direct ridge on target
    theta = ridge_solve(X_tr_n, y_tr_n, ALPHA_SOURCE)
    y_pred = predict(X_te_n, theta)
    m_direct = compute_all_metrics(y_te_n, y_pred)

    # --- Phase 2: Transfer learning with lambda optimization on sources ---
    best_lam_t = _efm_optimize_lambda(
        efm_cache, source_tickers, trunc, lambda_grid,
    )

    if best_lam_t is None:
        return m_direct, best_lam_d, None, None

    lam_key_t = tuple(best_lam_t.tolist())

    source_X_list, source_y_list = [], []
    for ticker in source_tickers:
        entry = efm_cache.get((ticker, lam_key_t, trunc))
        if entry is None:
            continue
        feats_s, tgts_s, fd_s = entry
        X_tr_s, y_tr_s, _, _ = split_by_date(fd_s, feats_s, tgts_s, TRAIN_END)
        source_X_list.append(X_tr_s)
        source_y_list.append(y_tr_s)

    if not source_X_list:
        return m_direct, best_lam_d, None, None

    X_source = np.vstack(source_X_list)
    y_source = np.concatenate(source_y_list)

    mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
    my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
    X_source_n = (X_source - mu_s) / sd_s
    y_source_n = (y_source - my_s) / sy_s

    theta_S = ridge_solve(X_source_n, y_source_n, ALPHA_SOURCE)

    entry_t = efm_cache.get((target_ticker, lam_key_t, trunc))
    if entry_t is None:
        return m_direct, best_lam_d, None, None
    feats_t, tgts_t, fd_t = entry_t
    X_tr, y_tr, X_te, y_te = split_by_date(fd_t, feats_t, tgts_t, TRAIN_END)

    mu_x, sd_x = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
    mu_y, sd_y = np.mean(y_tr), np.std(y_tr) + 1e-10
    X_tr_n = (X_tr - mu_x) / sd_x
    y_tr_n = (y_tr - mu_y) / sd_y
    X_te_n = (X_te - mu_x) / sd_x
    y_te_n = (y_te - mu_y) / sd_y

    theta_T = transfer_ridge_solve(X_tr_n, y_tr_n, theta_S, ALPHA_TRANSFER)
    y_pred = predict(X_te_n, theta_T)
    m_transfer = compute_all_metrics(y_te_n, y_pred)

    risk = compute_transfer_risk(X_source_n, y_source_n, X_tr_n, y_tr_n)

    return m_direct, best_lam_d, m_transfer, best_lam_t, risk


# %%
def run_all_experiments(all_data, valid_tickers, n_experiments=N_EXPERIMENTS):
    """Run the full EFM benchmark: precompute features, then run all experiments.

    Two-phase execution:
      PHASE 1: Precompuate path signature and EFM features for all needed tickers.
      PHASE 2: For each experiment, run path signature and EFM protocols,
               collecting results as DataFrame rows.

    Args:
        all_data (dict): Ticker -> yfinance DataFrame.
        valid_tickers (list[str]): Pool of valid tickers.
        n_experiments (int): Number of experiments to run.

    Returns:
        pd.DataFrame: Raw per-experiment results (one row per method/lag/order/trunc).
    """
    lambda_grid = _make_lambda_grid()

    needed_tickers = set()
    for exp_id in range(n_experiments):
        seed = RANDOM_SEED + exp_id
        source_tickers, target_ticker = sample_stocks(
            valid_tickers, n_source=10, random_state=seed,
        )
        needed_tickers.update(source_tickers)
        needed_tickers.add(target_ticker)
    needed_tickers = sorted(needed_tickers)

    print(f"Tickers needed for {n_experiments} experiments: {len(needed_tickers)}")

    print("=" * 60)
    print("  PHASE 1: Precomputing features")
    print("=" * 60)

    _, efm_cache = precompute_efm_features(
        all_data, needed_tickers, lambda_grid, EFM_TRUNCS,
    )
    path_cache = precompute_path_features(all_data, needed_tickers)

    print("\n" + "=" * 60)
    print(f"  PHASE 2: Running {n_experiments} experiments")
    print("=" * 60)

    raw_rows = []

    for exp_id in range(n_experiments):
        seed = RANDOM_SEED + exp_id
        source_tickers, target_ticker = sample_stocks(
            valid_tickers, n_source=10, random_state=seed,
        )
        print(f"\n--- Experiment {exp_id + 1}/{n_experiments} "
              f"(seed={seed}, target={target_ticker}) ---")

        for lag in LAGS:
            for order in ORDERS:
                m_d, m_t, risk = _run_single_path_experiment(
                    path_cache, source_tickers, target_ticker, lag, order,
                )
                raw_rows.append({
                    "exp_id": exp_id, "method": "Path",
                    "lag": lag, "order": order,
                    "direct_mse": m_d["mse"], "direct_r2": m_d["r2"],
                    "direct_corr": m_d["corr"],
                    "transfer_mse": m_t["mse"], "transfer_r2": m_t["r2"],
                    "transfer_corr": m_t["corr"],
                    "transfer_risk": risk,
                    "alpha_direct": ALPHA_SOURCE,
                    "alpha_transfer": ALPHA_TRANSFER,
                    "target": str(target_ticker),
                })

        for trunc in EFM_TRUNCS:
            result = _run_single_efm_experiment(
                efm_cache, source_tickers, target_ticker,
                trunc, lambda_grid,
            )

            if result[0] is not None:
                m_d, lam_d = result[0], result[1]
                raw_rows.append({
                    "exp_id": exp_id, "method": "EFM",
                    "lag": "-", "order": trunc,
                    "direct_mse": m_d["mse"], "direct_r2": m_d["r2"],
                    "direct_corr": m_d["corr"],
                    "transfer_mse": np.nan, "transfer_r2": np.nan,
                    "transfer_corr": np.nan,
                    "transfer_risk": np.nan,
                    "alpha_direct": ALPHA_SOURCE,
                    "efm_lambda_direct": lam_d.tolist(),
                    "target": str(target_ticker),
                })

            if result[2] is not None:
                m_t, lam_t, risk = result[2], result[3], result[4]
                raw_rows[-1]["transfer_mse"] = m_t["mse"]
                raw_rows[-1]["transfer_r2"] = m_t["r2"]
                raw_rows[-1]["transfer_corr"] = m_t["corr"]
                raw_rows[-1]["transfer_risk"] = risk
                raw_rows[-1]["alpha_transfer"] = ALPHA_TRANSFER
                raw_rows[-1]["efm_lambda_transfer"] = lam_t.tolist()

    df_raw = pd.DataFrame(raw_rows)
    return df_raw


# %%
def save_raw_csv(df_raw):
    """Save raw per-experiment results to data/benchmark_results.csv.

    Args:
        df_raw (pd.DataFrame): Raw results DataFrame from run_all_experiments().
    """
    out_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "benchmark_results.csv")
    df_raw.to_csv(path, index=False)
    print(f"Raw results saved to {path}")


def compute_and_save_summary(df_raw):
    """Aggregate raw results by (method, lag/order) and save summary CSV.

    Computes mean metrics across experiments for both Path and EFM methods,
    reports average lambda values for EFM, and prints a formatted summary table.

    Args:
        df_raw (pd.DataFrame): Raw results from run_all_experiments().

    Returns:
        pd.DataFrame: Aggregated summary DataFrame.
    """
    out_dir = os.path.join(BASE_DIR, "data")

    summary_rows = []

    df_path = df_raw[df_raw["method"] == "Path"]
    for (lag, order), grp in df_path.groupby(["lag", "order"]):
        summary_rows.append({
            "method": "Path", "lag": lag, "order": order,
            "direct_mse": grp["direct_mse"].mean(),
            "direct_r2": grp["direct_r2"].mean(),
            "direct_corr": grp["direct_corr"].mean(),
            "transfer_mse": grp["transfer_mse"].mean(),
            "transfer_r2": grp["transfer_r2"].mean(),
            "transfer_corr": grp["transfer_corr"].mean(),
            "transfer_risk": grp["transfer_risk"].mean(),
            "alpha_direct": ALPHA_SOURCE,
            "alpha_transfer": ALPHA_TRANSFER,
            "n_experiments": len(grp),
        })

    df_efm = df_raw[df_raw["method"] == "EFM"]
    for trunc, grp in df_efm.groupby("order"):
        avg_lam_d = np.mean(grp["efm_lambda_direct"].tolist(), axis=0).tolist()
        if "efm_lambda_transfer" in grp.columns and grp["efm_lambda_transfer"].notna().any():
            avg_lam_t = np.mean(
                [x for x in grp["efm_lambda_transfer"].dropna().tolist()],
                axis=0,
            ).tolist()
        else:
            avg_lam_t = None

        summary_rows.append({
            "method": "EFM", "lag": "-", "order": trunc,
            "direct_mse": grp["direct_mse"].mean(),
            "direct_r2": grp["direct_r2"].mean(),
            "direct_corr": grp["direct_corr"].mean(),
            "transfer_mse": grp["transfer_mse"].mean(),
            "transfer_r2": grp["transfer_r2"].mean(),
            "transfer_corr": grp["transfer_corr"].mean(),
            "transfer_risk": grp["transfer_risk"].mean(),
            "avg_lambda_direct": avg_lam_d,
            "avg_lambda_transfer": avg_lam_t,
            "n_experiments": len(grp),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(out_dir, "efm_benchmark_results.csv"), index=False)

    print("\n" + "=" * 80)
    print("  AVERAGED RESULTS")
    print("=" * 80)
    print(tabulate(df_summary.round(4), headers="keys", tablefmt="grid", showindex=False))

    return df_summary


def plot_distributions(df_raw):
    """Generate histograms comparing Direct vs Transfer MSE and R² distributions.

    Two rows (Path, EFM) × two columns (MSE, R²).  Vertical dashed lines
    mark the mean of each distribution.  Saved to data/distributions_mse.png.

    Args:
        df_raw (pd.DataFrame): Raw results from run_all_experiments().
    """
    out_dir = os.path.join(BASE_DIR, "data")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = [("Path", "Path Signatures"), ("EFM", "EFM Signatures")]
    metrics = [("mse", "MSE"), ("r2", r"$R^2$")]

    for i, (method_key, method_label) in enumerate(methods):
        for j, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[i, j]
            df_m = df_raw[df_raw["method"] == method_key]

            d_vals = df_m[f"direct_{metric_key}"].dropna().values
            t_vals = df_m[f"transfer_{metric_key}"].dropna().values

            ax.hist(d_vals, bins=15, alpha=0.6, label="Direct", color="steelblue")
            ax.hist(t_vals, bins=15, alpha=0.6, label="Transfer", color="coral")

            if len(d_vals) > 0:
                ax.axvline(np.mean(d_vals), color="steelblue", linestyle="--",
                           linewidth=1.5, label=f"Direct mean={np.mean(d_vals):.3f}")
            if len(t_vals) > 0:
                ax.axvline(np.mean(t_vals), color="coral", linestyle="--",
                           linewidth=1.5, label=f"Transfer mean={np.mean(t_vals):.3f}")

            ax.set_xlabel(metric_label)
            ax.set_ylabel("Count")
            ax.set_title(f"{method_label} — {metric_label} Distribution")
            ax.legend(fontsize=8)

    plt.suptitle(f"Distributions across {N_EXPERIMENTS} experiments", fontsize=13, y=1.02)
    plt.tight_layout()

    path_mse = os.path.join(out_dir, "distributions_mse.png")
    path_r2 = os.path.join(out_dir, "distributions_r2.png")

    fig.savefig(path_mse, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path_mse}")
    plt.close(fig)


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-experiments", type=int, default=N_EXPERIMENTS)
    parser.add_argument("--n-source", type=int, default=10)
    args = parser.parse_args()

    N_EXPERIMENTS = args.n_experiments

    # Load cached data (download was done previously)
    print("Loading data...")
    all_data = download_all_data(skip_download=True)
    valid_tickers = get_valid_tickers(all_data)

    if len(valid_tickers) < args.n_source + 1:
        print(f"Need at least {args.n_source + 1} tickers, got {len(valid_tickers)}")
        sys.exit(1)

    # Run the two-phase benchmark
    df_raw = run_all_experiments(all_data, valid_tickers, n_experiments=N_EXPERIMENTS)

    # Save and visualize
    save_raw_csv(df_raw)
    df_summary = compute_and_save_summary(df_raw)
    plot_distributions(df_raw)

    print("\nDone!")
