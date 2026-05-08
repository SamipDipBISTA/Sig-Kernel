import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate

from .config import (
    LAGS, ORDERS, ALPHA_SOURCE, ALPHA_TRANSFER,
    N_EXPERIMENTS, N_SOURCE, RANDOM_SEED, TRAIN_END, BASE_DIR,
)
from .data_utils import sample_stocks
from .signature_utils import compute_signature_features
from .models import ridge_solve, transfer_ridge_solve, predict
from .transfer_risk import compute_transfer_risk
from .evaluation import compute_all_metrics


# %%
def precompute_features(all_data, lags=None, orders=None):
    if lags is None:
        lags = LAGS
    if orders is None:
        orders = ORDERS

    cache = {}
    for ticker, df in tqdm(all_data.items(), desc="Precomputing features"):
        from .data_utils import prepare_stock_data
        dates, log_p, log_v, log_ret = prepare_stock_data(df)

        cache[ticker] = {
            "raw": {
                "dates": dates,
                "log_returns": log_ret,
            },
            "features": {},
        }

        for lag in lags:
            for order in orders:
                feats = compute_signature_features(log_p, log_v, lag, order)
                cache[ticker]["features"][(lag, order)] = feats

    return cache


# %%
def _get_features_and_targets(cache, ticker, lag, order):
    features = cache[ticker]["features"][(lag, order)]
    dates = cache[ticker]["raw"]["dates"]
    log_ret = cache[ticker]["raw"]["log_returns"]
    feat_dates = dates[lag - 1 : -1]
    targets = log_ret[lag - 1 :]
    return features, targets, feat_dates


def _split_by_date(feat_dates, features, targets):
    train_end_ts = pd.Timestamp(TRAIN_END)
    train_mask = np.array([d <= train_end_ts for d in feat_dates])
    test_mask = ~train_mask
    return (
        features[train_mask], targets[train_mask],
        features[test_mask], targets[test_mask],
    )


# %%
def run_single_experiment(cache, valid_tickers, lag, order, exp_id,
                          standardization="dataset"):
    seed = RANDOM_SEED + exp_id
    source_tickers, target_ticker = sample_stocks(
        valid_tickers, n_source=N_SOURCE, random_state=seed
    )

    source_X_list, source_y_list = [], []

    for ticker in source_tickers:
        feats, tgts, feat_dates = _get_features_and_targets(
            cache, ticker, lag, order
        )
        X_tr, y_tr, _, _ = _split_by_date(feat_dates, feats, tgts)

        if standardization == "per_stock":
            mu = np.mean(X_tr, axis=0)
            sd = np.std(X_tr, axis=0) + 1e-10
            my = np.mean(y_tr)
            sy = np.std(y_tr) + 1e-10
            X_tr = (X_tr - mu) / sd
            y_tr = (y_tr - my) / sy

        source_X_list.append(X_tr)
        source_y_list.append(y_tr)

    X_source = np.vstack(source_X_list)
    y_source = np.concatenate(source_y_list)

    if standardization == "dataset":
        mu_s = np.mean(X_source, axis=0)
        sd_s = np.std(X_source, axis=0) + 1e-10
        my_s = np.mean(y_source)
        sy_s = np.std(y_source) + 1e-10
        X_source = (X_source - mu_s) / sd_s
        y_source = (y_source - my_s) / sy_s

    feats_t, tgts_t, feat_dates_t = _get_features_and_targets(
        cache, target_ticker, lag, order
    )
    X_t_tr, y_t_tr, X_t_te, y_t_te = _split_by_date(feat_dates_t, feats_t, tgts_t)

    mu_t = np.mean(X_t_tr, axis=0)
    sd_t = np.std(X_t_tr, axis=0) + 1e-10
    my_t = np.mean(y_t_tr)
    sy_t = np.std(y_t_tr) + 1e-10
    X_t_tr_n = (X_t_tr - mu_t) / sd_t
    y_t_tr_n = (y_t_tr - my_t) / sy_t
    X_t_te_n = (X_t_te - mu_t) / sd_t
    y_t_te_n = (y_t_te - my_t) / sy_t

    theta_direct = ridge_solve(X_t_tr_n, y_t_tr_n, ALPHA_SOURCE)
    y_pred_direct = predict(X_t_te_n, theta_direct)

    theta_S = ridge_solve(X_source, y_source, ALPHA_SOURCE)
    theta_T = transfer_ridge_solve(X_t_tr_n, y_t_tr_n, theta_S, ALPHA_TRANSFER)
    y_pred_transfer = predict(X_t_te_n, theta_T)

    metrics_direct = compute_all_metrics(y_t_te_n, y_pred_direct)
    metrics_transfer = compute_all_metrics(y_t_te_n, y_pred_transfer)

    risk = compute_transfer_risk(X_source, y_source, X_t_tr_n, y_t_tr_n)

    return {
        "direct": metrics_direct,
        "transfer": metrics_transfer,
        "transfer_risk": risk,
        "source_tickers": source_tickers,
        "target_ticker": target_ticker,
    }


# %%
def run_table1(cache, valid_tickers, n_experiments=None, standardization="dataset"):
    if n_experiments is None:
        n_experiments = N_EXPERIMENTS

    results = {}
    for lag in LAGS:
        for order in ORDERS:
            print(f"\n--- Lag={lag}, Order={order} ---")
            all_direct = {"mse": [], "r2": [], "corr": []}
            all_transfer = {"mse": [], "r2": [], "corr": []}

            for exp_id in tqdm(range(n_experiments), desc=f"L={lag} M={order}"):
                res = run_single_experiment(
                    cache, valid_tickers, lag, order, exp_id, standardization
                )
                for m in ["mse", "r2", "corr"]:
                    all_direct[m].append(res["direct"][m])
                    all_transfer[m].append(res["transfer"][m])

            results[(lag, order)] = {
                "direct": {k: np.mean(v) for k, v in all_direct.items()},
                "transfer": {k: np.mean(v) for k, v in all_transfer.items()},
                "direct_all": all_direct,
                "transfer_all": all_transfer,
            }

            d = results[(lag, order)]
            print(
                f"  Direct  -> MSE={d['direct']['mse']:.2f}, "
                f"R2={d['direct']['r2']:.2f}, Corr={d['direct']['corr']:.2f}"
            )
            print(
                f"  Transfer-> MSE={d['transfer']['mse']:.2f}, "
                f"R2={d['transfer']['r2']:.2f}, Corr={d['transfer']['corr']:.2f}"
            )

    return results


# %%
def run_table2(cache, valid_tickers, lag=5, order=2,
               n_experiments=None, standardization="dataset"):
    if n_experiments is None:
        n_experiments = N_EXPERIMENTS

    risks, mses, r2s, corrs = [], [], [], []

    for exp_id in tqdm(range(n_experiments), desc=f"Table2 L={lag} M={order}"):
        res = run_single_experiment(
            cache, valid_tickers, lag, order, exp_id, standardization
        )
        risks.append(res["transfer_risk"])
        mses.append(res["transfer"]["mse"])
        r2s.append(-res["transfer"]["r2"])
        corrs.append(-res["transfer"]["corr"])

    data = np.array([mses, r2s, corrs, risks])
    labels = ["MSE", r"$-R^2$", r"$-Corr$", "Transfer Risk"]
    corr_matrix = np.corrcoef(data)

    print("\n=== Table 2: Correlation Matrix (L=5, M=2) ===")
    header = [""] + labels
    rows = []
    for i, lbl in enumerate(labels):
        rows.append([lbl] + [f"{corr_matrix[i, j]:.3f}" for j in range(4)])
    print(tabulate(rows, headers=header, tablefmt="grid"))

    return {
        "corr_matrix": corr_matrix,
        "labels": labels,
        "risks": risks,
        "mses": mses,
        "r2s": r2s,
        "corrs": corrs,
    }


# %%
def save_results(results, filename="results.pkl"):
    path = os.path.join(BASE_DIR, "data", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {path}")


def load_results(filename="results.pkl"):
    path = os.path.join(BASE_DIR, "data", filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# %%
def plot_table1_table(results):
    header = ["Lag", "Order", "Direct MSE", "Transfer MSE",
              "Direct R2", "Transfer R2", "Direct Corr", "Transfer Corr"]
    rows = []
    for lag in LAGS:
        for order in ORDERS:
            d = results[(lag, order)]
            rows.append([
                lag, order,
                f"{d['direct']['mse']:.2f}", f"{d['transfer']['mse']:.2f}",
                f"{d['direct']['r2']:.2f}", f"{d['transfer']['r2']:.2f}",
                f"{d['direct']['corr']:.2f}", f"{d['transfer']['corr']:.2f}",
            ])

    print("\n=== Table 1: Prediction Performance ===")
    print(tabulate(rows, headers=header, tablefmt="grid"))
    return rows


# %%
def plot_table1_heatmap(results):
    metrics = ["mse", "r2", "corr"]
    metric_labels = ["MSE", r"$R^2$", "Corr"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    lag_labels = [str(l) for l in LAGS]
    order_labels = [str(o) for o in ORDERS]

    for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for row, method in enumerate(["direct", "transfer"]):
            ax = axes[row, col]
            mat = np.zeros((len(LAGS), len(ORDERS)))
            for i, lag in enumerate(LAGS):
                for j, order in enumerate(ORDERS):
                    mat[i, j] = results[(lag, order)][method][metric]

            sns.heatmap(
                mat, annot=True, fmt=".2f", cmap="RdYlGn" if metric != "mse" else "RdYlGn_r",
                xticklabels=order_labels, yticklabels=lag_labels,
                ax=ax, vmin=None, vmax=None,
            )
            ax.set_title(f"{method.capitalize()} - {mlabel}")
            ax.set_xlabel("Order M")
            ax.set_ylabel("Lag L")

    plt.suptitle("Table 1: Direct vs Transfer Learning Performance", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "data", "table1_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# %%
def plot_table2_heatmap(table2_results):
    corr_matrix = table2_results["corr_matrix"]
    labels = ["MSE", r"$-R^2$", r"$-Corr$", "Transfer Risk"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
        xticklabels=labels, yticklabels=labels,
        ax=ax, vmin=-1, vmax=1, center=0,
        annot_kws={"size": 14},
    )
    ax.set_title("Table 2: Correlation Matrix\n(L=5, M=2, Transfer Learning Metrics)")
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "data", "table2_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# %%
def plot_direct_vs_transfer_bars(results):
    metrics = ["mse", "r2", "corr"]
    metric_labels = ["MSE", r"$R^2$", "Correlation"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(LAGS) * len(ORDERS))
    width = 0.35

    for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        direct_vals = []
        transfer_vals = []
        tick_labels = []
        for lag in LAGS:
            for order in ORDERS:
                d = results[(lag, order)]
                direct_vals.append(d["direct"][metric])
                transfer_vals.append(d["transfer"][metric])
                tick_labels.append(f"L={lag}\nM={order}")

        axes[col].bar(x - width / 2, direct_vals, width, label="Direct", color="steelblue")
        axes[col].bar(x + width / 2, transfer_vals, width, label="Transfer", color="coral")
        axes[col].set_xticks(x)
        axes[col].set_xticklabels(tick_labels, fontsize=8)
        axes[col].set_ylabel(mlabel)
        axes[col].set_title(f"{mlabel}: Direct vs Transfer")
        axes[col].legend()
        axes[col].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.suptitle("Table 1: Direct vs Transfer Learning Across All (Lag, Order)", fontsize=13)
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "data", "direct_vs_transfer_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# %%
def plot_transfer_risk_scatter(table2_results):
    risks = table2_results["risks"]
    mses = table2_results["mses"]
    r2s = table2_results["r2s"]
    corrs = table2_results["corrs"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        (risks, mses, "Transfer Risk", "MSE"),
        (risks, r2s, "Transfer Risk", r"$-R^2$"),
        (risks, corrs, "Transfer Risk", r"$-Corr$"),
    ]

    for ax, (x, y, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(x, y, alpha=0.5, s=20, color="steelblue")
        r = np.corrcoef(x, y)[0, 1]
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, label=f"r={r:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")
        ax.legend(fontsize=10)

    plt.suptitle("Transfer Risk vs Prediction Metrics (L=5, M=2, n=200)", fontsize=13)
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "data", "transfer_risk_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# %%
def plot_metric_distributions(results, lag=5, order=2):
    d = results[(lag, order)]
    metrics = ["mse", "r2", "corr"]
    labels = ["MSE", r"$R^2$", "Correlation"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, (metric, label) in enumerate(zip(metrics, labels)):
        d_vals = d["direct_all"][metric]
        t_vals = d["transfer_all"][metric]
        axes[col].hist(d_vals, bins=30, alpha=0.6, label="Direct", color="steelblue")
        axes[col].hist(t_vals, bins=30, alpha=0.6, label="Transfer", color="coral")
        axes[col].set_xlabel(label)
        axes[col].set_ylabel("Count")
        axes[col].set_title(f"{label} Distribution (L={lag}, M={order})")
        axes[col].legend()

    plt.suptitle("Metric Distributions Across 200 Experiments", fontsize=13)
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "data", "metric_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# %%
def run_full_reproduction(cache, valid_tickers, n_experiments=None,
                          standardization="dataset"):
    print("=" * 60)
    print("  Section 4 Reproduction: Transfer Learning for Stock")
    print("  Return Prediction using Signatures")
    print("=" * 60)

    print("\n>>> Running Table 1 experiments...")
    table1_results = run_table1(
        cache, valid_tickers, n_experiments=n_experiments,
        standardization=standardization,
    )

    print("\n>>> Running Table 2 experiments...")
    table2_results = run_table2(
        cache, valid_tickers, lag=5, order=2,
        n_experiments=n_experiments, standardization=standardization,
    )

    all_results = {"table1": table1_results, "table2": table2_results}
    save_results(all_results)

    print("\n>>> Generating plots...")
    plot_table1_table(table1_results)
    plot_table1_heatmap(table1_results)
    plot_direct_vs_transfer_bars(table1_results)
    plot_table2_heatmap(table2_results)
    plot_transfer_risk_scatter(table2_results)
    plot_metric_distributions(table1_results)

    print("\nDone!")
    return all_results
