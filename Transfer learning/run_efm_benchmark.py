import sys
import os
import itertools
import json
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "signature"))

import numpy as np
import pandas as pd
from tabulate import tabulate

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
    compute_efm_features, optimize_params_single, optimize_params_basket, WARMUP_DAYS,
)

OPT_TRAIN_END = "2020-12-31"
OPT_VAL_END = TRAIN_END
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0]


def _make_lambda_grid():
    lam_time_vals = np.array([0.1, 1.0, 10.0])
    lam_fin_vals = np.array([0.1, 1.0, 10.0])
    return [
        np.array([lt, lp, lv])
        for lt, lp, lv in itertools.product(lam_time_vals, lam_fin_vals, lam_fin_vals)
    ]


def split_by_date(feat_dates, features, targets, split_date):
    ts = pd.Timestamp(split_date)
    mask = np.array([d <= ts for d in feat_dates])
    return features[mask], targets[mask], features[~mask], targets[~mask]


def _optimize_path_alpha_single(X_tr, y_tr, X_val, y_val, alpha_grid):
    mu_x, sd_x = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
    mu_y, sd_y = np.mean(y_tr), np.std(y_tr) + 1e-10
    X_tr_n = (X_tr - mu_x) / sd_x
    y_tr_n = (y_tr - mu_y) / sd_y
    X_val_n = (X_val - mu_x) / sd_x
    y_val_n = (y_val - mu_y) / sd_y

    best_alpha = None
    best_mse = np.inf
    for alpha in alpha_grid:
        theta = ridge_solve(X_tr_n, y_tr_n, alpha)
        y_pred = predict(X_val_n, theta)
        mse = float(np.mean((y_val_n - y_pred) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    return best_alpha, best_mse, (mu_x, sd_x, mu_y, sd_y)


# %%
def run_efm_direct(all_data, target_ticker, truncs=(2, 3, 4)):
    dates, lp, lv, lr = prepare_stock_data(all_data[target_ticker])
    lambda_grid = _make_lambda_grid()

    all_results = {}
    for trunc in truncs:
        print(f"\n  EFM Direct: optimizing params for trunc={trunc}...")
        best_lam, best_alpha, val_mse = optimize_params_single(
            lp, lv, dates, trunc=trunc,
            train_end=OPT_TRAIN_END, val_end=OPT_VAL_END,
            lambda_grid=lambda_grid, ridge_alphas=RIDGE_ALPHAS,
        )
        if best_lam is None:
            print(f"    SKIPPING trunc={trunc}: optimization failed")
            continue

        feats, tgts, fd = compute_efm_features(lp, lv, dates, best_lam, trunc)
        X_tr, y_tr, X_te, y_te = split_by_date(fd, feats, tgts, TRAIN_END)

        mu_x, sd_x = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
        mu_y, sd_y = np.mean(y_tr), np.std(y_tr) + 1e-10
        X_tr_n = (X_tr - mu_x) / sd_x
        y_tr_n = (y_tr - mu_y) / sd_y
        X_te_n = (X_te - mu_x) / sd_x
        y_te_n = (y_te - mu_y) / sd_y

        theta = ridge_solve(X_tr_n, y_tr_n, best_alpha)
        y_pred = predict(X_te_n, theta)
        metrics = compute_all_metrics(y_te_n, y_pred)
        print(f"    TEST: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}, Corr={metrics['corr']:.4f}")

        all_results[trunc] = {"lam": best_lam, "alpha": best_alpha, "metrics": metrics}

    return all_results


# %%
def run_efm_transfer(all_data, source_tickers, target_ticker, truncs=(2, 3, 4)):
    dates_t, lp_t, lv_t, lr_t = prepare_stock_data(all_data[target_ticker])

    source_stocks_data = []
    for ticker in source_tickers:
        dates_s, lp_s, lv_s, _ = prepare_stock_data(all_data[ticker])
        source_stocks_data.append((lp_s, lv_s, dates_s))

    lambda_grid = _make_lambda_grid()

    all_results = {}
    for trunc in truncs:
        print(f"\n  EFM Transfer: optimizing params on basket for trunc={trunc}...")
        best_lam, best_alpha, val_mse = optimize_params_basket(
            source_stocks_data, trunc=trunc,
            train_end=OPT_TRAIN_END, val_end=OPT_VAL_END,
            lambda_grid=lambda_grid, ridge_alphas=RIDGE_ALPHAS,
        )
        if best_lam is None:
            print(f"    SKIPPING trunc={trunc}: optimization failed")
            continue

        source_X_list, source_y_list = [], []
        for lp_s, lv_s, dates_s in source_stocks_data:
            feats_s, tgts_s, fd_s = compute_efm_features(lp_s, lv_s, dates_s, best_lam, trunc)
            X_tr_s, y_tr_s, _, _ = split_by_date(fd_s, feats_s, tgts_s, TRAIN_END)
            source_X_list.append(X_tr_s)
            source_y_list.append(y_tr_s)

        X_source = np.vstack(source_X_list)
        y_source = np.concatenate(source_y_list)

        mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
        my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
        X_source_n = (X_source - mu_s) / sd_s
        y_source_n = (y_source - my_s) / sy_s

        theta_S = ridge_solve(X_source_n, y_source_n, best_alpha)

        feats_t, tgts_t, fd_t = compute_efm_features(lp_t, lv_t, dates_t, best_lam, trunc)
        X_tr, y_tr, X_te, y_te = split_by_date(fd_t, feats_t, tgts_t, TRAIN_END)

        X_tr_n = (X_tr - mu_s) / sd_s
        y_tr_n = (y_tr - my_s) / sy_s
        X_te_n = (X_te - mu_s) / sd_s
        y_te_n = (y_te - my_s) / sy_s

        theta_T = transfer_ridge_solve(X_tr_n, y_tr_n, theta_S, ALPHA_TRANSFER)
        y_pred = predict(X_te_n, theta_T)
        metrics = compute_all_metrics(y_te_n, y_pred)

        risk = compute_transfer_risk(X_source, y_source, X_tr, y_tr)

        print(f"    TEST: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}, "
              f"Corr={metrics['corr']:.4f}, risk={risk:.6f}")

        all_results[trunc] = {
            "lam": best_lam, "alpha": best_alpha,
            "metrics": metrics, "transfer_risk": risk,
        }

    return all_results


# %%
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
            source_X_val_list, source_y_val_list = [], []
            for ticker in source_tickers:
                dates_s, lp_s, lv_s, lr_s = source_data[ticker]
                feats = compute_signature_features(lp_s, lv_s, lag, order)
                fd = dates_s[lag - 1:-1]
                tgts = lr_s[lag - 1:]
                X_tr_s, y_tr_s, X_te_s, _ = split_by_date(fd, feats, tgts, TRAIN_END)
                X_opt_tr, y_opt_tr, X_opt_val, y_opt_val = split_by_date(fd, feats, tgts, OPT_TRAIN_END)

                source_X_list.append(X_tr_s)
                source_y_list.append(y_tr_s)

            X_source = np.vstack(source_X_list)
            y_source = np.concatenate(source_y_list)

            mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
            my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
            X_source_n = (X_source - mu_s) / sd_s
            y_source_n = (y_source - my_s) / sy_s

            feats_t = compute_signature_features(lp_t, lv_t, lag, order)
            fd_t = dates_t[lag - 1:-1]
            tgts_t = lr_t[lag - 1:]
            X_tr, y_tr, X_te, y_te = split_by_date(fd_t, feats_t, tgts_t, TRAIN_END)

            X_opt_tr, y_opt_tr, X_opt_val, y_opt_val = split_by_date(fd_t, feats_t, tgts_t, OPT_TRAIN_END)

            best_alpha_d, _, _ = _optimize_path_alpha_single(
                X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, RIDGE_ALPHAS,
            )

            mu_t, sd_t = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
            my_t, sy_t = np.mean(y_tr), np.std(y_tr) + 1e-10
            X_tr_n = (X_tr - mu_t) / sd_t
            y_tr_n = (y_tr - my_t) / sy_t
            X_te_n = (X_te - mu_t) / sd_t
            y_te_n = (y_te - my_t) / sy_t

            theta_d = ridge_solve(X_tr_n, y_tr_n, best_alpha_d)
            y_pred_d = predict(X_te_n, theta_d)
            m_direct = compute_all_metrics(y_te_n, y_pred_d)

            X_tr_n2 = (X_tr - mu_s) / sd_s
            y_tr_n2 = (y_tr - my_s) / sy_s
            X_te_n2 = (X_te - mu_s) / sd_s
            y_te_n2 = (y_te - my_s) / sy_s

            X_opt_tr2 = (X_opt_tr - mu_s) / sd_s
            y_opt_tr2 = (y_opt_tr - my_s) / sy_s
            X_opt_val2 = (X_opt_val - mu_s) / sd_s
            y_opt_val2 = (y_opt_val - my_s) / sy_s

            theta_S_opt = ridge_solve(X_opt_tr2, y_opt_tr2, ALPHA_SOURCE)
            best_alpha_t = None
            best_val_mse_t = np.inf
            for alpha in RIDGE_ALPHAS:
                theta_T_opt = transfer_ridge_solve(X_opt_val2, y_opt_val2, theta_S_opt, alpha)
                y_pred_opt = predict(X_opt_val2, theta_T_opt)
                mse_opt = float(np.mean((y_opt_val2 - y_pred_opt) ** 2))
                if mse_opt < best_val_mse_t:
                    best_val_mse_t = mse_opt
                    best_alpha_t = alpha

            theta_S = ridge_solve(X_source_n, y_source_n, ALPHA_SOURCE)
            theta_T = transfer_ridge_solve(X_tr_n2, y_tr_n2, theta_S, best_alpha_t)
            y_pred_t = predict(X_te_n2, theta_T)
            m_transfer = compute_all_metrics(y_te_n2, y_pred_t)

            risk = compute_transfer_risk(X_source, y_source, X_tr, y_tr)

            results[(lag, order)] = {
                "direct": m_direct, "transfer": m_transfer,
                "transfer_risk": risk,
                "alpha_direct": best_alpha_d,
                "alpha_transfer": best_alpha_t,
            }
            print(f"  Path L={lag} M={order}: alpha_d={best_alpha_d}, alpha_t={best_alpha_t}, "
                  f"Direct MSE={m_direct['mse']:.4f}, Transfer MSE={m_transfer['mse']:.4f}")

    return results


# %%
def _find_best_per_metric(results, method, metric_keys):
    best = {}
    for key in metric_keys:
        if key == "mse":
            best_key = min(results, key=lambda k: results[k][method][key])
        else:
            best_key = max(results, key=lambda k: results[k][method][key])
        best[key] = best_key
    return best


def _fmt(val, is_best):
    s = f"{val:.4f}"
    return s + " *" if is_best else s


# %%
def print_efm_table(efm_direct, efm_transfer):
    print("\n" + "=" * 100)
    print("  EFM SIGNATURES: Optimized Parameters & Results")
    print("=" * 100)

    print("\n  --- Direct Learning ---")
    for trunc, r in sorted(efm_direct.items()):
        print(f"    trunc={trunc}: lambda={np.round(r['lam'], 4)}, ridge_alpha={r['alpha']}")

    print("\n  --- Transfer Learning (Basket) ---")
    for trunc, r in sorted(efm_transfer.items()):
        print(f"    trunc={trunc}: lambda={np.round(r['lam'], 4)}, "
              f"ridge_alpha={r['alpha']}, transfer_risk={r['transfer_risk']:.6f}")

    best_d = _find_best_per_metric(efm_direct, "metrics", ["mse", "r2", "corr"])
    best_t = _find_best_per_metric(efm_transfer, "metrics", ["mse", "r2", "corr"])

    header = ["Trunc", "Direct MSE", "Direct R2", "Direct Corr",
              "Transfer MSE", "Transfer R2", "Transfer Corr"]
    rows = []
    for trunc in sorted(efm_direct.keys()):
        d = efm_direct[trunc]["metrics"]
        t = efm_transfer[trunc]["metrics"]
        rows.append([
            str(trunc),
            _fmt(d["mse"], best_d["mse"] == trunc),
            _fmt(d["r2"], best_d["r2"] == trunc),
            _fmt(d["corr"], best_d["corr"] == trunc),
            _fmt(t["mse"], best_t["mse"] == trunc),
            _fmt(t["r2"], best_t["r2"] == trunc),
            _fmt(t["corr"], best_t["corr"] == trunc),
        ])

    print("\n  --- EFM Results Table ---")
    print(tabulate(rows, headers=header, tablefmt="grid"))
    print("  * = best per column (lowest MSE, highest R2, highest Corr)")


# %%
def print_path_table(path_results):
    print("\n" + "=" * 100)
    print("  PATH SIGNATURES: Results")
    print("=" * 100)

    best_d = _find_best_per_metric(path_results, "direct", ["mse", "r2", "corr"])
    best_t = _find_best_per_metric(path_results, "transfer", ["mse", "r2", "corr"])

    header = ["Lag", "Order", "Direct MSE", "Direct R2", "Direct Corr",
              "Transfer MSE", "Transfer R2", "Transfer Corr"]
    rows = []
    for (lag, order) in sorted(path_results.keys()):
        d = path_results[(lag, order)]["direct"]
        t = path_results[(lag, order)]["transfer"]
        rows.append([
            str(lag), str(order),
            _fmt(d["mse"], best_d["mse"] == (lag, order)),
            _fmt(d["r2"], best_d["r2"] == (lag, order)),
            _fmt(d["corr"], best_d["corr"] == (lag, order)),
            _fmt(t["mse"], best_t["mse"] == (lag, order)),
            _fmt(t["r2"], best_t["r2"] == (lag, order)),
            _fmt(t["corr"], best_t["corr"] == (lag, order)),
        ])

    print("\n  --- Path Signature Results Table ---")
    print(tabulate(rows, headers=header, tablefmt="grid"))
    print("  * = best per column (lowest MSE, highest R2, highest Corr)")


# %%
def save_results(efm_direct, efm_transfer, path_results, source_tickers, target_ticker, seed):
    out_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)

    save = {
        "target": str(target_ticker),
        "sources": [str(t) for t in source_tickers],
        "seed": seed,
        "train_end": TRAIN_END,
        "opt_train_end": OPT_TRAIN_END,
        "opt_val_end": OPT_VAL_END,
        "ridge_alpha_grid": RIDGE_ALPHAS,
        "efm_direct": {
            str(trunc): {
                "efm_lambda": r["lam"].tolist(),
                "ridge_alpha": r["alpha"],
                "metrics": r["metrics"],
            }
            for trunc, r in efm_direct.items()
        },
        "efm_transfer": {
            str(trunc): {
                "efm_lambda_basket": r["lam"].tolist(),
                "ridge_alpha": r["alpha"],
                "transfer_risk": float(r["transfer_risk"]),
                "metrics": r["metrics"],
            }
            for trunc, r in efm_transfer.items()
        },
        "path_signatures": {
            f"L={lag}_M={order}": {
                "direct": r["direct"],
                "transfer": r["transfer"],
                "transfer_risk": float(r["transfer_risk"]),
                "ridge_alpha_direct": r["alpha_direct"],
                "ridge_alpha_transfer": r["alpha_transfer"],
            }
            for (lag, order), r in path_results.items()
        },
    }
    with open(os.path.join(out_dir, "efm_benchmark_results.json"), "w") as f:
        json.dump(save, f, indent=2)

    rows = []
    for trunc in sorted(efm_direct.keys(), key=int):
        d = efm_direct[trunc]["metrics"]
        t = efm_transfer[trunc]["metrics"]
        rows.append({
            "method": "EFM", "lag": "-", "order": trunc,
            "direct_mse": d["mse"], "direct_r2": d["r2"], "direct_corr": d["corr"],
            "transfer_mse": t["mse"], "transfer_r2": t["r2"], "transfer_corr": t["corr"],
            "transfer_risk": efm_transfer[trunc]["transfer_risk"],
            "efm_lambda_direct": efm_direct[trunc]["lam"].tolist(),
            "ridge_alpha_direct": efm_direct[trunc]["alpha"],
            "efm_lambda_transfer_basket": efm_transfer[trunc]["lam"].tolist(),
            "ridge_alpha_transfer": efm_transfer[trunc]["alpha"],
        })
    for (lag, order) in sorted(path_results.keys()):
        d = path_results[(lag, order)]["direct"]
        t = path_results[(lag, order)]["transfer"]
        rows.append({
            "method": "Path", "lag": lag, "order": order,
            "direct_mse": d["mse"], "direct_r2": d["r2"], "direct_corr": d["corr"],
            "transfer_mse": t["mse"], "transfer_r2": t["r2"], "transfer_corr": t["corr"],
            "transfer_risk": path_results[(lag, order)]["transfer_risk"],
            "ridge_alpha_direct": path_results[(lag, order)]["alpha_direct"],
            "ridge_alpha_transfer": path_results[(lag, order)]["alpha_transfer"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "efm_benchmark_results.csv"), index=False)

    buf = io.StringIO()
    buf.write(f"Target: {target_ticker}\n")
    buf.write(f"Sources: {[str(t) for t in source_tickers]}\n")
    buf.write(f"Seed: {seed}\n")
    buf.write(f"Train end: {TRAIN_END}\n")
    buf.write(f"Opt train end: {OPT_TRAIN_END}\n")
    buf.write(f"Opt val end: {OPT_VAL_END}\n")
    buf.write(f"Ridge alpha grid: {RIDGE_ALPHAS}\n\n")

    buf.write("=" * 80 + "\n")
    buf.write("  EFM SIGNATURES\n")
    buf.write("=" * 80 + "\n\n")

    buf.write("--- Optimized Parameters ---\n")
    buf.write("Direct:\n")
    for trunc, r in sorted(efm_direct.items()):
        buf.write(f"  trunc={trunc}: efm_lambda={np.round(r['lam'], 4)}, ridge_alpha={r['alpha']}\n")
    buf.write("Transfer (basket):\n")
    for trunc, r in sorted(efm_transfer.items()):
        buf.write(f"  trunc={trunc}: efm_lambda={np.round(r['lam'], 4)}, "
                  f"ridge_alpha={r['alpha']}, transfer_risk={r['transfer_risk']:.6f}\n")

    best_d = _find_best_per_metric(efm_direct, "metrics", ["mse", "r2", "corr"])
    best_t = _find_best_per_metric(efm_transfer, "metrics", ["mse", "r2", "corr"])
    efm_rows = []
    for trunc in sorted(efm_direct.keys()):
        d = efm_direct[trunc]["metrics"]
        t = efm_transfer[trunc]["metrics"]
        efm_rows.append([
            str(trunc),
            _fmt(d["mse"], best_d["mse"] == trunc),
            _fmt(d["r2"], best_d["r2"] == trunc),
            _fmt(d["corr"], best_d["corr"] == trunc),
            _fmt(t["mse"], best_t["mse"] == trunc),
            _fmt(t["r2"], best_t["r2"] == trunc),
            _fmt(t["corr"], best_t["corr"] == trunc),
        ])
    buf.write("\n--- EFM Results Table ---\n")
    buf.write(tabulate(efm_rows,
                       headers=["Trunc", "Direct MSE", "Direct R2", "Direct Corr",
                                "Transfer MSE", "Transfer R2", "Transfer Corr"],
                       tablefmt="grid") + "\n")
    buf.write("* = best per column (lowest MSE, highest R2, highest Corr)\n")

    buf.write("\n" + "=" * 80 + "\n")
    buf.write("  PATH SIGNATURES\n")
    buf.write("=" * 80 + "\n\n")

    best_d = _find_best_per_metric(path_results, "direct", ["mse", "r2", "corr"])
    best_t = _find_best_per_metric(path_results, "transfer", ["mse", "r2", "corr"])
    path_rows = []
    for (lag, order) in sorted(path_results.keys()):
        d = path_results[(lag, order)]["direct"]
        t = path_results[(lag, order)]["transfer"]
        path_rows.append([
            str(lag), str(order),
            _fmt(d["mse"], best_d["mse"] == (lag, order)),
            _fmt(d["r2"], best_d["r2"] == (lag, order)),
            _fmt(d["corr"], best_d["corr"] == (lag, order)),
            _fmt(t["mse"], best_t["mse"] == (lag, order)),
            _fmt(t["r2"], best_t["r2"] == (lag, order)),
            _fmt(t["corr"], best_t["corr"] == (lag, order)),
        ])
    buf.write("--- Path Signature Results Table ---\n")
    buf.write(tabulate(path_rows,
                       headers=["Lag", "Order", "Direct MSE", "Direct R2", "Direct Corr",
                                "Transfer MSE", "Transfer R2", "Transfer Corr"],
                       tablefmt="grid") + "\n")
    buf.write("* = best per column (lowest MSE, highest R2, highest Corr)\n")

    buf.write("\n--- Path Signature Optimized Ridge Alphas ---\n")
    for (lag, order) in sorted(path_results.keys()):
        r = path_results[(lag, order)]
        buf.write(f"  L={lag} M={order}: alpha_direct={r['alpha_direct']}, alpha_transfer={r['alpha_transfer']}\n")

    with open(os.path.join(out_dir, "efm_benchmark_results.txt"), "w") as f:
        f.write(buf.getvalue())

    print(f"\nResults saved to:")
    print(f"  {os.path.join(out_dir, 'efm_benchmark_results.json')}")
    print(f"  {os.path.join(out_dir, 'efm_benchmark_results.csv')}")
    print(f"  {os.path.join(out_dir, 'efm_benchmark_results.txt')}")


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-source", type=int, default=10)
    parser.add_argument("--efm-truncs", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print("Loading data...")
    all_data = download_all_data(skip_download=True)
    valid_tickers = get_valid_tickers(all_data)

    n_source = min(args.n_source, len(valid_tickers) - 1)
    if len(valid_tickers) < 2:
        print(f"Need at least 2 tickers, got {len(valid_tickers)}")
        sys.exit(1)

    source_tickers, target_ticker = sample_stocks(
        valid_tickers, n_source=n_source, random_state=args.seed,
    )
    print(f"Target: {target_ticker}, Sources: {source_tickers}")

    print("\n" + "=" * 60)
    print("  EFM SIGNATURES")
    print("=" * 60)

    efm_direct = run_efm_direct(all_data, target_ticker, truncs=args.efm_truncs)
    efm_transfer = run_efm_transfer(all_data, source_tickers, target_ticker,
                                     truncs=args.efm_truncs)

    print("\n" + "=" * 60)
    print("  PATH SIGNATURES")
    print("=" * 60)

    path_results = run_path_signatures(all_data, source_tickers, target_ticker)

    print_efm_table(efm_direct, efm_transfer)
    print_path_table(path_results)
    save_results(efm_direct, efm_transfer, path_results, source_tickers, target_ticker, args.seed)
