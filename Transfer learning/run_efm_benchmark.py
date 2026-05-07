import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "signature"))

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.config import (
    TRAIN_END, LAMBDA_DIRECT, LAMBDA_SOURCE, LAMBDA_TRANSFER,
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
    compute_efm_features, optimize_lambda, optimize_lambda_basket, WARMUP_DAYS,
)


# %%
def standardize(X_tr, y_tr, X_te, y_te):
    mu_x, sd_x = np.mean(X_tr, axis=0), np.std(X_tr, axis=0) + 1e-10
    mu_y, sd_y = np.mean(y_tr), np.std(y_tr) + 1e-10
    return (
        (X_tr - mu_x) / sd_x, (y_tr - mu_y) / sd_y,
        (X_te - mu_x) / sd_x, (y_te - mu_y) / sd_y,
    )


def split_train_test(feat_dates, features, targets):
    ts = pd.Timestamp(TRAIN_END)
    train_mask = np.array([d <= ts for d in feat_dates])
    test_mask = ~train_mask
    return features[train_mask], targets[train_mask], features[test_mask], targets[test_mask]


# %%
def run_efm_direct(all_data, target_ticker, truncs=(2, 3, 4)):
    dates, lp, lv, lr = prepare_stock_data(all_data[target_ticker])

    all_results = {}
    for trunc in truncs:
        print(f"  EFM Direct: optimizing lambda for trunc={trunc}...")
        lam, val_loss = optimize_lambda(lp, lv, dates, trunc=trunc,
                                         ridge_alpha=LAMBDA_DIRECT)
        print(f"    optimized lambda={np.round(lam, 4)}, val MSE={val_loss:.4f}")

        feats, tgts, fd = compute_efm_features(lp, lv, dates, lam, trunc)
        X_tr, y_tr, X_te, y_te = split_train_test(fd, feats, tgts)
        X_tr_n, y_tr_n, X_te_n, y_te_n = standardize(X_tr, y_tr, X_te, y_te)

        theta = ridge_solve(X_tr_n, y_tr_n, LAMBDA_DIRECT)
        y_pred = predict(X_te_n, theta)
        metrics = compute_all_metrics(y_te_n, y_pred)
        print(f"    test MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}, Corr={metrics['corr']:.4f}")

        all_results[trunc] = {"lam": lam, "metrics": metrics}

    return all_results


# %%
def run_efm_transfer(all_data, source_tickers, target_ticker, truncs=(2, 3, 4)):
    dates_t, lp_t, lv_t, lr_t = prepare_stock_data(all_data[target_ticker])

    source_stocks_data = []
    for ticker in source_tickers:
        dates_s, lp_s, lv_s, _ = prepare_stock_data(all_data[ticker])
        source_stocks_data.append((lp_s, lv_s, dates_s))

    all_results = {}
    for trunc in truncs:
        print(f"  EFM Transfer: optimizing lambda on basket for trunc={trunc}...")
        lam, val_loss = optimize_lambda_basket(
            source_stocks_data, trunc=trunc, ridge_alpha=LAMBDA_SOURCE,
        )
        print(f"    optimized basket lambda={np.round(lam, 4)}, CV MSE={val_loss:.4f}")

        source_X_list, source_y_list = [], []
        for lp_s, lv_s, dates_s in source_stocks_data:
            feats_s, tgts_s, fd_s = compute_efm_features(lp_s, lv_s, dates_s, lam, trunc)
            X_tr_s, y_tr_s, _, _ = split_train_test(fd_s, feats_s, tgts_s)
            source_X_list.append(X_tr_s)
            source_y_list.append(y_tr_s)

        X_source = np.vstack(source_X_list)
        y_source = np.concatenate(source_y_list)
        mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
        my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
        X_source_n = (X_source - mu_s) / sd_s
        y_source_n = (y_source - my_s) / sy_s

        theta_S = ridge_solve(X_source_n, y_source_n, LAMBDA_SOURCE)

        feats_t, tgts_t, fd_t = compute_efm_features(lp_t, lv_t, dates_t, lam, trunc)
        X_tr, y_tr, X_te, y_te = split_train_test(fd_t, feats_t, tgts_t)
        X_tr_n, y_tr_n, X_te_n, y_te_n = standardize(X_tr, y_tr, X_te, y_te)

        theta_T = transfer_ridge_solve(X_tr_n, y_tr_n, theta_S, LAMBDA_TRANSFER)
        y_pred = predict(X_te_n, theta_T)
        metrics = compute_all_metrics(y_te_n, y_pred)

        risk = compute_transfer_risk(X_source_n, y_source_n, X_tr_n, y_tr_n)

        print(f"    test MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}, "
              f"Corr={metrics['corr']:.4f}, risk={risk:.6f}")

        all_results[trunc] = {
            "lam": lam,
            "metrics": metrics,
            "transfer_risk": risk,
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
            for ticker in source_tickers:
                dates_s, lp_s, lv_s, lr_s = source_data[ticker]
                feats = compute_signature_features(lp_s, lv_s, lag, order)
                fd = dates_s[lag - 1:-1]
                tgts = lr_s[lag - 1:]
                X_tr, y_tr, _, _ = split_train_test(fd, feats, tgts)
                source_X_list.append(X_tr)
                source_y_list.append(y_tr)

            X_source = np.vstack(source_X_list)
            y_source = np.concatenate(source_y_list)
            mu_s, sd_s = np.mean(X_source, axis=0), np.std(X_source, axis=0) + 1e-10
            my_s, sy_s = np.mean(y_source), np.std(y_source) + 1e-10
            X_source_n = (X_source - mu_s) / sd_s
            y_source_n = (y_source - my_s) / sy_s

            feats_t = compute_signature_features(lp_t, lv_t, lag, order)
            fd_t = dates_t[lag - 1:-1]
            tgts_t = lr_t[lag - 1:]
            X_tr, y_tr, X_te, y_te = split_train_test(fd_t, feats_t, tgts_t)
            X_tr_n, y_tr_n, X_te_n, y_te_n = standardize(X_tr, y_tr, X_te, y_te)

            theta_d = ridge_solve(X_tr_n, y_tr_n, LAMBDA_DIRECT)
            y_pred_d = predict(X_te_n, theta_d)
            m_direct = compute_all_metrics(y_te_n, y_pred_d)

            theta_S = ridge_solve(X_source_n, y_source_n, LAMBDA_SOURCE)
            theta_T = transfer_ridge_solve(X_tr_n, y_tr_n, theta_S, LAMBDA_TRANSFER)
            y_pred_t = predict(X_te_n, theta_T)
            m_transfer = compute_all_metrics(y_te_n, y_pred_t)

            risk = compute_transfer_risk(X_source_n, y_source_n, X_tr_n, y_tr_n)

            results[(lag, order)] = {
                "direct": m_direct, "transfer": m_transfer,
                "transfer_risk": risk,
            }
            print(f"  Path L={lag} M={order}: Direct MSE={m_direct['mse']:.4f}, "
                  f"Transfer MSE={m_transfer['mse']:.4f}")

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
    print("\n" + "=" * 90)
    print("  EFM SIGNATURES: Optimized Parameters & Results")
    print("=" * 90)

    print("\n  --- Optimized Lambda (Direct Learning) ---")
    for trunc, r in sorted(efm_direct.items()):
        print(f"    trunc={trunc}: lambda={np.round(r['lam'], 4)}")

    print("\n  --- Optimized Lambda (Transfer Learning - Basket) ---")
    for trunc, r in sorted(efm_transfer.items()):
        print(f"    trunc={trunc}: basket lambda={np.round(r['lam'], 4)}, "
              f"transfer risk: {r['transfer_risk']:.6f}")

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
    print("\n" + "=" * 90)
    print("  PATH SIGNATURES: Results")
    print("=" * 90)

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
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-source", type=int, default=2)
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

    import json
    out_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)

    save = {
        "target": str(target_ticker),
        "sources": [str(t) for t in source_tickers],
        "seed": args.seed,
        "efm_direct": {
            str(trunc): {
                "lambda": r["lam"].tolist(),
                "metrics": r["metrics"],
            }
            for trunc, r in efm_direct.items()
        },
        "efm_transfer": {
            str(trunc): {
                "lambda_basket": r["lam"].tolist(),
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
            "optimized_lambda_direct": efm_direct[trunc]["lam"].tolist(),
            "optimized_lambda_transfer_basket": efm_transfer[trunc]["lam"].tolist(),
        })
    for (lag, order) in sorted(path_results.keys()):
        d = path_results[(lag, order)]["direct"]
        t = path_results[(lag, order)]["transfer"]
        rows.append({
            "method": "Path", "lag": lag, "order": order,
            "direct_mse": d["mse"], "direct_r2": d["r2"], "direct_corr": d["corr"],
            "transfer_mse": t["mse"], "transfer_r2": t["r2"], "transfer_corr": t["corr"],
            "transfer_risk": path_results[(lag, order)]["transfer_risk"],
            "optimized_lambda_direct": None,
            "optimized_lambda_transfer_basket": None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "efm_benchmark_results.csv"), index=False)

    import io
    buf = io.StringIO()
    buf.write(f"Target: {target_ticker}\n")
    buf.write(f"Sources: {source_tickers}\n")
    buf.write(f"Seed: {args.seed}\n\n")
    buf.write("=" * 90 + "\n")
    buf.write("  EFM SIGNATURES: Optimized Parameters & Results\n")
    buf.write("=" * 90 + "\n\n")
    buf.write("--- Optimized Lambda (Direct Learning) ---\n")
    for trunc, r in sorted(efm_direct.items()):
        buf.write(f"  trunc={trunc}: lambda={np.round(r['lam'], 4)}\n")
    buf.write("\n--- Optimized Lambda (Transfer Learning - Basket) ---\n")
    for trunc, r in sorted(efm_transfer.items()):
        buf.write(f"  trunc={trunc}: basket lambda={np.round(r['lam'], 4)}, "
                  f"transfer risk: {r['transfer_risk']:.6f}\n")
    buf.write("\n")
    efm_rows = []
    best_d = _find_best_per_metric(efm_direct, "metrics", ["mse", "r2", "corr"])
    best_t = _find_best_per_metric(efm_transfer, "metrics", ["mse", "r2", "corr"])
    for trunc in sorted(efm_direct.keys(), key=int):
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
    buf.write(tabulate(efm_rows, headers=["Trunc", "Direct MSE", "Direct R2", "Direct Corr",
                                           "Transfer MSE", "Transfer R2", "Transfer Corr"],
                        tablefmt="grid") + "\n")
    buf.write("* = best per column\n\n")
    buf.write("=" * 90 + "\n")
    buf.write("  PATH SIGNATURES: Results\n")
    buf.write("=" * 90 + "\n\n")
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
    buf.write(tabulate(path_rows, headers=["Lag", "Order", "Direct MSE", "Direct R2", "Direct Corr",
                                            "Transfer MSE", "Transfer R2", "Transfer Corr"],
                        tablefmt="grid") + "\n")
    buf.write("* = best per column\n")

    with open(os.path.join(out_dir, "efm_benchmark_results.txt"), "w") as f:
        f.write(buf.getvalue())

    print(f"\nResults saved to:")
    print(f"  {os.path.join(out_dir, 'efm_benchmark_results.json')}")
    print(f"  {os.path.join(out_dir, 'efm_benchmark_results.csv')}")
    print(f"  {os.path.join(out_dir, 'efm_benchmark_results.txt')}")
