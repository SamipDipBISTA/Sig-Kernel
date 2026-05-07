import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests
    _SESSION = curl_requests.Session(impersonate="safari15_5")
except ImportError:
    _SESSION = None

from .config import (
    TRAIN_START, TRAIN_END, TEST_END, DATA_DIR, CACHE_DIR, IT_TICKERS,
)


def _get_ticker(ticker):
    if _SESSION is not None:
        return yf.Ticker(ticker, session=_SESSION)
    return yf.Ticker(ticker)


def _download_single(ticker, start, end, max_retries=3):
    for attempt in range(max_retries):
        try:
            time.sleep(1 + attempt)
            t = _get_ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            if attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"    {ticker}: retry {attempt+1}/{max_retries} in {wait}s...")
                time.sleep(wait)
    return None


def download_all_data(tickers=None, start=TRAIN_START, end=TEST_END, force=False,
                      skip_download=False):
    if tickers is None:
        tickers = IT_TICKERS
    os.makedirs(CACHE_DIR, exist_ok=True)

    results = {}
    to_download = []

    for ticker in tickers:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(cache_path) and not force:
            try:
                df = pd.read_csv(cache_path, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
                df = df.sort_index()
                if not df.empty:
                    results[ticker] = df
                    continue
            except Exception:
                pass
        to_download.append(ticker)

    if to_download and not skip_download:
        print(f"Downloading {len(to_download)} tickers (cached: {len(results)})...")
        for i, ticker in enumerate(to_download):
            print(f"  [{i+1}/{len(to_download)}] {ticker}...", end=" ", flush=True)
            df = _download_single(ticker, start, end)
            if df is not None:
                cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
                if hasattr(df.index, "tz") and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.to_csv(cache_path)
                results[ticker] = df
                print(f"OK ({len(df)} rows)")
            else:
                print("FAILED")
    elif to_download:
        print(f"Skipping download for {len(to_download)} uncached tickers (cached: {len(results)})")

    print(f"Loaded: {len(results)}/{len(tickers)} tickers")
    return results


def get_valid_tickers(all_data, min_coverage=0.90):
    full_range = pd.bdate_range(start=TRAIN_START, end=TEST_END, freq="B")
    total_days = len(full_range)
    valid = []
    for ticker, df in all_data.items():
        if df.empty:
            continue
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
            df = df.copy()
            df.index = idx
            all_data[ticker] = df
        common = idx.intersection(full_range)
        coverage = len(common) / total_days
        if coverage >= min_coverage:
            valid.append(ticker)
    print(f"Valid tickers with >={min_coverage*100:.0f}% coverage: {len(valid)}/{len(all_data)}")
    return sorted(valid)


def sample_stocks(valid_tickers, n_source=10, random_state=None):
    rng = np.random.RandomState(random_state)
    n_source = min(n_source, len(valid_tickers) - 1)
    if n_source < 1:
        raise ValueError(f"Need at least 2 tickers, got {len(valid_tickers)}")
    n_total = n_source + 1
    selected = rng.choice(valid_tickers, size=n_total, replace=False)
    return list(selected[:n_source]), selected[n_source]


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


def get_train_test_indices(dates, n_prices):
    train_mask = np.array([d <= pd.Timestamp(TRAIN_END) for d in dates])
    return train_mask
