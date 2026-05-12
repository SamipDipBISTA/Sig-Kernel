"""
Data downloading and preprocessing utilities for the transfer learning experiment.

Handles Yahoo Finance data acquisition via yfinance (with optional curl_cffi
session for anti-bot evasion), caching to disk, ticker validation by business-day
coverage, random stock sampling, and conversion of raw OHLCV data to log-prices,
log-volumes, and log-returns used as features/targets.
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests
    # Use impersonated browser session to reduce rate-limiting (optional dep)
    _SESSION = curl_requests.Session(impersonate="safari15_5")
except ImportError:
    _SESSION = None

from .config import (
    TRAIN_START, TRAIN_END, TEST_END, DATA_DIR, CACHE_DIR, IT_TICKERS,
)


def _get_ticker(ticker):
    """Create a yfinance Ticker object, optionally with a curl_cffi session.

    Args:
        ticker (str): Stock ticker symbol (e.g., "AAPL").

    Returns:
        yf.Ticker: Yahoo Finance Ticker instance.
    """
    if _SESSION is not None:
        return yf.Ticker(ticker, session=_SESSION)
    return yf.Ticker(ticker)


def _download_single(ticker, start, end, max_retries=3):
    """Download historical data for a single ticker with exponential-backoff retries.

    Args:
        ticker (str): Stock ticker symbol.
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        max_retries (int): Maximum download attempts.

    Returns:
        pd.DataFrame or None: OHLCV DataFrame on success, None on failure.
    """
    for attempt in range(max_retries):
        try:
            time.sleep(1 + attempt)  # gentle backoff before each attempt
            t = _get_ticker(ticker)
            df = t.history(start=start, end=end, auto_adjust=True)
            if df is not None and not df.empty:
                return df
        except Exception:
            if attempt < max_retries - 1:
                # exponential backoff: 5s, 10s, 20s, ...
                wait = 5 * (2 ** attempt)
                print(f"    {ticker}: retry {attempt+1}/{max_retries} in {wait}s...")
                time.sleep(wait)
    return None


def download_all_data(tickers=None, start=TRAIN_START, end=TEST_END, force=False,
                      skip_download=False):
    """Download or load from cache historical data for all requested tickers.

    First checks the local cache directory. Tickers not found are downloaded
    via yfinance (unless skip_download=True).

    Args:
        tickers (list[str] or None): Ticker symbols. Defaults to IT_TICKERS.
        start (str): Start date for download.
        end (str): End date for download.
        force (bool): If True, re-download even if cache exists.
        skip_download (bool): If True, skip downloading uncached tickers.

    Returns:
        dict[str, pd.DataFrame]: Mapping from ticker to OHLCV DataFrame.
    """
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
    """Filter tickers with sufficient business-day coverage over the full date range.

    A ticker is valid if its trading days intersect at least `min_coverage`
    fraction of all NYSE business days in [TRAIN_START, TEST_END].

    Args:
        all_data (dict[str, pd.DataFrame]): Ticker -> DataFrame mapping.
        min_coverage (float): Minimum fraction of business days required.

    Returns:
        list[str]: Sorted list of valid ticker symbols.
    """
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
    """Randomly sample n_source source stocks plus 1 target stock without replacement.

    Used to form one transfer-learning problem instance (paper Section 4.2).

    Args:
        valid_tickers (list[str]): Pool of valid tickers to sample from.
        n_source (int): Number of source stocks.
        random_state (int or None): Seed for reproducible sampling.

    Returns:
        tuple[list[str], str]: (list of source tickers, target ticker).
    """
    rng = np.random.RandomState(random_state)
    n_source = min(n_source, len(valid_tickers) - 1)
    if n_source < 1:
        raise ValueError(f"Need at least 2 tickers, got {len(valid_tickers)}")
    n_total = n_source + 1
    selected = rng.choice(valid_tickers, size=n_total, replace=False)
    return list(selected[:n_source]), selected[n_source]


def prepare_stock_data(df):
    """Convert a raw yfinance DataFrame to log-prices, log-volumes, and log-returns.

    Uses Adj Close when available (falls back to Close), filters non-positive
    prices/volumes, and applies log transforms.  Volume uses log(v+1) to handle
    zero-volume days gracefully.

    Args:
        df (pd.DataFrame): yfinance OHLCV DataFrame (must have Volume column).

    Returns:
        tuple: (dates, log_prices, log_volumes, log_returns)
            dates (pd.DatetimeIndex): Business dates after filtering.
            log_prices (np.ndarray): log(close).
            log_volumes (np.ndarray): log(volume + 1).
            log_returns (np.ndarray): diff(log_prices) — one element shorter.
    """
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
    """Create a boolean mask for train/test split at TRAIN_END.

    Args:
        dates (array-like): DatetimeIndex or array of dates.
        n_prices (int): Number of price observations (unused; kept for compatibility).

    Returns:
        np.ndarray: Boolean array, True for training dates (<= TRAIN_END).
    """
    train_mask = np.array([d <= pd.Timestamp(TRAIN_END) for d in dates])
    return train_mask
