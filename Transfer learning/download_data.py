"""
Standalone script to download and cache IT sector stock data from Yahoo Finance.

Downloads historical OHLCV data for all 59 IT-sector tickers defined in
config.IT_TICKERS.  Data is cached to config.CACHE_DIR as individual CSV files.

Features:
  - Stops early once MIN_VALID=30 tickers have been successfully cached.
  - Applies a DELAY_BETWEEN=10 second pause between downloads to avoid rate limiting.
  - Uses _download_single from data_utils with 5 retries per ticker.
  - Skips tickers already present in the cache.

Usage:
    python download_data.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import IT_TICKERS, TRAIN_START, TEST_END, CACHE_DIR
from src.data_utils import _download_single

# Minimum number of valid tickers needed before stopping
MIN_VALID = 30
# Seconds to wait between successive downloads (rate limiting)
DELAY_BETWEEN = 10


def main():
    """Download IT sector stock data until at least MIN_VALID tickers are cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Separate tickers into already cached and those needing download
    already_cached = []
    to_download = []
    for ticker in IT_TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(cache_path):
            already_cached.append(ticker)
        else:
            to_download.append(ticker)

    print(f"Cached: {len(already_cached)} | To download: {len(to_download)}")
    print(f"Target: at least {MIN_VALID} valid tickers")

    if len(already_cached) >= MIN_VALID:
        print("Already have enough data. Exiting.")
        return

    success = 0
    for i, ticker in enumerate(to_download):
        print(f"\n[{i+1}/{len(to_download)}] {ticker}...", end=" ", flush=True)
        # Use 5 retries for robustness
        df = _download_single(ticker, TRAIN_START, TEST_END, max_retries=5)
        if df is not None:
            cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
            df.to_csv(cache_path)
            success += 1
            print(f"OK ({len(df)} rows) [total: {len(already_cached) + success}]")
        else:
            print("FAILED")

        # Early stopping once we have enough cached tickers
        if len(already_cached) + success >= MIN_VALID:
            print(f"\nReached {MIN_VALID} tickers. Done!")
            return

        # Rate-limit: pause between downloads (except after the last one)
        if i < len(to_download) - 1:
            print(f"  Waiting {DELAY_BETWEEN}s...", end="", flush=True)
            time.sleep(DELAY_BETWEEN)

    print(f"\nDownloaded {success}/{len(to_download)} new tickers")
    print(f"Total available: {len(already_cached) + success}")


if __name__ == "__main__":
    main()
