import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import IT_TICKERS, TRAIN_START, TEST_END, CACHE_DIR
from src.data_utils import _download_single

MIN_VALID = 30
DELAY_BETWEEN = 10


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

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
        df = _download_single(ticker, TRAIN_START, TEST_END, max_retries=5)
        if df is not None:
            cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
            df.to_csv(cache_path)
            success += 1
            print(f"OK ({len(df)} rows) [total: {len(already_cached) + success}]")
        else:
            print("FAILED")

        if len(already_cached) + success >= MIN_VALID:
            print(f"\nReached {MIN_VALID} tickers. Done!")
            return

        if i < len(to_download) - 1:
            print(f"  Waiting {DELAY_BETWEEN}s...", end="", flush=True)
            time.sleep(DELAY_BETWEEN)

    print(f"\nDownloaded {success}/{len(to_download)} new tickers")
    print(f"Total available: {len(already_cached) + success}")


if __name__ == "__main__":
    main()
