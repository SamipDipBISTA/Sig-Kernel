"""
Entry point for the Path Signature Transfer Learning experiment.

Reproduces the results described in "Risk of Transfer Learning and its
Applications in Finance" (Cao, Gu, Guo, Rosenbaum, arXiv:2311.03283v1),
Section 4.  The pipeline proceeds as:

  1. Download (or load cached) IT sector stock data from Yahoo Finance.
  2. Filter tickers with >= 90% business-day coverage.
  3. Precompute path-signature features for all (lag, order) combinations.
  4. Run Table 1 (grid search over lag/order) and Table 2 (correlation at L=5,M=2).
  5. Generate plots: heatmaps, bar charts, scatter plots, distribution histograms.

CLI arguments:
  --n-experiments    Number of random stock selections (default 200)
  --n-source         Number of source stocks per experiment (default 10)
  --standardization  Standardization mode: "dataset" or "per_stock"
  --skip-download    Skip data download if already cached
  --plot-only PATH   Load saved results and only regenerate plots
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import IT_TICKERS
from src.data_utils import download_all_data, get_valid_tickers
from src.experiment import (
    precompute_features,
    run_full_reproduction,
    load_results,
    plot_table1_table,
    plot_table1_heatmap,
    plot_direct_vs_transfer_bars,
    plot_table2_heatmap,
    plot_transfer_risk_scatter,
    plot_metric_distributions,
)
import src.config as cfg
import src.experiment as exp_mod


# %%
def main():
    """Parse CLI arguments and run the transfer learning experiment pipeline."""
    parser = argparse.ArgumentParser(
        description="Section 4: Transfer Learning for Stock Return Prediction"
    )
    parser.add_argument(
        "--n-experiments", type=int, default=200,
        help="Number of random stock selections (default: 200)",
    )
    parser.add_argument(
        "--n-source", type=int, default=10,
        help="Number of source stocks per experiment (default: 10)",
    )
    parser.add_argument(
        "--standardization", type=str, default="dataset",
        choices=["dataset", "per_stock"],
        help="Standardization mode (default: dataset)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip data download if cached",
    )
    parser.add_argument(
        "--plot-only", type=str, default=None,
        help="Load results from pickle file and generate plots only",
    )
    args = parser.parse_args()

    # Override global N_SOURCE for consistency across modules
    cfg.N_SOURCE = args.n_source
    exp_mod.N_SOURCE = args.n_source

    if args.plot_only:
        print(f"Loading results from {args.plot_only}...")
        all_results = load_results(args.plot_only)
        plot_table1_table(all_results["table1"])
        plot_table1_heatmap(all_results["table1"])
        plot_direct_vs_transfer_bars(all_results["table1"])
        plot_table2_heatmap(all_results["table2"])
        plot_transfer_risk_scatter(all_results["table2"])
        plot_metric_distributions(all_results["table1"])
        return

    print("Step 1: Downloading stock data...")
    all_data = download_all_data(IT_TICKERS, force=False)

    print("\nStep 2: Filtering valid tickers...")
    valid_tickers = get_valid_tickers(all_data)
    print(f"  Using {len(valid_tickers)} tickers: {valid_tickers}")

    min_needed = args.n_source + 1
    if len(valid_tickers) < min_needed:
        print(f"ERROR: Need at least {min_needed} valid tickers, got {len(valid_tickers)}")
        print("Run 'python download_data.py' first to download stock data.")
        sys.exit(1)

    print("\nStep 3: Precomputing signature features...")
    cache = precompute_features(all_data, valid_tickers)

    print("\nStep 4: Running experiments...")
    run_full_reproduction(
        cache, valid_tickers,
        n_experiments=args.n_experiments,
        standardization=args.standardization,
    )


if __name__ == "__main__":
    main()
