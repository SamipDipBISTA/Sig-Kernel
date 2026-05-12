"""
Central configuration for the Transfer Learning with Path Signatures experiment.

Implements the experimental setup described in "Risk of Transfer Learning and its
Applications in Finance" (Cao, Gu, Guo, Rosenbaum, arXiv:2311.03283v1), Section 4.

The configuration defines date ranges, signature hyperparameters, regularization
constants, experiment counts, and the list of IT sector stocks used in all
downstream scripts.
"""

import os

# --- Date ranges covering the full IT sector study period ---
# Training period: Feb 2010 to Sep 2021 (paper Section 4.2)
TRAIN_START = "2010-02-01"
TRAIN_END = "2021-09-30"
# Out-of-sample test period: Oct 2021 to Sep 2022 (paper Section 4.2)
TEST_START = "2021-10-01"
TEST_END = "2022-09-30"

# --- Signature hyperparameters (paper Section 4.1) ---
# Lag L: sliding-window length (number of days) for the signature path
LAGS = [2, 3, 5, 10]
# Order M: truncation order of the signature (paper Definition 4.1)
ORDERS = [2, 3, 4]

# --- Ridge regularization constants (paper Eqs 30-31) ---
# lambda_S: regularization for source pre-training (paper Eq 30)
ALPHA_SOURCE = 1.0
# lambda_T: regularization for transfer fine-tuning (paper Eq 31)
ALPHA_TRANSFER = 5.0

# --- Experiment design (paper Section 4.2) ---
# Number of random stock-selection experiments
N_EXPERIMENTS = 200
# Number of source stocks per experiment
N_SOURCE = 10
# Fixed random seed for reproducibility
RANDOM_SEED = 42

# --- Path dimension: (time, log_price, log_volume) = 3 ---
PATH_DIM = 3

# --- Filesystem paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

# --- 59 Information Technology sector stocks from US equity market ---
# (paper Section 4.2: IT sector, Feb 2010 to Sep 2022)
IT_TICKERS = [
    "AAPL", "ACN", "ADBE", "ADP", "ADSK", "AMD", "ANET", "AVGO", "CDNS",
    "CDW", "CRM", "CTSH", "CSCO", "FIS", "FISV", "FFIV", "FTNT", "GLW",
    "GDDY", "HPE", "HPQ", "IBM", "INFY", "INTU", "INTC", "IT", "JKHY",
    "JNPR", "KEYS", "KLAC", "LRCX", "MA", "MCHP", "MPWR", "MRVL", "MSFT",
    "MSI", "NTAP", "NVDA", "NXPI", "ON", "ORCL", "PANW", "PAYX", "PYPL",
    "QCOM", "QRVO", "SNPS", "STX", "SWKS", "TSM", "TXN", "V", "VMW",
    "VRSN", "WDC", "WIT", "XLNX", "ZBRA",
]
