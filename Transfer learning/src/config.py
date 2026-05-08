import os

TRAIN_START = "2010-02-01"
TRAIN_END = "2021-09-30"
TEST_START = "2021-10-01"
TEST_END = "2022-09-30"

LAGS = [2, 3, 5, 10]
ORDERS = [2, 3, 4]

ALPHA_SOURCE = 1.0
ALPHA_TRANSFER = 5.0

N_EXPERIMENTS = 200
N_SOURCE = 10
RANDOM_SEED = 42

PATH_DIM = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")

IT_TICKERS = [
    "AAPL", "ACN", "ADBE", "ADP", "ADSK", "AMD", "ANET", "AVGO", "CDNS",
    "CDW", "CRM", "CTSH", "CSCO", "FIS", "FISV", "FFIV", "FTNT", "GLW",
    "GDDY", "HPE", "HPQ", "IBM", "INFY", "INTU", "INTC", "IT", "JKHY",
    "JNPR", "KEYS", "KLAC", "LRCX", "MA", "MCHP", "MPWR", "MRVL", "MSFT",
    "MSI", "NTAP", "NVDA", "NXPI", "ON", "ORCL", "PANW", "PAYX", "PYPL",
    "QCOM", "QRVO", "SNPS", "STX", "SWKS", "TSM", "TXN", "V", "VMW",
    "VRSN", "WDC", "WIT", "XLNX", "ZBRA",
]
