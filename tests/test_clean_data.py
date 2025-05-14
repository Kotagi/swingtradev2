# tests/test_clean_data.py

import pytest
import pandas as pd
from pathlib import Path

# Five tickers to validate
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL"]

# Expected dtypes in the cleaned CSVs
EXPECTED_DTYPES = {
    "Open":  "float64",
    "High":  "float64",
    "Low":   "float64",
    "Close": "float64",
    "Volume":"int64"
}

@pytest.mark.parametrize("ticker", TICKERS)
def test_cleaned_file_exists(ticker):
    """Each ticker in our validation list must have a cleaned CSV."""
    clean_path = Path("data/clean") / f"{ticker}.csv"
    assert clean_path.exists(), f"Missing cleaned file for {ticker}: {clean_path}"

@pytest.mark.parametrize("ticker", TICKERS)
def test_cleaned_data_properties(ticker):
    """Cleaned data must have a proper DateTimeIndex and correct dtypes."""
    clean_path = Path("data/clean") / f"{ticker}.csv"
    df = pd.read_csv(clean_path, index_col=0, parse_dates=True)

    # 1) Index checks
    idx = df.index
    assert isinstance(idx, pd.DatetimeIndex), "Index is not a DatetimeIndex"
    assert not idx.hasnans, "Index contains NaT values"
    assert idx.is_monotonic_increasing, "Index is not strictly increasing"
    assert idx.is_unique, "Index contains duplicate timestamps"

    # 2) Column & dtype checks
    for col, expected in EXPECTED_DTYPES.items():
        assert col in df.columns, f"Missing column '{col}'"
        actual = df[col].dtype.name
        assert actual == expected, f"Column '{col}' has dtype {actual}, expected {expected}"
