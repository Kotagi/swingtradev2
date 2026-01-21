"""
Shared utilities for feature computation.

Contains non-feature-specific utilities like:
- SPY data loading
- Helper functions used by multiple features
- Common data processing utilities
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from typing import Optional
from sklearn.linear_model import LinearRegression
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Cache for SPY data to avoid reloading
_SPY_DATA_CACHE = None


def _load_spy_data() -> Optional[DataFrame]:
    """
    Load SPY data from CSV file for beta calculation.
    
    SPY data is stored in data/raw/SPY.csv and is used for market context features.
    This function caches the loaded data to avoid reloading on every call.
    
    Returns:
        DataFrame with SPY data (columns: Date, Open, High, Low, Close, Volume, etc.)
        or None if file not found or error loading.
    """
    global _SPY_DATA_CACHE
    
    # Return cached data if available
    if _SPY_DATA_CACHE is not None:
        return _SPY_DATA_CACHE
    
    # Try to load SPY data from CSV
    project_root = Path.cwd()
    spy_file = project_root / "data" / "raw" / "SPY.csv"
    
    if not spy_file.exists():
        # Try alternative location (cleaned data)
        spy_file = project_root / "data" / "clean" / "SPY.csv"
        if not spy_file.exists():
            return None
    
    try:
        # Read CSV file - structure is:
        # Row 0: Column names (Price, Adj Close, Close, High, Low, Open, Volume)
        # Row 1: "Date", NaN, NaN, ...
        # Row 2+: Actual data
        # Read first row to get column names, then skip row 1
        header_row = pd.read_csv(spy_file, nrows=0)
        column_names = header_row.columns.tolist()
        
        # Read data starting from row 3 (skip first 3 rows: header, Date row, and first data row)
        # Actually, skip 2 rows and filter out the "Date" row
        spy_data = pd.read_csv(spy_file, skiprows=2, names=column_names)
        
        # Remove the "Date" row if it exists
        spy_data = spy_data[spy_data.iloc[:, 0] != 'Date'].copy()
        
        # The first column (Price) contains dates
        date_col = spy_data.columns[0]
        spy_data[date_col] = pd.to_datetime(spy_data[date_col])
        spy_data = spy_data.set_index(date_col)
        
        # Ensure index is sorted
        spy_data = spy_data.sort_index()
        
        # Cache the data
        _SPY_DATA_CACHE = spy_data
        return spy_data
    except Exception as e:
        # Return None on any error
        return None


def _get_column(df: DataFrame, col_name: str, required: bool = True) -> Series:
    """
    Get a column from DataFrame with case-insensitive matching.
    
    Args:
        df: Input DataFrame.
        col_name: Column name to find (case-insensitive).
        required: If True, raise KeyError if column not found. If False, return NaN Series.
    
    Returns:
        Series from the DataFrame, or NaN Series if not found and required=False.
    
    Raises:
        KeyError: If column not found and required=True.
    """
    col_lower = col_name.lower()
    for col in df.columns:
        if col.lower() == col_lower:
            return df[col]
    
    if not required:
        # Return NaN Series with same index as df
        return pd.Series(np.nan, index=df.index, name=col_name)
    
    raise KeyError(f"DataFrame must contain '{col_name}' column (case-insensitive)")


def _get_close_series(df: DataFrame) -> Series:
    """
    Return the closing price series, preferring 'adj close' over 'close'.
    
    Adjusted close accounts for splits and dividends, making it superior for
    ML features that need consistent price series over time. Falls back to
    'close' if 'adj close' is not available.

    Args:
        df: Input DataFrame.

    Returns:
        Series of closing prices (adjusted close if available, else close).

    Raises:
        KeyError: If neither 'adj close'/'Adj Close' nor 'close'/'Close' is present.
    """
    # Prefer adjusted close (split-adjusted, better for ML)
    try:
        return _get_column(df, 'adj close')
    except KeyError:
        # Fallback to regular close if adj close not available
        return _get_column(df, 'close')


def _get_open_series(df: DataFrame) -> Series:
    """Return the 'open' price series, case-insensitive."""
    return _get_column(df, 'open')


def _get_high_series(df: DataFrame) -> Series:
    """Return the 'high' price series, case-insensitive."""
    return _get_column(df, 'high')


def _get_low_series(df: DataFrame) -> Series:
    """Return the 'low' price series, case-insensitive."""
    return _get_column(df, 'low')


def _get_volume_series(df: DataFrame) -> Series:
    """Return the 'volume' series, case-insensitive."""
    return _get_column(df, 'volume')


def _rolling_percentile_rank(series: Series, window: int, min_periods: int = 1) -> Series:
    """
    Efficiently calculate rolling percentile rank (0-100).
    
    Uses vectorized operations for better performance than .apply() with rank().
    """
    result = pd.Series(index=series.index, dtype=float)
    values = series.values
    
    for i in range(len(series)):
        if i < min_periods - 1:
            result.iloc[i] = np.nan
            continue
        
        start_idx = max(0, i - window + 1)
        window_data = values[start_idx:i+1]
        
        if len(window_data) < min_periods:
            result.iloc[i] = np.nan
            continue
        
        current_val = values[i]
        if pd.isna(current_val) or np.isnan(current_val):
            result.iloc[i] = np.nan
            continue
        
        # Count values <= current value, subtract 1 (don't count current value itself)
        # Use numpy for faster comparison
        rank = np.sum(window_data <= current_val) - 1
        percentile = (rank / (len(window_data) - 1)) * 100 if len(window_data) > 1 else 50.0
        result.iloc[i] = percentile
    
    return result


def _trend_residual_window(prices: np.ndarray) -> float:
    """
    Helper function to calculate trend residual for a single window.
    
    Performs linear regression on the last 50 prices and returns the
    normalized residual of the last value.
    
    Args:
        prices: Array of price values (should be length 50).
    
    Returns:
        Normalized residual value (actual - fitted) / actual for the last price.
    """
    if len(prices) < 50 or np.isnan(prices).any():
        return np.nan
    
    # Create index array for regression (0 to 49)
    idx = np.arange(len(prices)).reshape(-1, 1)
    vals = prices.reshape(-1, 1)
    
    # Fit linear regression
    model = LinearRegression().fit(idx, vals)
    fitted = model.predict(idx).flatten()
    
    # Calculate residual: (actual - fitted) / actual
    resid = (vals.flatten() - fitted) / vals.flatten()
    
    # Return the last residual value
    return resid[-1]

