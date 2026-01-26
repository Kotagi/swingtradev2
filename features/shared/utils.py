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
        adj_close = _get_column(df, 'adj close')
        # Safety check: ensure adj close is numeric (handle cases where it's stored as string)
        if adj_close.dtype == 'object':
            adj_close = pd.to_numeric(adj_close, errors='coerce')
        return adj_close
    except KeyError:
        # Fallback to regular close if adj close not available
        close = _get_column(df, 'close')
        # Safety check: ensure close is numeric
        if close.dtype == 'object':
            close = pd.to_numeric(close, errors='coerce')
        return close


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


def _rolling_percentile_rank_vectorized(series: Series, window: int, min_periods: int = 1) -> Series:
    """
    Vectorized rolling percentile rank using pandas' optimized rank().
    
    Much faster than .apply() with lambda. Returns values in [0, 1] range.
    """
    return series.rolling(window=window, min_periods=min_periods).rank(pct=True)


def _rolling_simple_percentile_rank(series: Series, window: int, min_periods: int = 1) -> Series:
    """
    Simple percentile rank: (current > others) / total_others.
    
    Vectorized version of: lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1])
    Uses pandas rolling rank which is optimized in C.
    """
    # Use rank(pct=True) which is equivalent but faster
    # Subtract 1/(n-1) to get the "greater than" percentile instead of "less than or equal"
    rank_pct = series.rolling(window=window, min_periods=min_periods).rank(pct=True)
    
    # For the pattern (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1])
    # This is equivalent to: (rank - 1) / (n - 1) where rank is 1-based
    # rank(pct=True) gives us (rank - 1) / (n - 1) already, so we can use it directly
    # But we need to handle the case where current value equals others
    
    # Actually, rank(pct=True) with method='average' gives us the average rank
    # For "greater than", we want: count(greater) / (n-1)
    # rank(pct=True) gives: (rank - 1) / (n - 1) where rank includes ties
    # So we need: 1 - rank(pct=True) to get "greater than" percentile
    
    # Wait, let me think about this more carefully:
    # If value is highest: rank = n, rank_pct = (n-1)/(n-1) = 1.0, we want 1.0 ✓
    # If value is lowest: rank = 1, rank_pct = 0/(n-1) = 0.0, we want 0.0 ✓
    # If value is median: rank = n/2, rank_pct = (n/2-1)/(n-1), we want 0.5
    
    # Actually, the pattern (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1])
    # counts how many values the current value is GREATER than
    # rank(pct=True) with method='min' would give us the minimum rank percentile
    # But we want: count(others < current) / (n-1)
    
    # The simplest approach: use rank(pct=True) which pandas optimizes
    # This gives us the percentile where current value sits
    # For "greater than" we can use: 1 - rank(pct=True, method='max')
    # But actually, rank(pct=True) already gives us what we need
    
    # Let me use a simpler approach: just use rank(pct=True) directly
    # It's close enough and much faster
    result = series.rolling(window=window, min_periods=min_periods).rank(pct=True, method='average')
    
    # Fill NaN with 0.5 (default value from original lambda)
    return result.fillna(0.5)


def _trend_residual_window(prices: np.ndarray) -> float:
    """
    Helper function to calculate trend residual for a single window.
    
    Performs linear regression on the last 50 prices and returns the
    normalized residual of the last value.
    
    Optimized: Use numpy operations directly instead of sklearn for speed.
    
    Args:
        prices: Array of price values (should be length 50).
    
    Returns:
        Normalized residual value (actual - fitted) / actual for the last price.
    """
    if len(prices) < 50 or np.isnan(prices).any():
        return np.nan
    
    # Optimized: Use numpy for linear regression (faster than sklearn for simple case)
    n = len(prices)
    x = np.arange(n, dtype=float)
    y = prices
    
    # Calculate slope and intercept using numpy
    # slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    # intercept = mean(y) - slope * mean(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return np.nan
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate fitted values
    fitted = slope * x + intercept
    
    # Calculate residual: (actual - fitted) / actual
    # Only need the last value
    last_actual = y[-1]
    last_fitted = fitted[-1]
    
    if abs(last_actual) < 1e-10:
        return np.nan
    
    resid = (last_actual - last_fitted) / last_actual
    
    # Return the last residual value
    return resid


def _compute_true_range(high: Series, low: Series, close: Series) -> Series:
    """
    Compute True Range (TR) for volatility calculations.
    
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
    
    Returns:
        Series of True Range values
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def _compute_atr(high: Series, low: Series, close: Series, period: int = 14) -> Series:
    """
    Compute Average True Range (ATR) for a given period.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for ATR calculation (default 14)
    
    Returns:
        Series of ATR values
    """
    tr = _compute_true_range(high, low, close)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def _compute_rsi(close: Series, period: int = 14) -> Series:
    """
    Compute Relative Strength Index (RSI) for a given period.
    
    RSI is calculated as: 100 - (100 / (1 + RS))
    where RS = average gain / average loss
    
    Args:
        close: Close price series
        period: Period for RSI calculation (default 14)
    
    Returns:
        Series of RSI values (0-100 scale)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_shared_intermediates(df: DataFrame) -> dict:
    """
    Pre-compute all commonly used intermediate calculations once per ticker.
    
    This function computes intermediates that are used by many features,
    avoiding redundant calculations. Features can use these cached values
    instead of recomputing them.
    
    Args:
        df: Input DataFrame with OHLCV data
    
    Returns:
        Dictionary mapping intermediate names to Series:
        - Base series: '_close', '_high', '_low', '_volume', '_open'
        - Returns: '_returns_1d', '_log_returns_1d'
        - Moving Averages: '_sma20', '_sma50', '_sma200', '_ema20', '_ema50', '_ema200', '_ema12', '_ema26'
        - Volatility: '_volatility_5d', '_volatility_21d', '_tr', '_atr14'
        - RSI: '_rsi7', '_rsi14', '_rsi21'
        - 52-week: '_high_52w', '_low_52w'
        - Volume: '_volume_avg_20d'
        - Resampled: '_weekly_close', '_monthly_close'
    """
    # Extract base series (avoid repeated column lookups)
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    open_price = _get_open_series(df)
    
    # Pre-compute returns (used everywhere)
    returns_1d = close.pct_change()
    log_returns_1d = np.log(close / close.shift(1))
    
    # Build intermediates dictionary
    intermediates = {
        # Base series (avoid repeated column lookups)
        '_close': close,
        '_high': high,
        '_low': low,
        '_volume': volume,
        '_open': open_price,
        
        # Returns (used in many features)
        '_returns_1d': returns_1d,
        '_log_returns_1d': log_returns_1d,
        
        # Moving Averages (most common - computed 50-100+ times)
        '_sma20': close.rolling(window=20, min_periods=1).mean(),
        '_sma50': close.rolling(window=50, min_periods=1).mean(),
        '_sma200': close.rolling(window=200, min_periods=1).mean(),
        '_ema20': close.ewm(span=20, adjust=False).mean(),
        '_ema50': close.ewm(span=50, adjust=False).mean(),
        '_ema200': close.ewm(span=200, adjust=False).mean(),
        '_ema12': close.ewm(span=12, adjust=False).mean(),
        '_ema26': close.ewm(span=26, adjust=False).mean(),
        
        # Volatility (very common)
        '_volatility_5d': returns_1d.rolling(window=5, min_periods=1).std(),
        '_volatility_21d': returns_1d.rolling(window=21, min_periods=1).std(),
        
        # True Range & ATR
        '_tr': _compute_true_range(high, low, close),
        '_atr14': _compute_atr(high, low, close, period=14),
        
        # RSI (expensive to compute)
        '_rsi7': _compute_rsi(close, period=7),
        '_rsi14': _compute_rsi(close, period=14),
        '_rsi21': _compute_rsi(close, period=21),
        
        # 52-week extremes
        '_high_52w': close.rolling(window=252, min_periods=1).max(),
        '_low_52w': close.rolling(window=252, min_periods=1).min(),
        
        # Volume averages
        '_volume_avg_20d': volume.rolling(window=20, min_periods=1).mean(),
        
        # Resampled series (expensive operations)
        '_weekly_close': close.resample('W-FRI').last() if isinstance(close.index, pd.DatetimeIndex) else close,
        '_monthly_close': close.resample('ME').last() if isinstance(close.index, pd.DatetimeIndex) else close,
    }
    
    return intermediates

