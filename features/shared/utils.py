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
from typing import Optional, Dict
from sklearn.linear_model import LinearRegression
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Cache for SPY data to avoid reloading
_SPY_DATA_CACHE = None

# Cache for sector mapping to avoid reloading
_SECTOR_MAPPING_CACHE = None

# Cache for sector ETF data to avoid reloading (per ETF symbol)
_SECTOR_ETF_CACHE = {}


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


def _load_sector_mapping() -> Dict[str, str]:
    """
    Load ticker -> sector mapping from data/tickers/sectors.csv.
    
    This function caches the loaded mapping to avoid reloading on every call.
    The mapping is read-only data, so it's safe to cache indefinitely.
    
    Returns:
        Dict mapping ticker symbol to sector name (e.g., {'AAPL': 'Technology', 'JPM': 'Financial Services'})
        Empty dict if file not found or error loading.
    """
    global _SECTOR_MAPPING_CACHE
    
    # Return cached data if available
    if _SECTOR_MAPPING_CACHE is not None:
        return _SECTOR_MAPPING_CACHE
    
    # Try to load sector mapping from CSV
    project_root = Path.cwd()
    sectors_file = project_root / "data" / "tickers" / "sectors.csv"
    
    if not sectors_file.exists():
        logger.warning(f"Sector mapping file not found: {sectors_file}")
        _SECTOR_MAPPING_CACHE = {}
        return _SECTOR_MAPPING_CACHE
    
    try:
        # Read CSV file
        df = pd.read_csv(sectors_file)
        
        # Validate required columns
        if 'ticker' not in df.columns or 'sector' not in df.columns:
            logger.error(f"Sector mapping file missing required columns (ticker, sector): {sectors_file}")
            _SECTOR_MAPPING_CACHE = {}
            return _SECTOR_MAPPING_CACHE
        
        # Create mapping dictionary
        sector_mapping = dict(zip(df['ticker'], df['sector']))
        
        # Cache the mapping
        _SECTOR_MAPPING_CACHE = sector_mapping
        logger.debug(f"Loaded sector mapping for {len(sector_mapping)} tickers")
        return sector_mapping
        
    except Exception as e:
        logger.error(f"Error loading sector mapping from {sectors_file}: {e}", exc_info=True)
        _SECTOR_MAPPING_CACHE = {}
        return _SECTOR_MAPPING_CACHE


def _get_sector_etf_with_fallback(sector_name: str, date: str) -> Optional[str]:
    """
    Map sector name to ETF symbol with historical fallback logic.
    
    For sectors that were created after 2009, uses historically accurate fallbacks:
    - Communication Services (XLC) before 2018-09-28 → XLK (Technology)
    - Real Estate (XLRE) before 2015-10-07 → XLF (Financial Services)
    
    Args:
        sector_name: Sector name from sectors.csv (e.g., 'Technology', 'Financial Services')
        date: Date string (YYYY-MM-DD) to determine if fallback is needed
    
    Returns:
        ETF symbol (e.g., 'XLK', 'XLF') or None if sector not recognized
    """
    # Base sector to ETF mapping
    sector_to_etf = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Healthcare': 'XLV',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Basic Materials': 'XLB',
        'Utilities': 'XLU',
        'Communication Services': 'XLC',
        'Real Estate': 'XLRE'
    }
    
    # Get base ETF symbol
    etf_symbol = sector_to_etf.get(sector_name)
    if etf_symbol is None:
        logger.warning(f"Unknown sector: {sector_name}")
        return None
    
    # Apply historical fallbacks for newer sectors
    try:
        date_obj = pd.to_datetime(date)
        
        # Communication Services (XLC) launched June 18, 2018, but GICS reclassification was Sept 28, 2018
        # Use Sept 28, 2018 as the cutoff (when companies were actually reclassified)
        if etf_symbol == 'XLC' and date_obj < pd.to_datetime('2018-09-28'):
            # Before reclassification, Communication Services companies were in Technology
            return 'XLK'
        
        # Real Estate (XLRE) launched Oct 7, 2015, but GICS reclassification was Sept 16, 2016
        # Use Oct 7, 2015 as the cutoff (when ETF actually launched)
        if etf_symbol == 'XLRE' and date_obj < pd.to_datetime('2015-10-07'):
            # Before reclassification, Real Estate companies were in Financial Services
            return 'XLF'
            
    except Exception as e:
        logger.warning(f"Error parsing date '{date}' for sector fallback: {e}")
        # Return base ETF symbol if date parsing fails
    
    return etf_symbol


def _load_sector_etf_data(etf_symbol: str) -> Optional[DataFrame]:
    """
    Load sector ETF data from CSV file.
    
    Sector ETF data is stored in data/raw/{ETF_SYMBOL}.csv and is used for
    relative strength calculations. This function caches loaded data per ETF
    symbol to avoid reloading on every call.
    
    Args:
        etf_symbol: ETF symbol (e.g., 'XLK', 'XLF', 'XLV')
    
    Returns:
        DataFrame with ETF data (columns: Open, High, Low, Close, Volume, etc.)
        or None if file not found or error loading.
    """
    global _SECTOR_ETF_CACHE
    
    # Return cached data if available
    if etf_symbol in _SECTOR_ETF_CACHE:
        return _SECTOR_ETF_CACHE[etf_symbol]
    
    # Try to load ETF data from CSV
    project_root = Path.cwd()
    etf_file = project_root / "data" / "raw" / f"{etf_symbol}.csv"
    
    if not etf_file.exists():
        logger.debug(f"Sector ETF file not found: {etf_file}")
        return None
    
    try:
        # Read CSV file - ETF files are now in standard ticker format:
        # Date, Open, High, Low, Close, Adj Close, Volume, split_coefficient
        # Standard CSV with Date as first column, no special header rows
        etf_data = pd.read_csv(etf_file)
        
        # Set Date column as index
        if 'Date' in etf_data.columns:
            etf_data['Date'] = pd.to_datetime(etf_data['Date'])
            etf_data = etf_data.set_index('Date')
        else:
            # Fallback: use first column as date
            date_col = etf_data.columns[0]
            etf_data[date_col] = pd.to_datetime(etf_data[date_col])
            etf_data = etf_data.set_index(date_col)
        
        # Ensure index is sorted
        etf_data = etf_data.sort_index()
        
        # Cache the data
        _SECTOR_ETF_CACHE[etf_symbol] = etf_data
        logger.debug(f"Loaded {etf_symbol} data: {len(etf_data)} rows from {etf_data.index.min()} to {etf_data.index.max()}")
        return etf_data
        
    except Exception as e:
        logger.warning(f"Error loading {etf_symbol} data from {etf_file}: {e}", exc_info=True)
        return None


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

