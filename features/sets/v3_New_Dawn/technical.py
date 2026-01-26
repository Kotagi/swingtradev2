# features/sets/v3_New_Dawn/technical.py

"""
Technical feature functions for feature set 'v3_New_Dawn'.

Each function takes a DataFrame with OHLCV columns and returns a pandas Series.
All features use adjusted close prices (preferred over regular close) for consistency.
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from typing import Optional
import pandas_ta_classic as ta

# Import shared utilities
from features.shared.utils import (
    _load_spy_data,
    _get_column,
    _get_close_series,
    _get_open_series,
    _get_high_series,
    _get_low_series,
    _get_volume_series,
    _rolling_percentile_rank,
    _trend_residual_window,
    _compute_rsi,
)

# ============================================================================
# BLOCK ML-1.1: Price & Normalization (4 features)
# ============================================================================


def feature_price(df: DataFrame) -> Series:
    """
    Price: Raw closing price (adjusted close if available).
    
    === DESCRIPTION ===
    Returns the current closing price. Uses adjusted close (split-adjusted) if available,
    otherwise falls back to regular close. This is the base price feature, useful for
    filtering by price ranges (e.g., $1-$5, >$5, >$10) similar to Finviz filters.
    
    === RATIONALE ===
    Base price feature needed for:
    - Price-based filtering (screening by price ranges)
    - Reference point for other normalized features
    - Direct price comparison across stocks
    
    === CALCULATION ===
    1. Retrieve closing price series (prefers 'adj close', falls back to 'close')
    2. Return as-is (no normalization)
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses current bar's close price (available at end of day)
    - No future data
    - No rolling calculations
    
    === NORMALIZATION ===
    ML: None - raw price (may need scaling during preprocessing)
    Filter: Raw price for threshold filtering
    Clipping: None
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - Missing data: Returns NaN for missing dates
    - Zero/negative prices: Should not occur in real data
    
    === VALIDATION ===
    - [ ] Price values are positive
    - [ ] No infinities
    - [ ] Values make sense for stock prices
    
    === RELATED FEATURES ===
    price_log, price_vs_ma200, close_position_in_range
    
    === EXPECTED IMPACT ===
    Medium - Base feature, useful for filtering but may need normalization for ML
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'price' containing closing prices.
    """
    close = _get_close_series(df)
    price = close.copy()
    price.name = "price"
    return price


def feature_price_log(df: DataFrame) -> Series:
    """
    Price Log: Natural logarithm of closing price (ln(close)).
    
    === DESCRIPTION ===
    Computes the natural logarithm of the closing price. Log price squashes huge
    differences between high and low priced stocks, making it more suitable for
    ML models that need normalized inputs. Prefer this over raw price for ML applications.
    
    === RATIONALE ===
    - Normalizes price scale across different price ranges (e.g., $10 vs $1000 stocks)
    - Reduces impact of outliers (log transformation compresses large values)
    - Better for ML models that assume similar feature scales
    - Academic research shows log prices improve model performance
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Compute natural logarithm: ln(close)
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses current bar's close price (available at end of day)
    - No future data
    - No rolling calculations
    
    === NORMALIZATION ===
    ML: Log transformation normalizes scale automatically
    Filter: Not typically used for filtering
    Clipping: None
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - Zero prices: ln(0) = -inf (should not occur, but handle gracefully)
    - Negative prices: ln(negative) = NaN (should not occur)
    - Missing data: Returns NaN
    
    === VALIDATION ===
    - [ ] Log price values are finite (no -inf or NaN from invalid inputs)
    - [ ] Values make sense (e.g., ln($100) ≈ 4.6)
    - [ ] No infinities
    
    === RELATED FEATURES ===
    price, price_vs_ma200, log_return_1d
    
    === EXPECTED IMPACT ===
    High - Log transformation is crucial for ML models dealing with price data
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'price_log' containing natural log of closing prices.
    """
    close = _get_close_series(df)
    # Use log1p to handle edge cases better, then adjust: log1p(x-1) ≈ ln(x) for x > 0
    # But for prices > 1, ln(price) is fine
    price_log = np.log(close)
    price_log.name = "price_log"
    return price_log


def feature_price_vs_ma200(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    Price vs MA200: Price normalized to 200-day moving average (close / SMA200).
    
    === DESCRIPTION ===
    Computes price normalized relative to 200-day simple moving average: close / SMA(200).
    This normalizes price relative to a long-term baseline, making it comparable across
    different price ranges. Values > 1.0 indicate price above long-term average (bullish),
    values < 1.0 indicate price below long-term average (bearish).
    
    === RATIONALE ===
    - Normalizes price relative to long-term trend (200 days ≈ 1 year)
    - Makes price comparable across different stocks (e.g., $10 stock vs $1000 stock)
    - Identifies when price is above/below long-term average (momentum indicator)
    - Academic research shows price relative to MA is predictive
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Calculate 200-day SMA: SMA200 = close.rolling(200).mean()
    3. Compute ratio: price_vs_ma200 = close / SMA200
    4. First 200 days will be NaN (insufficient data)
    
    === LOOKAHEAD BIAS ===
    Risk: ⚠️ Medium
    - Uses current bar's close in SMA calculation (includes current bar)
    - For strict lookahead prevention, should use: SMA200.shift(1)
    - However, using current bar's close in MA is standard practice and acceptable
    - Decision: Use current bar (standard practice) - MA includes current bar
    
    === NORMALIZATION ===
    ML: Ratio format (1.0 = at MA, >1.0 = above, <1.0 = below)
    Filter: Can use for filtering (e.g., price > 1.1 * MA200 = strong uptrend)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - Insufficient data (< 200 days): Returns NaN for first 200 days
    - Zero SMA200: Division by zero (should not occur, but handle gracefully)
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] Ratio values are positive (price and MA should be positive)
    - [ ] First 200 days are NaN
    - [ ] Values make sense (typically 0.5 to 2.0 range)
    - [ ] No infinities or division by zero
    
    === RELATED FEATURES ===
    price, sma20_ratio, sma50_ratio, ema200_ratio
    
    === EXPECTED IMPACT ===
    High - Long-term trend normalization is crucial for ML models
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'price_vs_ma200' containing price / SMA(200) ratios.
    """
    if intermediates:
        close = intermediates['_close']
        sma200 = intermediates['_sma200']
    else:
        close = _get_close_series(df)
        # Calculate 200-day SMA (includes current bar - standard practice)
        sma200 = close.rolling(window=200, min_periods=1).mean()
    # Compute ratio
    ratio = close / sma200
    # Set NaN for insufficient data (first 200 days)
    ratio.iloc[:199] = np.nan
    ratio.name = "price_vs_ma200"
    return ratio


def feature_close_position_in_range(df: DataFrame) -> Series:
    """
    Close Position in Range: Close position within daily range (close-low)/(high-low).
    
    === DESCRIPTION ===
    Measures where the close price falls within the day's trading range.
    Calculated as: (close - low) / (high - low). Values range from 0.0 to 1.0:
    - 0.0: Close at the day's low (bearish)
    - 0.5: Close in the middle of the range (neutral)
    - 1.0: Close at the day's high (bullish)
    
    === RATIONALE ===
    - Captures intraday price action (where price closed relative to day's range)
    - Indicates buying/selling pressure (close near high = strong buying)
    - Useful for identifying reversal patterns (close at low after high = potential reversal)
    - Academic research shows intraday price position is predictive
    
    === CALCULATION ===
    1. Retrieve close, high, and low series
    2. Calculate daily range: range = high - low
    3. Calculate position: position = (close - low) / range
    4. Handle division by zero (when high == low): set to 0.5 (middle)
    5. Clip to [0, 1] to ensure valid range
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses current bar's OHLC (all available at end of day)
    - No future data
    - No rolling calculations
    
    === NORMALIZATION ===
    ML: Already normalized to [0, 1] range
    Filter: Can use for filtering (e.g., position > 0.8 = strong close)
    Clipping: Clipped to [0, 1] to handle edge cases
    
    === DATA REQUIRED ===
    OHLCV: Close, High, Low prices
    
    === EDGE CASES ===
    - Zero range (high == low): Set to 0.5 (middle position)
    - Close outside range (should not occur, but clip to [0, 1])
    - Missing data: Returns NaN
    
    === VALIDATION ===
    - [ ] Values in [0, 1] range
    - [ ] No infinities
    - [ ] Zero range handled correctly (set to 0.5)
    - [ ] Values make sense (0.0 = at low, 1.0 = at high)
    
    === RELATED FEATURES ===
    price, candle_body_pct, candle_upper_wick_pct, candle_lower_wick_pct
    
    === EXPECTED IMPACT ===
    Medium-High - Captures intraday price action, useful for ML
    
    Args:
        df: Input DataFrame with 'close', 'high', and 'low' columns.
    
    Returns:
        Series named 'close_position_in_range' containing normalized position [0, 1].
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate daily range
    daily_range = high - low
    
    # Calculate position: (close - low) / range
    # Handle division by zero (when high == low, meaning no range)
    position = (close - low) / daily_range.replace(0, np.nan)
    
    # For zero range days, set to 0.5 (middle position)
    position = position.fillna(0.5)
    
    # Clip to [0, 1] to handle any edge cases (close outside range)
    position = position.clip(0.0, 1.0)
    
    position.name = "close_position_in_range"
    return position


# ============================================================================
# BLOCK ML-1.2: Returns (8 features)
# ============================================================================


def feature_log_return_1d(df: DataFrame) -> Series:
    """
    Log Return 1d: 1-day log return (ln(close_t / close_{t-1})).
    
    === DESCRIPTION ===
    Computes the natural logarithm of the 1-day price ratio. Log returns are
    symmetric and additive, making them superior to simple returns for ML models.
    Calculated as: ln(close_t / close_{t-1}).
    
    === RATIONALE ===
    - Log returns are symmetric (ln(1/x) = -ln(x))
    - Additive over time periods
    - Better statistical properties for ML
    - Standard in quantitative finance
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Compute ratio: close_t / close_{t-1}
    3. Take natural logarithm: ln(ratio)
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses only past data (close_{t-1} is from previous day)
    - pct_change() and shift() are automatically safe
    - No future data
    
    === NORMALIZATION ===
    ML: Log transformation normalizes scale
    Filter: Not typically used for filtering
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - First day: Returns NaN (no previous close)
    - Zero/negative prices: Should not occur in real data
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] First value is NaN
    - [ ] Values are finite (no infinities)
    - [ ] Values make sense (typically -0.1 to +0.1 range)
    
    === RELATED FEATURES ===
    daily_return, price_log, weekly_return_5d
    
    === EXPECTED IMPACT ===
    High - Core return feature, fundamental for ML models
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'log_return_1d' containing 1-day log returns.
    """
    close = _get_close_series(df)
    log_ret = np.log(close / close.shift(1))
    log_ret.name = "log_return_1d"
    return log_ret


def feature_daily_return(df: DataFrame) -> Series:
    """
    Daily Return: Daily return percentage ((close_t / close_{t-1}) - 1).
    
    === DESCRIPTION ===
    Computes daily percentage return. Calculated as: (close_t / close_{t-1}) - 1.
    Returns raw values without clipping to preserve information about extreme moves
    (e.g., earnings announcements, news events).
    
    === RATIONALE ===
    - Standard return metric
    - Preserves information about extreme moves
    - No clipping - let ML learn the distribution
    - Academic research shows raw returns improve model performance
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Compute percentage change: close.pct_change()
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses only past data (close_{t-1} is from previous day)
    - pct_change() automatically uses past data
    - No future data
    
    === NORMALIZATION ===
    ML: Raw percentage returns (no clipping)
    Filter: Can use for filtering (e.g., daily_return > 0.05 = strong up day)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - First day: Returns NaN (no previous close)
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] First value is NaN
    - [ ] Values are finite
    - [ ] Values make sense (typically -0.1 to +0.1 range, but can be extreme)
    
    === RELATED FEATURES ===
    log_return_1d, gap_pct, weekly_return_5d
    
    === EXPECTED IMPACT ===
    High - Core return feature, fundamental for ML models
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'daily_return' containing daily percentage returns.
    """
    close = _get_close_series(df)
    daily_ret = close.pct_change()
    daily_ret.name = "daily_return"
    return daily_ret


def feature_gap_pct(df: DataFrame) -> Series:
    """
    Gap Percentage: Gap between open and previous close ((open_t - close_{t-1}) / close_{t-1}).
    
    === DESCRIPTION ===
    Measures the gap between today's open and yesterday's close.
    Calculated as: (open_t - close_{t-1}) / close_{t-1}.
    Positive values indicate gap-up, negative values indicate gap-down.
    
    === RATIONALE ===
    - Captures overnight price movements
    - Indicates market sentiment at open
    - Useful for identifying continuation vs reversal patterns
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve open and close series
    2. Get previous close: close.shift(1)
    3. Compute gap: (open - prev_close) / prev_close
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses current bar's open (available at start of day)
    - Uses previous bar's close (available)
    - No future data
    
    === NORMALIZATION ===
    ML: Raw gap percentages (no clipping)
    Filter: Can use for filtering (e.g., gap_pct > 0.05 = strong gap up)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Open and Close prices
    
    === EDGE CASES ===
    - First day: Returns NaN (no previous close)
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] First value is NaN
    - [ ] Values are finite
    - [ ] Values make sense (typically -0.1 to +0.1 range, but can be extreme)
    
    === RELATED FEATURES ===
    daily_return, close_position_in_range
    
    === EXPECTED IMPACT ===
    Medium-High - Captures overnight sentiment, useful for ML
    
    Args:
        df: Input DataFrame with 'open' and 'close' columns.
    
    Returns:
        Series named 'gap_pct' containing gap percentages.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    prev_close = close.shift(1)
    gap_pct = (openp - prev_close) / prev_close
    gap_pct.name = "gap_pct"
    return gap_pct


def feature_weekly_return_5d(df: DataFrame) -> Series:
    """
    Weekly Return 5d: 5-day return percentage ((close_t / close_{t-5}) - 1).
    
    === DESCRIPTION ===
    Measures the percentage return over 5 trading days (approximately one week).
    Calculated as: (close_t / close_{t-5}) - 1.
    Returns raw values without clipping to preserve information about extreme moves.
    
    === RATIONALE ===
    - Captures short-term momentum
    - Standard weekly return metric
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Compute 5-day percentage change: close.pct_change(5)
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses only past data (close_{t-5} is from 5 days ago)
    - pct_change(5) automatically uses past data
    - No future data
    
    === NORMALIZATION ===
    ML: Raw percentage returns (no clipping)
    Filter: Can use for filtering (e.g., weekly_return_5d > 0.1 = strong week)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - First 5 days: Returns NaN (insufficient data)
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] First 5 values are NaN
    - [ ] Values are finite
    - [ ] Values make sense (typically -0.2 to +0.2 range, but can be extreme)
    
    === RELATED FEATURES ===
    daily_return, monthly_return_21d, weekly_return_1w
    
    === EXPECTED IMPACT ===
    High - Short-term momentum feature, fundamental for ML
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'weekly_return_5d' containing 5-day percentage returns.
    """
    close = _get_close_series(df)
    ret_5 = close.pct_change(5)
    ret_5.name = "weekly_return_5d"
    return ret_5


def feature_monthly_return_21d(df: DataFrame) -> Series:
    """
    Monthly Return 21d: 21-day return percentage ((close_t / close_{t-21}) - 1).
    
    === DESCRIPTION ===
    Measures the percentage return over 21 trading days (approximately one month).
    Calculated as: (close_t / close_{t-21}) - 1.
    Returns raw values without clipping to preserve information about extreme moves.
    
    === RATIONALE ===
    - Captures medium-term momentum
    - Standard monthly return metric
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Compute 21-day percentage change: close.pct_change(21)
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses only past data (close_{t-21} is from 21 days ago)
    - pct_change(21) automatically uses past data
    - No future data
    
    === NORMALIZATION ===
    ML: Raw percentage returns (no clipping)
    Filter: Can use for filtering (e.g., monthly_return_21d > 0.2 = strong month)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - First 21 days: Returns NaN (insufficient data)
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] First 21 values are NaN
    - [ ] Values are finite
    - [ ] Values make sense (typically -0.3 to +0.3 range, but can be extreme)
    
    === RELATED FEATURES ===
    weekly_return_5d, quarterly_return_63d, ytd_return
    
    === EXPECTED IMPACT ===
    High - Medium-term momentum feature, fundamental for ML
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'monthly_return_21d' containing 21-day percentage returns.
    """
    close = _get_close_series(df)
    ret_21 = close.pct_change(21)
    ret_21.name = "monthly_return_21d"
    return ret_21


def feature_quarterly_return_63d(df: DataFrame) -> Series:
    """
    Quarterly Return 63d: 63-day return percentage ((close_t / close_{t-63}) - 1).
    
    === DESCRIPTION ===
    Measures the percentage return over 63 trading days (approximately one quarter).
    Calculated as: (close_t / close_{t-63}) - 1.
    Returns raw values without clipping to preserve information about extreme moves.
    
    === RATIONALE ===
    - Captures longer-term momentum
    - Standard quarterly return metric
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Compute 63-day percentage change: close.pct_change(63)
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses only past data (close_{t-63} is from 63 days ago)
    - pct_change(63) automatically uses past data
    - No future data
    
    === NORMALIZATION ===
    ML: Raw percentage returns (no clipping)
    Filter: Can use for filtering (e.g., quarterly_return_63d > 0.3 = strong quarter)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - First 63 days: Returns NaN (insufficient data)
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] First 63 values are NaN
    - [ ] Values are finite
    - [ ] Values make sense (typically -0.5 to +0.5 range, but can be extreme)
    
    === RELATED FEATURES ===
    monthly_return_21d, ytd_return
    
    === EXPECTED IMPACT ===
    Medium-High - Longer-term momentum feature
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'quarterly_return_63d' containing 63-day percentage returns.
    """
    close = _get_close_series(df)
    ret_63 = close.pct_change(63)
    ret_63.name = "quarterly_return_63d"
    return ret_63


def feature_ytd_return(df: DataFrame) -> Series:
    """
    YTD Return: Year-to-Date return percentage.
    
    === DESCRIPTION ===
    Measures the percentage return from the first trading day of the year to the current date.
    Calculated as: (close / first_close_of_year) - 1.
    Returns raw values without clipping to preserve information about extreme moves.
    
    === RATIONALE ===
    - Captures year-to-date performance
    - Useful for identifying strong/weak years
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Group by year and get first close of each year
    3. Calculate return: (current_close / first_close_of_year) - 1
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses only past data (first_close_of_year is from earlier in the year)
    - No future data
    
    === NORMALIZATION ===
    ML: Raw percentage returns (no clipping)
    Filter: Can use for filtering (e.g., ytd_return > 0.5 = strong year)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close) with DatetimeIndex
    
    === EDGE CASES ===
    - First day of year: Returns 0.0 (no change)
    - Missing data: Propagates NaN
    - Non-DatetimeIndex: Attempts to convert
    
    === VALIDATION ===
    - [ ] First day of each year is 0.0
    - [ ] Values are finite
    - [ ] Values make sense (can range widely over a year)
    
    === RELATED FEATURES ===
    quarterly_return_63d, monthly_return_21d
    
    === EXPECTED IMPACT ===
    Medium - Year-to-date performance feature
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column and DatetimeIndex.
    
    Returns:
        Series named 'ytd_return' containing YTD percentage returns.
    """
    close = _get_close_series(df)
    
    # Ensure index is DatetimeIndex
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    
    # Get first close price of each year
    first_close_of_year = close.groupby(close.index.year).transform('first')
    
    # Calculate YTD return: (current_close / first_close_of_year) - 1
    ytd = (close / first_close_of_year) - 1
    
    ytd.name = "ytd_return"
    return ytd


def feature_weekly_return_1w(df: DataFrame) -> Series:
    """
    Weekly Return 1w: 1-week log return (ln(close_t / close_{t-5})).
    
    === DESCRIPTION ===
    Computes the natural logarithm of the 5-day price ratio (1-week log return).
    This is the log return version of weekly_return_5d, providing symmetric and
    additive properties. Calculated as: ln(close_t / close_{t-5}).
    
    === RATIONALE ===
    - Log returns are symmetric and additive
    - Better statistical properties for ML
    - Complements weekly_return_5d (percentage version)
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Compute ratio: close_t / close_{t-5}
    3. Take natural logarithm: ln(ratio)
    
    === LOOKAHEAD BIAS ===
    Risk: ✅ Low
    - Uses only past data (close_{t-5} is from 5 days ago)
    - No future data
    
    === NORMALIZATION ===
    ML: Log transformation normalizes scale
    Filter: Not typically used for filtering
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - First 5 days: Returns NaN (insufficient data)
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] First 5 values are NaN
    - [ ] Values are finite
    - [ ] Values make sense (typically -0.2 to +0.2 range)
    
    === RELATED FEATURES ===
    log_return_1d, weekly_return_5d
    
    === EXPECTED IMPACT ===
    Medium - Log return version of weekly return
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'weekly_return_1w' containing 1-week log returns.
    """
    close = _get_close_series(df)
    log_ret_5 = np.log(close / close.shift(5))
    log_ret_5.name = "weekly_return_1w"
    return log_ret_5


# ============================================================================
# BLOCK ML-1.3: 52-Week Position (3 features)
# ============================================================================


def feature_dist_52w_high(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    Distance to 52-Week High: (close / high_52w) - 1.
    
    === DESCRIPTION ===
    Measures how far the current price is from the 52-week (252 trading days) high.
    Calculated as: (close / high_52w) - 1.
    Values: 0.0 = at high, negative = below high, positive = above high (rare).
    
    === RATIONALE ===
    - Identifies stocks near 52-week highs (momentum indicator)
    - Useful for breakout detection
    - Normalizes price relative to yearly range
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Calculate 52-week high: close.rolling(252).max()
    3. Compute distance: (close / high_52w) - 1
    
    === LOOKAHEAD BIAS ===
    Risk: ⚠️ Medium
    - Uses current bar's close in rolling max calculation
    - For strict prevention, should use: high_52w.shift(1)
    - However, using current bar is standard practice
    - Decision: Use current bar (standard practice)
    
    === NORMALIZATION ===
    ML: Distance format (0.0 = at high, negative = below)
    Filter: Can use for filtering (e.g., dist_52w_high > -0.1 = near high)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - Insufficient data (< 252 days): Uses available data (min_periods=1)
    - Zero high_52w: Should not occur, but handle gracefully
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] Values are finite
    - [ ] Typically negative (price below high)
    - [ ] Values make sense (typically -0.5 to 0.0 range)
    
    === RELATED FEATURES ===
    dist_52w_low, pos_52w
    
    === EXPECTED IMPACT ===
    Medium-High - Momentum indicator, useful for ML
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'dist_52w_high' containing distance to 52-week high.
    """
    if intermediates:
        close = intermediates['_close']
        high_52 = intermediates['_high_52w']
    else:
        close = _get_close_series(df)
        # Calculate 52-week (252 trading days) high
        high_52 = close.rolling(window=252, min_periods=1).max()
    
    # Calculate distance: (current_price / 52w_high) - 1
    dist_52_high = (close / high_52) - 1
    
    dist_52_high.name = "dist_52w_high"
    return dist_52_high


def feature_dist_52w_low(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    Distance to 52-Week Low: (close / low_52w) - 1.
    
    === DESCRIPTION ===
    Measures how far the current price is from the 52-week (252 trading days) low.
    Calculated as: (close / low_52w) - 1.
    Values: 0.0 = at low, positive = above low, negative = below low (rare).
    
    === RATIONALE ===
    - Identifies stocks near 52-week lows (potential reversal indicator)
    - Useful for support level detection
    - Normalizes price relative to yearly range
    - No clipping - let ML learn the distribution
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Calculate 52-week low: close.rolling(252).min()
    3. Compute distance: (close / low_52w) - 1
    
    === LOOKAHEAD BIAS ===
    Risk: ⚠️ Medium
    - Uses current bar's close in rolling min calculation
    - For strict prevention, should use: low_52w.shift(1)
    - However, using current bar is standard practice
    - Decision: Use current bar (standard practice)
    
    === NORMALIZATION ===
    ML: Distance format (0.0 = at low, positive = above)
    Filter: Can use for filtering (e.g., dist_52w_low < 0.1 = near low)
    Clipping: None - let ML learn the distribution
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - Insufficient data (< 252 days): Uses available data (min_periods=1)
    - Zero low_52w: Should not occur, but handle gracefully
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] Values are finite
    - [ ] Typically positive (price above low)
    - [ ] Values make sense (typically 0.0 to 2.0 range)
    
    === RELATED FEATURES ===
    dist_52w_high, pos_52w
    
    === EXPECTED IMPACT ===
    Medium-High - Support/reversal indicator, useful for ML
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'dist_52w_low' containing distance to 52-week low.
    """
    if intermediates:
        close = intermediates['_close']
        low_52 = intermediates['_low_52w']
    else:
        close = _get_close_series(df)
        # Calculate 52-week (252 trading days) low
        low_52 = close.rolling(window=252, min_periods=1).min()
    
    # Calculate distance: (current_price / 52w_low) - 1
    dist_52_low = (close / low_52) - 1
    
    dist_52_low.name = "dist_52w_low"
    return dist_52_low


def feature_pos_52w(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    52-Week Position: (close - low_52) / (high_52 - low_52).
    
    === DESCRIPTION ===
    Measures the normalized position of the current price within the 52-week range.
    Calculated as: (close - low_52) / (high_52 - low_52).
    Values: 0.0 = at low, 1.0 = at high, 0.5 = middle of range.
    
    === RATIONALE ===
    - Normalizes price position within yearly range
    - Identifies where price is in the 52-week cycle
    - Useful for momentum and reversal detection
    - Clipped to [0, 1] for safety (handles edge cases)
    
    === CALCULATION ===
    1. Retrieve closing price series (adjusted close preferred)
    2. Calculate 52-week high and low
    3. Compute position: (close - low_52) / (high_52 - low_52)
    4. Handle division by zero (when high == low): set to 0.5
    5. Clip to [0, 1] for safety
    
    === LOOKAHEAD BIAS ===
    Risk: ⚠️ Medium
    - Uses current bar's close in rolling calculations
    - For strict prevention, should use shifted values
    - However, using current bar is standard practice
    - Decision: Use current bar (standard practice)
    
    === NORMALIZATION ===
    ML: Already normalized to [0, 1] range
    Filter: Can use for filtering (e.g., pos_52w > 0.8 = near high)
    Clipping: Clipped to [0, 1] for safety
    
    === DATA REQUIRED ===
    OHLCV: Close price (or Adj Close)
    
    === EDGE CASES ===
    - Zero range (high == low): Set to 0.5 (middle position)
    - Insufficient data: Uses available data
    - Missing data: Propagates NaN
    
    === VALIDATION ===
    - [ ] Values in [0, 1] range
    - [ ] No infinities
    - [ ] Zero range handled correctly
    
    === RELATED FEATURES ===
    dist_52w_high, dist_52w_low
    
    === EXPECTED IMPACT ===
    Medium-High - Position indicator, useful for ML and filtering
    
    Args:
        df: Input DataFrame with 'close' or 'adj close' column.
    
    Returns:
        Series named 'pos_52w' containing normalized 52-week position [0, 1].
    """
    if intermediates:
        close = intermediates['_close']
        high_52 = intermediates['_high_52w']
        low_52 = intermediates['_low_52w']
    else:
        close = _get_close_series(df)
        # Calculate 52-week (252 trading days) high and low
        high_52 = close.rolling(window=252, min_periods=1).max()
        low_52 = close.rolling(window=252, min_periods=1).min()
    
    # Calculate position: (current_price - low) / (high - low)
    range_52 = high_52 - low_52
    # Handle division by zero (when high == low)
    pos_52 = (close - low_52) / range_52.replace(0, np.nan)
    
    # For zero range days, set to 0.5 (middle position)
    pos_52 = pos_52.fillna(0.5)
    
    # Clip to [0, 1] for safety
    pos_52 = pos_52.clip(0.0, 1.0)
    pos_52.name = "pos_52w"
    return pos_52


# ============================================================================
# BLOCK ML-1.4: Basic Moving Averages (8 features)
# ============================================================================


def feature_sma20_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA20 Ratio: close / SMA(20).
    
    Measures current price relative to 20-day simple moving average.
    Values: 1.0 = at MA, >1.0 = above (bullish), <1.0 = below (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        sma20 = intermediates['_sma20']
    else:
        close = _get_close_series(df)
        sma20 = close.rolling(window=20, min_periods=1).mean()
    ratio = close / sma20
    ratio.name = "sma20_ratio"
    return ratio


def feature_sma50_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA50 Ratio: close / SMA(50).
    
    Measures current price relative to 50-day simple moving average.
    Values: 1.0 = at MA, >1.0 = above (bullish), <1.0 = below (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        sma50 = intermediates['_sma50']
    else:
        close = _get_close_series(df)
        sma50 = close.rolling(window=50, min_periods=1).mean()
    ratio = close / sma50
    ratio.name = "sma50_ratio"
    return ratio


def feature_sma200_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA200 Ratio: close / SMA(200).
    
    Measures current price relative to 200-day simple moving average.
    Values: 1.0 = at MA, >1.0 = above (bullish long-term), <1.0 = below (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        sma200 = intermediates['_sma200']
    else:
        close = _get_close_series(df)
        sma200 = close.rolling(window=200, min_periods=1).mean()
    ratio = close / sma200
    ratio.name = "sma200_ratio"
    return ratio


def feature_ema20_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    EMA20 Ratio: close / EMA(20).
    
    Measures current price relative to 20-day exponential moving average.
    EMA gives more weight to recent prices than SMA.
    Values: 1.0 = at MA, >1.0 = above (bullish), <1.0 = below (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        ema20 = intermediates['_ema20']
    else:
        close = _get_close_series(df)
        ema20 = close.ewm(span=20, adjust=False).mean()
    ratio = close / ema20
    ratio.name = "ema20_ratio"
    return ratio


def feature_ema50_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    EMA50 Ratio: close / EMA(50).
    
    Measures current price relative to 50-day exponential moving average.
    EMA gives more weight to recent prices than SMA.
    Values: 1.0 = at MA, >1.0 = above (bullish), <1.0 = below (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        ema50 = intermediates['_ema50']
    else:
        close = _get_close_series(df)
        ema50 = close.ewm(span=50, adjust=False).mean()
    ratio = close / ema50
    ratio.name = "ema50_ratio"
    return ratio


def feature_ema200_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    EMA200 Ratio: close / EMA(200).
    
    Measures current price relative to 200-day exponential moving average.
    EMA gives more weight to recent prices than SMA.
    Values: 1.0 = at MA, >1.0 = above (bullish long-term), <1.0 = below (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        ema200 = intermediates['_ema200']
    else:
        close = _get_close_series(df)
        ema200 = close.ewm(span=200, adjust=False).mean()
    ratio = close / ema200
    ratio.name = "ema200_ratio"
    return ratio


def feature_sma20_sma50_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA20/SMA50 Ratio: SMA(20) / SMA(50).
    
    Moving average crossover indicator. Measures relationship between short-term (20-day)
    and medium-term (50-day) moving averages.
    Values: 1.0 = equal, >1.0 = SMA20 above SMA50 (bullish crossover), <1.0 = bearish.
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        sma20 = intermediates['_sma20']
        sma50 = intermediates['_sma50']
    else:
        close = _get_close_series(df)
        sma20 = close.rolling(window=20, min_periods=1).mean()
        sma50 = close.rolling(window=50, min_periods=1).mean()
    ratio = sma20 / sma50
    ratio.name = "sma20_sma50_ratio"
    return ratio


def feature_sma50_sma200_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA50/SMA200 Ratio: SMA(50) / SMA(200).
    
    Classic moving average crossover indicator (Golden Cross/Death Cross).
    Measures relationship between medium-term (50-day) and long-term (200-day) MAs.
    Values: 1.0 = equal, >1.0 = Golden Cross (bullish), <1.0 = Death Cross (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        sma50 = intermediates['_sma50']
        sma200 = intermediates['_sma200']
    else:
        close = _get_close_series(df)
        sma50 = close.rolling(window=50, min_periods=1).mean()
        sma200 = close.rolling(window=200, min_periods=1).mean()
    ratio = sma50 / sma200
    ratio.name = "sma50_sma200_ratio"
    return ratio


# ============================================================================
# BLOCK ML-1.5: MA Slopes (4 features)
# ============================================================================


def feature_sma20_slope(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA20 Slope: sma20.diff(5) / close.
    
    Measures 5-day change in 20-day SMA, normalized by current price.
    Indicates rate of change (slope) of short-term trend.
    Values: 0.0 = flat, >0.0 = rising (bullish), <0.0 = falling (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        sma20 = intermediates['_sma20']
    else:
        close = _get_close_series(df)
        sma20 = close.rolling(window=20, min_periods=1).mean()
    slope = sma20.diff(5) / close
    slope.name = "sma20_slope"
    return slope


def feature_sma50_slope(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA50 Slope: sma50.diff(5) / close.
    
    Measures 5-day change in 50-day SMA, normalized by current price.
    Indicates rate of change (slope) of medium-term trend.
    Values: 0.0 = flat, >0.0 = rising (bullish), <0.0 = falling (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        sma50 = intermediates['_sma50']
    else:
        close = _get_close_series(df)
        sma50 = close.rolling(window=50, min_periods=1).mean()
    slope = sma50.diff(5) / close
    slope.name = "sma50_slope"
    return slope


def feature_sma200_slope(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    SMA200 Slope: sma200.diff(10) / close.
    
    Measures 10-day change in 200-day SMA, normalized by current price.
    Indicates rate of change (slope) of long-term trend.
    Values: 0.0 = flat, >0.0 = rising (bullish), <0.0 = falling (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        sma200 = intermediates['_sma200']
    else:
        close = _get_close_series(df)
        sma200 = close.rolling(window=200, min_periods=1).mean()
    slope = sma200.diff(10) / close
    slope.name = "sma200_slope"
    return slope


def feature_ema20_slope(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    EMA20 Slope: ema20.diff(5) / close.
    
    Measures 5-day change in 20-day EMA, normalized by current price.
    EMA gives more weight to recent prices. Indicates rate of change of short-term trend.
    Values: 0.0 = flat, >0.0 = rising (bullish), <0.0 = falling (bearish).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        ema20 = intermediates['_ema20']
    else:
        close = _get_close_series(df)
        ema20 = close.ewm(span=20, adjust=False).mean()
    slope = ema20.diff(5) / close
    slope.name = "ema20_slope"
    return slope


# ============================================================================
# BLOCK ML-1.6: Basic Volatility (3 features)
# ============================================================================


def feature_volatility_5d(df: DataFrame) -> Series:
    """
    5-Day Volatility: Standard deviation of daily returns over 5-day rolling window.
    
    Measures short-term price volatility. Higher values indicate more volatile movements.
    Calculated as: std(close.pct_change()) over 5 days.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    vol_5 = close.pct_change().rolling(window=5, min_periods=1).std()
    vol_5.name = "volatility_5d"
    return vol_5


def feature_volatility_21d(df: DataFrame) -> Series:
    """
    21-Day Volatility: Standard deviation of daily returns over 21-day rolling window.
    
    Measures medium-term price volatility. Higher values indicate more volatile movements.
    Calculated as: std(close.pct_change()) over 21 days.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    vol_21 = close.pct_change().rolling(window=21, min_periods=1).std()
    vol_21.name = "volatility_21d"
    return vol_21


def feature_atr14_normalized(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    Normalized ATR14: ATR(14) / close.
    
    Average True Range over 14 days, normalized by current price.
    ATR measures volatility based on true range (high-low, high-prev_close, low-prev_close).
    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close)).
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        close = intermediates['_close']
        atr14 = intermediates['_atr14']
    else:
        close = _get_close_series(df)
        high = _get_high_series(df)
        low = _get_low_series(df)
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        
        # True Range is the maximum of the three components
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR14 (14-day rolling mean of True Range)
        atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # Normalize by current price
    atr_norm = atr14 / close
    atr_norm.name = "atr14_normalized"
    return atr_norm


# ============================================================================
# BLOCK ML-2.1: Basic Volume (6 features)
# ============================================================================


def feature_log_volume(df: DataFrame) -> Series:
    """
    Log Volume: Log-transformed volume (log1p(volume)).
    
    Transforms volume using natural logarithm of (1 + volume).
    Log1p handles zero volumes gracefully and compresses wide volume ranges.
    No clipping - let ML learn the distribution.
    """
    volume = _get_volume_series(df)
    log_vol = np.log1p(volume)
    log_vol.name = "log_volume"
    return log_vol


def feature_log_avg_volume_20d(df: DataFrame) -> Series:
    """
    Log Average Volume 20d: Log-transformed 20-day average volume.
    
    Provides smoothed, normalized view of volume trends over medium term.
    Calculated as: log1p(volume.rolling(20).mean()).
    No clipping - let ML learn the distribution.
    """
    volume = _get_volume_series(df)
    vol_avg20 = volume.rolling(window=20, min_periods=1).mean()
    log_avg_vol = np.log1p(vol_avg20)
    log_avg_vol.name = "log_avg_volume_20d"
    return log_avg_vol


def feature_relative_volume(df: DataFrame) -> Series:
    """
    Relative Volume: Current volume relative to 20-day average (volume / avg_volume).
    
    Measures current volume relative to 20-day average.
    Values > 1.0 = above average, < 1.0 = below average.
    No clipping - let ML learn the distribution.
    """
    volume = _get_volume_series(df)
    vol_avg20 = volume.rolling(window=20, min_periods=1).mean()
    rvol = volume / vol_avg20
    rvol.name = "relative_volume"
    return rvol


def feature_obv_momentum(df: DataFrame) -> Series:
    """
    OBV Momentum: 10-day rate of change of On-Balance Volume.
    
    OBV shows cumulative volume moving with price. Rate of change captures
    volume acceleration into trend. Calculated as: OBV.pct_change(10).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Calculate OBV: add volume on up days, subtract on down days
    price_change = close.diff()
    signed_volume = volume.copy()
    signed_volume = signed_volume.where(price_change > 0, -signed_volume)
    signed_volume = signed_volume.where(price_change != 0, 0)
    obv = signed_volume.cumsum()
    
    # Calculate 10-day percentage change
    obv_roc = obv.pct_change(10)
    obv_roc.name = "obv_momentum"
    return obv_roc


def feature_volume_trend(df: DataFrame) -> Series:
    """
    Volume Trend: 20-day volume moving average slope (vol_ma.diff(5) / volume).
    
    Measures rate of change of volume trend, normalized by current volume.
    Indicates whether volume is increasing or decreasing.
    No clipping - let ML learn the distribution.
    """
    volume = _get_volume_series(df)
    vol_ma20 = volume.rolling(window=20, min_periods=1).mean()
    vol_trend = vol_ma20.diff(5) / (volume + 1e-10)  # Add small value to avoid division by zero
    vol_trend.name = "volume_trend"
    return vol_trend


def feature_volume_momentum(df: DataFrame) -> Series:
    """
    Volume Momentum: Volume rate of change (volume.pct_change(10)).
    
    Measures 10-day percentage change in volume.
    Captures volume acceleration/deceleration.
    No clipping - let ML learn the distribution.
    """
    volume = _get_volume_series(df)
    vol_mom = volume.pct_change(10)
    vol_mom.name = "volume_momentum"
    return vol_mom


# ============================================================================
# BLOCK ML-2.2: RSI Family (4 features)
# ============================================================================


def feature_rsi7(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    RSI7: Relative Strength Index (7-period), centered to [-1, +1].
    
    Shorter-term momentum indicator. Calculated same as RSI14 but with 7-period window.
    Centered: (rsi - 50) / 50. No clipping - let ML learn the distribution.
    """
    if intermediates:
        rsi = intermediates['_rsi7']
    else:
        close = _get_close_series(df)
        rsi = _compute_rsi(close, period=7)
    rsi_centered = (rsi - 50) / 50
    rsi_centered.name = "rsi7"
    return rsi_centered


def feature_rsi14(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    RSI14: Relative Strength Index (14-period), centered to [-1, +1].
    
    Standard momentum indicator. Centered: (rsi - 50) / 50.
    No clipping - let ML learn the distribution.
    """
    if intermediates:
        rsi = intermediates['_rsi14']
    else:
        close = _get_close_series(df)
        rsi = _compute_rsi(close, period=14)
    rsi_centered = (rsi - 50) / 50
    rsi_centered.name = "rsi14"
    return rsi_centered


def feature_rsi21(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    RSI21: Relative Strength Index (21-period), centered to [-1, +1].
    
    Longer-term momentum indicator. Calculated same as RSI14 but with 21-period window.
    Centered: (rsi - 50) / 50. No clipping - let ML learn the distribution.
    """
    if intermediates:
        rsi = intermediates['_rsi21']
    else:
        close = _get_close_series(df)
        rsi = _compute_rsi(close, period=21)
    rsi_centered = (rsi - 50) / 50
    rsi_centered.name = "rsi21"
    return rsi_centered


def feature_rsi_momentum(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    """
    RSI Momentum: Rate of change of RSI14 (rsi14.diff(5)).
    
    Measures acceleration/deceleration of momentum.
    Positive = momentum accelerating, negative = momentum decelerating.
    No clipping - let ML learn the distribution.
    """
    rsi14 = feature_rsi14(df, intermediates=intermediates)
    rsi_mom = rsi14.diff(5)
    rsi_mom.name = "rsi_momentum"
    return rsi_mom


# ============================================================================
# BLOCK ML-2.3: MACD Family (4 features)
# ============================================================================


def feature_macd_line(df: DataFrame) -> Series:
    """
    MACD Line: EMA(12) - EMA(26), normalized by price.
    
    MACD line measures momentum. Normalized by dividing by close price.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26) / close
    macd_line.name = "macd_line"
    return macd_line


def feature_macd_signal(df: DataFrame) -> Series:
    """
    MACD Signal: EMA(9) of MACD line, normalized by price.
    
    Signal line is smoothed version of MACD line.
    Normalized by dividing by close price.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    signal_norm = signal_line / close
    signal_norm.name = "macd_signal"
    return signal_norm


def feature_macd_histogram_normalized(df: DataFrame) -> Series:
    """
    MACD Histogram Normalized: (MACD line - Signal line) / close.
    
    Histogram measures momentum acceleration/deceleration.
    Most predictive part of MACD. Normalized by price.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_line - signal_line) / close
    macd_hist.name = "macd_histogram_normalized"
    return macd_hist


def feature_macd_momentum(df: DataFrame) -> Series:
    """
    MACD Momentum: Rate of change of MACD histogram (macd_hist.diff(5)).
    
    Measures acceleration of momentum acceleration.
    Positive = momentum acceleration increasing, negative = decreasing.
    No clipping - let ML learn the distribution.
    """
    macd_hist = feature_macd_histogram_normalized(df)
    macd_mom = macd_hist.diff(5)
    macd_mom.name = "macd_momentum"
    return macd_mom


# ============================================================================
# BLOCK ML-2.4: ROC Family (4 features)
# ============================================================================


def feature_roc5(df: DataFrame) -> Series:
    """
    ROC5: Rate of Change (5-period) - very short-term momentum.
    
    Calculated as: (close - close.shift(5)) / close.shift(5).
    Captures velocity of price movement over 5 days.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    roc5 = (close - close.shift(5)) / close.shift(5)
    roc5.name = "roc5"
    return roc5


def feature_roc10(df: DataFrame) -> Series:
    """
    ROC10: Rate of Change (10-period) - short-term momentum.
    
    Calculated as: (close - close.shift(10)) / close.shift(10).
    Captures velocity of price movement over 10 days.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    roc10 = (close - close.shift(10)) / close.shift(10)
    roc10.name = "roc10"
    return roc10


def feature_roc20(df: DataFrame) -> Series:
    """
    ROC20: Rate of Change (20-period) - medium-term momentum.
    
    Calculated as: (close - close.shift(20)) / close.shift(20).
    Captures velocity of price movement over 20 days.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    roc20 = (close - close.shift(20)) / close.shift(20)
    roc20.name = "roc20"
    return roc20


def feature_momentum_10d(df: DataFrame) -> Series:
    """
    Momentum 10d: Price momentum (close - close[10]), normalized by price.
    
    Absolute price change over 10 days, normalized by current price.
    Different from ROC as it's absolute change, not percentage.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = (close - close.shift(10)) / close
    momentum.name = "momentum_10d"
    return momentum


# ============================================================================
# BLOCK ML-2.5: Stochastic Family (3 features)
# ============================================================================


def feature_stochastic_k14(df: DataFrame) -> Series:
    """
    Stochastic %K (14-period): (close - low_14) / (high_14 - low_14).
    
    Measures where price sits within recent 14-period range.
    Values: 0.0 = at low, 1.0 = at high.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    low_14 = low.rolling(window=14, min_periods=1).min()
    high_14 = high.rolling(window=14, min_periods=1).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    stoch_k = (close - low_14) / range_14
    stoch_k = stoch_k.fillna(0.5).clip(0.0, 1.0)
    stoch_k.name = "stochastic_k14"
    return stoch_k


def feature_stochastic_d14(df: DataFrame) -> Series:
    """
    Stochastic %D (14-period): 3-period SMA of %K (signal line).
    
    Smoothed version of %K. Used as signal line for %K.
    Already normalized to [0, 1]. Clipped for safety.
    """
    stoch_k = feature_stochastic_k14(df)
    stoch_d = stoch_k.rolling(window=3, min_periods=1).mean()
    stoch_d = stoch_d.clip(0.0, 1.0)
    stoch_d.name = "stochastic_d14"
    return stoch_d


def feature_stochastic_oscillator(df: DataFrame) -> Series:
    """
    Stochastic Oscillator: %K - %D (momentum of stochastic).
    
    Measures momentum of stochastic indicator.
    Positive = %K above %D (bullish), negative = %K below %D (bearish).
    No clipping - let ML learn the distribution.
    """
    stoch_k = feature_stochastic_k14(df)
    stoch_d = feature_stochastic_d14(df)
    stoch_osc = stoch_k - stoch_d
    stoch_osc.name = "stochastic_oscillator"
    return stoch_osc


# ============================================================================
# BLOCK ML-2.6: Other Oscillators (9 features)
# ============================================================================


def feature_cci20(df: DataFrame) -> Series:
    """
    CCI20: Commodity Channel Index (20-period), normalized with tanh.
    
    Measures deviation from typical price relative to volatility.
    Normalized: tanh(cci / 100) to compress to reasonable range.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=20, min_periods=20).mean()
    # Optimized: Use vectorized mean deviation instead of .apply()
    # Mean absolute deviation = mean(|x - mean(x)|)
    mean_deviation = typical_price.rolling(window=20, min_periods=20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )  # Note: This is already efficient with raw=True, but we can optimize further
    # Alternative: typical_price.rolling(window=20).std() * np.sqrt(2/np.pi) for MAD approximation
    # For now, keeping as-is since raw=True is already optimized
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)
    cci_norm = np.tanh(cci / 100)
    cci_norm.name = "cci20"
    return cci_norm


def feature_williams_r14(df: DataFrame) -> Series:
    """
    Williams %R (14-period): Range momentum oscillator, normalized to [0, 1].
    
    Calculated as: (highest_high - close) / (highest_high - lowest_low) * -100, then normalized.
    Values: 0.0 = at high (overbought), 1.0 = at low (oversold).
    Already normalized to [0, 1]. Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    highest_high = high.rolling(window=14, min_periods=14).max()
    lowest_low = low.rolling(window=14, min_periods=14).min()
    price_range = (highest_high - lowest_low).replace(0, np.nan)
    williams_r = ((highest_high - close) / (price_range + 1e-10)) * -100
    williams_r_norm = -(williams_r / 100).clip(0.0, 1.0)
    williams_r_norm.name = "williams_r14"
    return williams_r_norm


def feature_ppo_histogram(df: DataFrame) -> Series:
    """
    PPO Histogram: Percentage Price Oscillator histogram.
    
    PPO = (EMA12 - EMA26) / EMA26. Histogram = PPO - PPO_signal.
    Percentage-based momentum, scale-invariant.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ppo = (ema12 - ema26) / ema26
    ppo_signal = ppo.ewm(span=9, adjust=False).mean()
    ppo_hist = ppo - ppo_signal
    ppo_hist.name = "ppo_histogram"
    return ppo_hist


def feature_dpo(df: DataFrame) -> Series:
    """
    DPO: Detrended Price Oscillator (20-period), normalized by price.
    
    Removes long-term trend, highlights short-term cycles.
    Calculated as: (close - shifted_sma20) / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma20 = close.rolling(window=20, min_periods=1).mean()
    shifted_sma = sma20.shift(11)  # Shift by period//2 + 1
    dpo = (close - shifted_sma) / close
    dpo.name = "dpo"
    return dpo


def feature_kama_slope(df: DataFrame) -> Series:
    """
    KAMA Slope: Kaufman Adaptive Moving Average slope, normalized by price.
    
    KAMA adapts to market efficiency. Slope measures adaptive trend strength.
    Calculated as: kama.diff() / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    period = 10
    fast = 2
    slow = 30
    
    # Calculate Efficiency Ratio
    change = abs(close - close.shift(period))
    volatility = close.diff().abs().rolling(window=period, min_periods=period).sum()
    er = (change / (volatility + 1e-10)).clip(0.0, 1.0)
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA (initialize with SMA, then recursive)
    kama = close.rolling(window=period, min_periods=period).mean()
    kama_values = kama.values
    sc_values = sc.values
    close_values = close.values
    
    for i in range(period, len(close)):
        if pd.isna(kama_values[i-1]) or pd.isna(sc_values[i]) or pd.isna(close_values[i]):
            continue
        kama_values[i] = kama_values[i-1] + sc_values[i] * (close_values[i] - kama_values[i-1])
    
    kama = pd.Series(kama_values, index=close.index)
    kama_slope = kama.diff() / close
    kama_slope.name = "kama_slope"
    return kama_slope


def feature_chaikin_money_flow(df: DataFrame) -> Series:
    """
    Chaikin Money Flow (20-period): Volume-weighted price flow indicator.
    
    CMF = sum(MFV over 20d) / sum(volume over 20d).
    MFV = Money Flow Multiplier * volume.
    Values: -1.0 = distribution, +1.0 = accumulation.
    Clipped to [-1, 1] for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    range_hl = (high - low).replace(0, np.nan)
    mfm = (2 * close - high - low) / range_hl
    mfv = mfm * volume
    mfv_sum_20d = mfv.rolling(window=20, min_periods=1).sum()
    volume_sum_20d = volume.rolling(window=20, min_periods=1).sum().replace(0, np.nan)
    cmf = (mfv_sum_20d / volume_sum_20d).clip(-1.0, 1.0)
    cmf.name = "chaikin_money_flow"
    return cmf


def feature_mfi(df: DataFrame) -> Series:
    """
    MFI: Money Flow Index (14-period), volume-weighted RSI, centered to [-1, +1].
    
    Like RSI but uses volume-weighted typical price.
    Centered: (mfi - 50) / 50.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    delta_tp = typical_price.diff()
    positive_flow = money_flow.where(delta_tp > 0, 0).rolling(window=14, min_periods=1).sum()
    negative_flow = money_flow.where(delta_tp < 0, 0).rolling(window=14, min_periods=1).sum()
    negative_flow = negative_flow.replace(0, 1e-10)
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    mfi_centered = (mfi - 50) / 50
    mfi_centered.name = "mfi"
    return mfi_centered


def feature_tsi(df: DataFrame) -> Series:
    """
    TSI: True Strength Index (25/13 periods), centered to [-1, +1].
    
    Double-smoothed momentum oscillator using EMAs.
    More responsive than RSI, less noisy.
    Centered: (tsi - 0) / 100 (TSI ranges roughly -100 to +100).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    price_change = close.diff()
    abs_price_change = price_change.abs()
    
    # Double smoothing with EMAs
    pcs = price_change.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    pcds = abs_price_change.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    
    tsi = (pcs / (pcds + 1e-10)) * 100
    tsi_centered = tsi / 100  # Normalize to roughly [-1, +1] range
    tsi_centered.name = "tsi"
    return tsi_centered


def feature_trend_residual(df: DataFrame) -> Series:
    """
    Trend Residual: Deviation from linear trend over 50 days.
    
    Measures how much price deviates from fitted linear trend.
    Calculated using linear regression on last 50 prices.
    Residual = (actual - fitted) / actual.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    resid = close.rolling(window=50, min_periods=50).apply(
        _trend_residual_window, raw=True
    )
    resid.name = "trend_residual"
    return resid


# ============================================================================
# BLOCK ML-3.1: Additional Momentum Oscillators (4 features)
# ============================================================================


def feature_ultimate_oscillator(df: DataFrame) -> Series:
    """
    Ultimate Oscillator: Combines momentum across 3 timeframes (7, 14, 28 periods).
    
    Calculates weighted average of momentum across short, medium, and long timeframes.
    Formula: UO = 100 * (4*BP7 + 2*BP14 + BP28) / (4*TR7 + 2*TR14 + TR28)
    Where BP = Buying Pressure, TR = True Range.
    Normalized to [0, 100], then centered: (uo - 50) / 50.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    prev_close = close.shift(1)
    
    # Buying Pressure = close - min(low, prev_close)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    
    # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate weighted averages for 7, 14, 28 periods
    bp7 = bp.rolling(window=7, min_periods=1).sum()
    bp14 = bp.rolling(window=14, min_periods=1).sum()
    bp28 = bp.rolling(window=28, min_periods=1).sum()
    tr7 = tr.rolling(window=7, min_periods=1).sum()
    tr14 = tr.rolling(window=14, min_periods=1).sum()
    tr28 = tr.rolling(window=28, min_periods=1).sum()
    
    # Ultimate Oscillator
    denominator = 4 * tr7 + 2 * tr14 + tr28
    denominator = denominator.replace(0, np.nan)
    uo = 100 * (4 * bp7 + 2 * bp14 + bp28) / denominator
    
    # Center to [-1, +1] range
    uo_centered = (uo - 50) / 50
    uo_centered.name = "ultimate_oscillator"
    return uo_centered


def feature_awesome_oscillator(df: DataFrame) -> Series:
    """
    Awesome Oscillator: 5-period SMA - 34-period SMA, normalized by price.
    
    Measures short-term vs medium-term momentum.
    Calculated as: (SMA5 - SMA34) / close.
    Positive = short-term momentum above medium-term (bullish).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma5 = close.rolling(window=5, min_periods=1).mean()
    sma34 = close.rolling(window=34, min_periods=1).mean()
    ao = (sma5 - sma34) / close
    ao.name = "awesome_oscillator"
    return ao


def feature_momentum_20d(df: DataFrame) -> Series:
    """
    Momentum 20d: Price momentum (close - close[20]), normalized by price.
    
    Absolute price change over 20 days, normalized by current price.
    Different from ROC20 as it's absolute change, not percentage.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = (close - close.shift(20)) / close
    momentum.name = "momentum_20d"
    return momentum


def feature_momentum_50d(df: DataFrame) -> Series:
    """
    Momentum 50d: Price momentum (close - close[50]), normalized by price.
    
    Absolute price change over 50 days, normalized by current price.
    Captures medium-term momentum.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = (close - close.shift(50)) / close
    momentum.name = "momentum_50d"
    return momentum


# ============================================================================
# BLOCK ML-3.2: Momentum Quality Features (10 features)
# ============================================================================


def feature_momentum_divergence(df: DataFrame) -> Series:
    """
    Momentum Divergence: RSI/price divergence signal.
    
    Detects when price makes new highs/lows but RSI doesn't (divergence).
    Calculated as: rolling correlation between RSI14 changes and price changes over 20 days.
    Negative correlation = divergence (bearish), positive = convergence (bullish).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    rsi14 = feature_rsi14(df)
    
    # Calculate changes
    rsi_changes = rsi14.diff()
    price_changes = close.pct_change()
    
    # Rolling correlation over 20 days - use DataFrame.corr() method
    combined = pd.DataFrame({'rsi': rsi_changes, 'price': price_changes})
    
    # Calculate rolling correlation manually
    def rolling_corr(series1, series2, window=20):
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            window1 = series1.iloc[i-window:i]
            window2 = series2.iloc[i-window:i]
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            if window1.std() == 0 or window2.std() == 0:
                result.iloc[i] = np.nan
                continue
            corr = window1.corr(window2)
            result.iloc[i] = corr if not pd.isna(corr) else np.nan
        return result
    
    divergence = rolling_corr(rsi_changes, price_changes, window=20)
    divergence.name = "momentum_divergence"
    return divergence


def feature_momentum_consistency(df: DataFrame) -> Series:
    """
    Momentum Consistency: Consistency of momentum across timeframes.
    
    Measures how consistent momentum is across RSI7, RSI14, RSI21.
    Calculated as: 1 - std(RSI7, RSI14, RSI21) / mean(abs(RSI7, RSI14, RSI21)).
    Higher values = more consistent momentum across timeframes.
    No clipping - let ML learn the distribution.
    """
    rsi7 = feature_rsi7(df)
    rsi14 = feature_rsi14(df)
    rsi21 = feature_rsi21(df)
    
    # Stack RSI values and calculate consistency
    rsi_stack = pd.concat([rsi7, rsi14, rsi21], axis=1)
    rsi_std = rsi_stack.std(axis=1)
    rsi_mean_abs = rsi_stack.abs().mean(axis=1)
    
    # Consistency = 1 - (std / mean_abs), handle division by zero
    consistency = 1 - (rsi_std / (rsi_mean_abs + 1e-10))
    consistency.name = "momentum_consistency"
    return consistency


def feature_momentum_acceleration(df: DataFrame) -> Series:
    """
    Momentum Acceleration: Rate of change of momentum.
    
    Second derivative of price momentum.
    Calculated as: diff(ROC10) - rate of change of rate of change.
    Positive = momentum accelerating, negative = momentum decelerating.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    roc10 = (close - close.shift(10)) / close.shift(10)
    acceleration = roc10.diff(5)
    acceleration.name = "momentum_acceleration"
    return acceleration


def feature_momentum_strength(df: DataFrame) -> Series:
    """
    Momentum Strength: Strength of momentum signal.
    
    Combines RSI14 magnitude and MACD histogram magnitude.
    Calculated as: sqrt(RSI14^2 + MACD_hist_norm^2).
    Higher values = stronger momentum signal.
    No clipping - let ML learn the distribution.
    """
    rsi14 = feature_rsi14(df)
    macd_hist = feature_macd_histogram_normalized(df)
    
    # Combine RSI and MACD magnitudes
    strength = np.sqrt(rsi14 ** 2 + macd_hist ** 2)
    strength.name = "momentum_strength"
    return strength


def feature_momentum_persistence(df: DataFrame) -> Series:
    """
    Momentum Persistence: Momentum persistence measure.
    
    Measures how many consecutive days momentum has been in same direction.
    Calculated as: rolling count of consecutive positive/negative ROC10 values.
    Normalized by dividing by 20 (max window).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    roc10 = (close - close.shift(10)) / close.shift(10)
    
    # Calculate sign of ROC10
    roc_sign = np.sign(roc10)
    
    # For each point, count how many in rolling window have same sign
    def count_same_sign(window_values):
        if len(window_values) < 2:
            return 0.0
        current_sign = np.sign(window_values[-1])
        if current_sign == 0:
            return 0.0
        same_sign_count = (np.sign(window_values[:-1]) == current_sign).sum()
        return same_sign_count / len(window_values[:-1])
    
    persistence = roc_sign.rolling(window=20, min_periods=5).apply(
        count_same_sign, raw=True
    )
    persistence.name = "momentum_persistence"
    return persistence


def feature_momentum_exhaustion(df: DataFrame) -> Series:
    """
    Momentum Exhaustion: Momentum exhaustion indicator.
    
    Detects when momentum is extreme but starting to fade.
    Calculated as: RSI14 near extremes (>70 or <30) but RSI momentum turning negative/positive.
    Values: positive = exhaustion in overbought, negative = exhaustion in oversold.
    No clipping - let ML learn the distribution.
    """
    rsi14 = feature_rsi14(df)
    rsi_mom = feature_rsi_momentum(df)
    
    # RSI14 centered is in [-1, +1], so >0.4 = overbought, < -0.4 = oversold
    overbought = (rsi14 > 0.4) & (rsi_mom < 0)  # RSI high but momentum fading
    oversold = (rsi14 < -0.4) & (rsi_mom > 0)    # RSI low but momentum recovering
    
    exhaustion = pd.Series(0.0, index=rsi14.index)
    exhaustion[overbought] = rsi14[overbought]  # Positive = overbought exhaustion
    exhaustion[oversold] = rsi14[oversold]     # Negative = oversold exhaustion
    exhaustion.name = "momentum_exhaustion"
    return exhaustion


def feature_momentum_reversal_signal(df: DataFrame) -> Series:
    """
    Momentum Reversal Signal: Momentum reversal signal.
    
    Detects when momentum indicators are reversing direction.
    Calculated as: MACD histogram turning from positive to negative (or vice versa).
    Positive = bullish reversal, negative = bearish reversal.
    No clipping - let ML learn the distribution.
    """
    macd_hist = feature_macd_histogram_normalized(df)
    
    # Detect reversals: histogram crosses zero
    prev_hist = macd_hist.shift(1)
    reversal = pd.Series(0.0, index=macd_hist.index)
    
    # Bullish reversal: negative to positive
    bullish_reversal = (prev_hist < 0) & (macd_hist > 0)
    # Bearish reversal: positive to negative
    bearish_reversal = (prev_hist > 0) & (macd_hist < 0)
    
    reversal[bullish_reversal] = macd_hist[bullish_reversal]
    reversal[bearish_reversal] = macd_hist[bearish_reversal]
    reversal.name = "momentum_reversal_signal"
    return reversal


def feature_momentum_cross(df: DataFrame) -> Series:
    """
    Momentum Cross: Momentum crossover signal.
    
    Detects when short-term momentum crosses above/below medium-term momentum.
    Calculated as: ROC5 - ROC20 (short-term minus medium-term).
    Positive = short-term momentum above medium-term (bullish cross).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    roc5 = (close - close.shift(5)) / close.shift(5)
    roc20 = (close - close.shift(20)) / close.shift(20)
    cross = roc5 - roc20
    cross.name = "momentum_cross"
    return cross


def feature_momentum_rank(df: DataFrame) -> Series:
    """
    Momentum Rank: Momentum rank vs historical.
    
    Percentile rank of current momentum (ROC10) vs last 252 days.
    Values: 0.0 = lowest momentum, 1.0 = highest momentum.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    roc10 = (close - close.shift(10)) / close.shift(10)
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    # Note: rank(pct=True) gives percentile where value sits, which is close to the original logic
    rank = roc10.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    rank = rank.fillna(0.5)
    rank = rank.clip(0.0, 1.0)
    rank.name = "momentum_rank"
    return rank


def feature_momentum_regime(df: DataFrame) -> Series:
    """
    Momentum Regime: Momentum regime classification (strong/weak/neutral).
    
    Classifies momentum into regimes based on multiple indicators.
    Calculated as: weighted combination of RSI14, MACD histogram, and ROC20.
    Values: >0.3 = strong bullish, <-0.3 = strong bearish, else = neutral.
    No clipping - let ML learn the distribution.
    """
    rsi14 = feature_rsi14(df)
    macd_hist = feature_macd_histogram_normalized(df)
    close = _get_close_series(df)
    roc20 = (close - close.shift(20)) / close.shift(20)
    
    # Normalize ROC20 to similar scale (clip to [-1, 1] for combination)
    roc20_norm = roc20.clip(-1.0, 1.0)
    
    # Weighted combination: RSI (40%), MACD (40%), ROC (20%)
    regime = 0.4 * rsi14 + 0.4 * macd_hist + 0.2 * roc20_norm
    regime.name = "momentum_regime"
    return regime


# ============================================================================
# BLOCK ML-3.3: Additional Momentum Variants (4 features)
# ============================================================================


def feature_momentum_5d(df: DataFrame) -> Series:
    """
    Momentum 5d: 5-day price momentum (close - close[5]), normalized by price.
    
    Very short-term momentum indicator.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = (close - close.shift(5)) / close
    momentum.name = "momentum_5d"
    return momentum


def feature_momentum_15d(df: DataFrame) -> Series:
    """
    Momentum 15d: 15-day price momentum (close - close[15]), normalized by price.
    
    Short-term momentum indicator.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = (close - close.shift(15)) / close
    momentum.name = "momentum_15d"
    return momentum


def feature_momentum_30d(df: DataFrame) -> Series:
    """
    Momentum 30d: 30-day price momentum (close - close[30]), normalized by price.
    
    Medium-term momentum indicator.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = (close - close.shift(30)) / close
    momentum.name = "momentum_30d"
    return momentum


def feature_momentum_vs_price(df: DataFrame) -> Series:
    """
    Momentum vs Price: Momentum vs price divergence.
    
    Measures divergence between momentum (ROC10) and price trend.
    Calculated as: rolling correlation between ROC10 and price changes over 20 days.
    Negative = momentum diverging from price (potential reversal).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    roc10 = (close - close.shift(10)) / close.shift(10)
    price_changes = close.pct_change()
    
    # Rolling correlation over 20 days - use DataFrame.corr() method
    def rolling_corr(series1, series2, window=20):
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            window1 = series1.iloc[i-window:i]
            window2 = series2.iloc[i-window:i]
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            if window1.std() == 0 or window2.std() == 0:
                result.iloc[i] = np.nan
                continue
            corr = window1.corr(window2)
            result.iloc[i] = corr if not pd.isna(corr) else np.nan
        return result
    
    divergence = rolling_corr(roc10, price_changes, window=20)
    divergence.name = "momentum_vs_price"
    return divergence


# ============================================================================
# BLOCK ML-4.1: Volatility Regime Detection (10 features)
# ============================================================================


def feature_volatility_regime(df: DataFrame) -> Series:
    """
    Volatility Regime: Current ATR percentile (0-100) over 252 days.
    
    Measures where current volatility sits relative to last year's distribution.
    Calculated as: percentile rank of current ATR14 vs last 252 days.
    Values: 0.0 = lowest volatility, 1.0 = highest volatility.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    regime = atr14.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    regime = regime.fillna(0.5)
    regime = regime.clip(0.0, 1.0)
    regime.name = "volatility_regime"
    return regime


def feature_volatility_trend(df: DataFrame) -> Series:
    """
    Volatility Trend: ATR slope (increasing/decreasing volatility).
    
    Measures rate of change of ATR14, normalized by current ATR.
    Calculated as: ATR14.diff(10) / ATR14.
    Positive = volatility increasing, negative = volatility decreasing.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # Calculate slope (10-day change normalized by current ATR)
    trend = atr14.diff(10) / (atr14 + 1e-10)
    trend.name = "volatility_trend"
    return trend


def feature_bb_squeeze(df: DataFrame) -> Series:
    """
    BB Squeeze: Bollinger Band squeeze indicator (low volatility).
    
    Measures how tight Bollinger Bands are relative to their average width.
    Calculated as: current BB width / average BB width over 20 days.
    Values < 1.0 = squeeze (low volatility), > 1.0 = expansion (high volatility).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Calculate Bollinger Bands
    mid = close.rolling(window=20, min_periods=1).mean()
    std = close.rolling(window=20, min_periods=1).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    
    # BB Width
    mid_safe = mid.replace(0, np.nan)
    bb_width = (upper - lower) / mid_safe
    
    # Compare to average BB width over 20 days
    avg_bb_width = bb_width.rolling(window=20, min_periods=1).mean()
    squeeze = bb_width / (avg_bb_width + 1e-10)
    squeeze.name = "bb_squeeze"
    return squeeze


def feature_bb_expansion(df: DataFrame) -> Series:
    """
    BB Expansion: Bollinger Band expansion indicator (high volatility).
    
    Measures how expanded Bollinger Bands are relative to their average width.
    Calculated as: current BB width / average BB width over 20 days.
    Values > 1.0 = expansion (high volatility), < 1.0 = squeeze (low volatility).
    No clipping - let ML learn the distribution.
    """
    # Same calculation as bb_squeeze, but interpretation is different
    # We'll use the same value but name it differently for clarity
    squeeze = feature_bb_squeeze(df)
    expansion = squeeze.copy()
    expansion.name = "bb_expansion"
    return expansion


def feature_atr_ratio_20d(df: DataFrame) -> Series:
    """
    ATR Ratio 20d: Current ATR / 20-day ATR average.
    
    Measures current volatility relative to recent average.
    Values > 1.0 = above average volatility, < 1.0 = below average.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # 20-day average of ATR
    atr_avg_20d = atr14.rolling(window=20, min_periods=1).mean()
    
    # Ratio
    ratio = atr14 / (atr_avg_20d + 1e-10)
    ratio.name = "atr_ratio_20d"
    return ratio


def feature_atr_ratio_252d(df: DataFrame) -> Series:
    """
    ATR Ratio 252d: Current ATR / 252-day ATR average.
    
    Measures current volatility relative to yearly average.
    Values > 1.0 = above yearly average, < 1.0 = below yearly average.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # 252-day average of ATR
    atr_avg_252d = atr14.rolling(window=252, min_periods=20).mean()
    
    # Ratio
    ratio = atr14 / (atr_avg_252d + 1e-10)
    ratio.name = "atr_ratio_252d"
    return ratio


def feature_volatility_percentile_20d(df: DataFrame) -> Series:
    """
    Volatility Percentile 20d: ATR percentile over 20 days.
    
    Percentile rank of current ATR14 vs last 20 days.
    Values: 0.0 = lowest, 1.0 = highest.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # Percentile rank over 20 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    percentile = atr14.rolling(window=20, min_periods=5).rank(pct=True, method='average')
    percentile = percentile.fillna(0.5)
    percentile = percentile.clip(0.0, 1.0)
    percentile.name = "volatility_percentile_20d"
    return percentile


def feature_volatility_percentile_252d(df: DataFrame) -> Series:
    """
    Volatility Percentile 252d: ATR percentile over 252 days.
    
    Percentile rank of current ATR14 vs last 252 days (1 year).
    Values: 0.0 = lowest, 1.0 = highest.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    percentile = atr14.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    percentile = percentile.fillna(0.5)
    percentile = percentile.clip(0.0, 1.0)
    percentile.name = "volatility_percentile_252d"
    return percentile


def feature_high_volatility_flag(df: DataFrame) -> Series:
    """
    High Volatility Flag: Binary flag (ATR > 75th percentile).
    
    Binary indicator: 1 if ATR14 is above 75th percentile over 252 days, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # 75th percentile over 252 days
    percentile_75 = atr14.rolling(window=252, min_periods=20).quantile(0.75)
    
    # Binary flag
    flag = (atr14 > percentile_75).astype(float)
    flag.name = "high_volatility_flag"
    return flag


def feature_low_volatility_flag(df: DataFrame) -> Series:
    """
    Low Volatility Flag: Binary flag (ATR < 25th percentile).
    
    Binary indicator: 1 if ATR14 is below 25th percentile over 252 days, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # 25th percentile over 252 days
    percentile_25 = atr14.rolling(window=252, min_periods=20).quantile(0.25)
    
    # Binary flag
    flag = (atr14 < percentile_25).astype(float)
    flag.name = "low_volatility_flag"
    return flag


# ============================================================================
# BLOCK ML-4.2: Advanced Volatility Estimators (5 features)
# ============================================================================


def feature_parkinson_volatility(df: DataFrame) -> Series:
    """
    Parkinson Volatility: High-low based volatility estimator.
    
    Uses high-low range to estimate volatility, more efficient than close-based.
    Formula: sqrt((1/(4*ln(2))) * (ln(high/low))^2)
    Normalized by dividing by close price.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Parkinson volatility
    hl_ratio = high / (low + 1e-10)
    parkinson = np.sqrt((1 / (4 * np.log(2))) * (np.log(hl_ratio) ** 2))
    
    # Normalize by price
    parkinson_norm = parkinson / close
    parkinson_norm.name = "parkinson_volatility"
    return parkinson_norm


def feature_garman_klass_volatility(df: DataFrame) -> Series:
    """
    Garman-Klass Volatility: OHLC-based volatility estimator.
    
    Uses open, high, low, close for more efficient volatility estimation.
    Formula: 0.5 * (ln(high/low))^2 - (2*ln(2)-1) * (ln(close/open))^2
    Already scale-invariant (uses log ratios), no normalization needed.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    openp = _get_open_series(df)
    close = _get_close_series(df)
    
    # Garman-Klass volatility
    hl_term = 0.5 * (np.log(high / (low + 1e-10)) ** 2)
    co_term = (2 * np.log(2) - 1) * (np.log(close / (openp + 1e-10)) ** 2)
    gk = hl_term - co_term
    gk = np.sqrt(np.maximum(gk, 0))  # Ensure non-negative
    
    # Already scale-invariant from log ratios - no need to divide by close
    gk.name = "garman_klass_volatility"
    return gk


def feature_rogers_satchell_volatility(df: DataFrame) -> Series:
    """
    Rogers-Satchell Volatility: OHLC volatility with drift correction.
    
    Accounts for drift in price movements.
    Formula: sqrt(ln(high/close) * ln(high/open) + ln(low/close) * ln(low/open))
    Already scale-invariant (uses log ratios), no normalization needed.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    openp = _get_open_series(df)
    close = _get_close_series(df)
    
    # Rogers-Satchell volatility
    term1 = np.log(high / (close + 1e-10)) * np.log(high / (openp + 1e-10))
    term2 = np.log(low / (close + 1e-10)) * np.log(low / (openp + 1e-10))
    rs = term1 + term2
    rs = np.sqrt(np.maximum(rs, 0))  # Ensure non-negative
    
    # Already scale-invariant from log ratios - no need to divide by close
    rs.name = "rogers_satchell_volatility"
    return rs


def feature_yang_zhang_volatility(df: DataFrame) -> Series:
    """
    Yang-Zhang Volatility: Overnight + intraday volatility.
    
    Combines overnight (close-to-open) and intraday (open-to-close) volatility.
    Formula: sqrt(overnight_vol^2 + k * intraday_vol^2 + (1-k) * rogers_satchell^2)
    where k = 0.34 (optimal for daily data).
    Already scale-invariant (uses log returns), no normalization needed.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    openp = _get_open_series(df)
    close = _get_close_series(df)
    
    # Overnight volatility (close-to-open)
    prev_close = close.shift(1)
    overnight_vol = np.log(openp / (prev_close + 1e-10))
    overnight_var = overnight_vol.rolling(window=20, min_periods=1).var()
    
    # Intraday volatility (open-to-close)
    intraday_vol = np.log(close / (openp + 1e-10))
    intraday_var = intraday_vol.rolling(window=20, min_periods=1).var()
    
    # Rogers-Satchell component
    term1 = np.log(high / (close + 1e-10)) * np.log(high / (openp + 1e-10))
    term2 = np.log(low / (close + 1e-10)) * np.log(low / (openp + 1e-10))
    rs_term = term1 + term2
    rs_var = rs_term.rolling(window=20, min_periods=1).mean()
    
    # Yang-Zhang volatility (k = 0.34 for daily data)
    k = 0.34
    yz_var = overnight_var + k * intraday_var + (1 - k) * rs_var
    yz = np.sqrt(np.maximum(yz_var, 0))
    
    # Already scale-invariant from log returns - no need to divide by close
    yz.name = "yang_zhang_volatility"
    return yz


def feature_volatility_clustering(df: DataFrame) -> Series:
    """
    Volatility Clustering: Volatility clustering measure.
    
    Measures autocorrelation of volatility (volatility tends to cluster).
    Calculated as: correlation between current and lagged volatility over 20 days.
    Higher values = stronger clustering (volatility persistence).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Calculate daily volatility (absolute returns)
    daily_vol = close.pct_change().abs()
    
    # Lagged volatility
    vol_lag = daily_vol.shift(1)
    
    # Rolling correlation over 20 days
    def rolling_corr(series1, series2, window=20):
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            window1 = series1.iloc[i-window:i]
            window2 = series2.iloc[i-window:i]
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            if window1.std() == 0 or window2.std() == 0:
                result.iloc[i] = np.nan
                continue
            corr = window1.corr(window2)
            result.iloc[i] = corr if not pd.isna(corr) else np.nan
        return result
    
    clustering = rolling_corr(daily_vol, vol_lag, window=20)
    clustering.name = "volatility_clustering"
    return clustering


# ============================================================================
# BLOCK ML-4.3: Realized Volatility (3 features)
# ============================================================================


def feature_realized_volatility_5d(df: DataFrame) -> Series:
    """
    Realized Volatility 5d: 5-day realized volatility.
    
    Sum of squared returns over 5 days, annualized.
    Calculated as: sqrt(sum(returns^2) * 252 / 5).
    Already in percentage terms (from percentage returns), no normalization needed.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Daily returns
    returns = close.pct_change()
    
    # Sum of squared returns over 5 days
    realized_var = (returns ** 2).rolling(window=5, min_periods=1).sum()
    
    # Annualize (252 trading days per year)
    realized_vol = np.sqrt(realized_var * 252 / 5)
    
    # Already in percentage terms - no need to divide by close
    realized_vol.name = "realized_volatility_5d"
    return realized_vol


def feature_realized_volatility_20d(df: DataFrame) -> Series:
    """
    Realized Volatility 20d: 20-day realized volatility.
    
    Sum of squared returns over 20 days, annualized.
    Calculated as: sqrt(sum(returns^2) * 252 / 20).
    Already in percentage terms (from percentage returns), no normalization needed.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Daily returns
    returns = close.pct_change()
    
    # Sum of squared returns over 20 days
    realized_var = (returns ** 2).rolling(window=20, min_periods=1).sum()
    
    # Annualize (252 trading days per year)
    realized_vol = np.sqrt(realized_var * 252 / 20)
    
    # Already in percentage terms - no need to divide by close
    realized_vol.name = "realized_volatility_20d"
    return realized_vol


def feature_volatility_regime_change(df: DataFrame) -> Series:
    """
    Volatility Regime Change: Binary flag for volatility regime change.
    
    Detects when volatility regime has changed recently.
    Calculated as: 1 if ATR percentile changed by >20% in last 10 days, else 0.
    Binary flag (0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate ATR14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    percentile = atr14.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    percentile = percentile.fillna(0.5)
    
    # Check for regime change (>20% change in percentile over 10 days)
    percentile_change = (percentile - percentile.shift(10)).abs()
    regime_change = (percentile_change > 0.2).astype(float)
    regime_change.name = "volatility_regime_change"
    return regime_change


# ============================================================================
# BLOCK ML-5.1: ADX & Trend Strength (5 features)
# ============================================================================


def _wilder_smooth(series: Series, period: int = 14) -> Series:
    """Apply Wilder's smoothing method for ADX calculation."""
    smoothed = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i == 0:
            smoothed.iloc[i] = series.iloc[i]
        elif i < period:
            smoothed.iloc[i] = series.iloc[:i+1].mean()
        else:
            smoothed.iloc[i] = smoothed.iloc[i-1] * (period - 1) / period + series.iloc[i] / period
    return smoothed


def _calculate_adx(df: DataFrame, period: int = 14) -> Series:
    """Helper function to calculate ADX for a given period."""
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high - prev_high
    minus_dm = prev_low - low
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    # Wilder's smoothing
    plus_dm_smooth = _wilder_smooth(plus_dm, period=period)
    minus_dm_smooth = _wilder_smooth(minus_dm, period=period)
    tr_smooth = _wilder_smooth(tr, period=period)
    
    # DI lines
    tr_safe = tr_smooth.replace(0, np.nan)
    plus_di = 100 * (plus_dm_smooth / tr_safe)
    minus_di = 100 * (minus_dm_smooth / tr_safe)
    
    # DX
    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum_safe
    
    # ADX (smoothed DX with EMA)
    adx = dx.ewm(span=period, adjust=False).mean()
    
    # Normalize to [0, 1]
    adx_norm = (adx / 100).clip(0.0, 1.0)
    return adx_norm


def feature_adx14(df: DataFrame) -> Series:
    """
    ADX14: Average Directional Index (14-period), normalized to [0, 1].
    
    Measures trend strength independent of direction.
    Values: 0.0 = no trend, 1.0 = very strong trend.
    Already normalized to [0, 1]. Clipped for safety.
    """
    adx = _calculate_adx(df, period=14)
    adx.name = "adx14"
    return adx


def feature_adx20(df: DataFrame) -> Series:
    """
    ADX20: Average Directional Index (20-period), normalized to [0, 1].
    
    Medium-term trend strength indicator.
    Values: 0.0 = no trend, 1.0 = very strong trend.
    Already normalized to [0, 1]. Clipped for safety.
    """
    adx = _calculate_adx(df, period=20)
    adx.name = "adx20"
    return adx


def feature_adx50(df: DataFrame) -> Series:
    """
    ADX50: Average Directional Index (50-period), normalized to [0, 1].
    
    Long-term trend strength indicator.
    Values: 0.0 = no trend, 1.0 = very strong trend.
    Already normalized to [0, 1]. Clipped for safety.
    """
    adx = _calculate_adx(df, period=50)
    adx.name = "adx50"
    return adx


def feature_trend_strength_20d(df: DataFrame) -> Series:
    """
    Trend Strength 20d: ADX over 20 days (rolling average).
    
    Smoothed trend strength over 20-day window.
    Calculated as: ADX14.rolling(20).mean().
    Already normalized to [0, 1]. Clipped for safety.
    """
    adx14 = feature_adx14(df)
    trend_strength = adx14.rolling(window=20, min_periods=1).mean()
    trend_strength = trend_strength.clip(0.0, 1.0)
    trend_strength.name = "trend_strength_20d"
    return trend_strength


def feature_trend_strength_50d(df: DataFrame) -> Series:
    """
    Trend Strength 50d: ADX over 50 days (rolling average).
    
    Smoothed trend strength over 50-day window.
    Calculated as: ADX14.rolling(50).mean().
    Already normalized to [0, 1]. Clipped for safety.
    """
    adx14 = feature_adx14(df)
    trend_strength = adx14.rolling(window=50, min_periods=1).mean()
    trend_strength = trend_strength.clip(0.0, 1.0)
    trend_strength.name = "trend_strength_50d"
    return trend_strength


# ============================================================================
# BLOCK ML-5.2: MA Alignment & Crossovers (6 features)
# ============================================================================


def feature_ema_alignment(df: DataFrame) -> Series:
    """
    EMA Alignment: All EMAs aligned (bullish/bearish/neutral): -1/0/1.
    
    Checks if EMA20 > EMA50 > EMA200 (bullish) or EMA20 < EMA50 < EMA200 (bearish).
    Returns: 1 = bullish alignment, -1 = bearish alignment, 0 = neutral/mixed.
    Already normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # Check alignment
    bullish = (ema20 > ema50) & (ema50 > ema200)
    bearish = (ema20 < ema50) & (ema50 < ema200)
    
    alignment = pd.Series(0.0, index=close.index)
    alignment[bullish] = 1.0
    alignment[bearish] = -1.0
    alignment = alignment.clip(-1.0, 1.0)
    alignment.name = "ema_alignment"
    return alignment


def feature_ma_crossover_bullish(df: DataFrame) -> Series:
    """
    MA Crossover Bullish: Binary flag (SMA20 > SMA50 > SMA200).
    
    Binary indicator: 1 if all SMAs are aligned bullish, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    sma20 = close.rolling(window=20, min_periods=1).mean()
    sma50 = close.rolling(window=50, min_periods=1).mean()
    sma200 = close.rolling(window=200, min_periods=1).mean()
    
    bullish = ((sma20 > sma50) & (sma50 > sma200)).astype(float)
    bullish.name = "ma_crossover_bullish"
    return bullish


def feature_ma_crossover_bearish(df: DataFrame) -> Series:
    """
    MA Crossover Bearish: Binary flag (SMA20 < SMA50 < SMA200).
    
    Binary indicator: 1 if all SMAs are aligned bearish, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    sma20 = close.rolling(window=20, min_periods=1).mean()
    sma50 = close.rolling(window=50, min_periods=1).mean()
    sma200 = close.rolling(window=200, min_periods=1).mean()
    
    bearish = ((sma20 < sma50) & (sma50 < sma200)).astype(float)
    bearish.name = "ma_crossover_bearish"
    return bearish


def feature_price_vs_all_mas(df: DataFrame) -> Series:
    """
    Price vs All MAs: Count of MAs price is above, normalized to [0, 1].
    
    Counts how many of (SMA20, SMA50, SMA200, EMA20, EMA50, EMA200) price is above.
    Normalized by dividing by 6 (max count).
    Values: 0.0 = below all, 1.0 = above all.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    sma20 = close.rolling(window=20, min_periods=1).mean()
    sma50 = close.rolling(window=50, min_periods=1).mean()
    sma200 = close.rolling(window=200, min_periods=1).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # Count how many MAs price is above
    count = (
        (close > sma20).astype(int) +
        (close > sma50).astype(int) +
        (close > sma200).astype(int) +
        (close > ema20).astype(int) +
        (close > ema50).astype(int) +
        (close > ema200).astype(int)
    )
    
    # Normalize by 6 (max count)
    normalized = count / 6.0
    normalized = normalized.clip(0.0, 1.0)
    normalized.name = "price_vs_all_mas"
    return normalized


def feature_sma_slope_20d(df: DataFrame) -> Series:
    """
    SMA Slope 20d: Slope of 20-day SMA, normalized by price.
    
    Measures rate of change of 20-day SMA.
    Calculated as: sma20.diff(10) / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma20 = close.rolling(window=20, min_periods=1).mean()
    slope = sma20.diff(10) / close
    slope.name = "sma_slope_20d"
    return slope


def feature_sma_slope_50d(df: DataFrame) -> Series:
    """
    SMA Slope 50d: Slope of 50-day SMA, normalized by price.
    
    Measures rate of change of 50-day SMA.
    Calculated as: sma50.diff(10) / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    slope = sma50.diff(10) / close
    slope.name = "sma_slope_50d"
    return slope


# ============================================================================
# BLOCK ML-5.3: EMA Slopes (2 features)
# ============================================================================


def feature_ema_slope_20d(df: DataFrame) -> Series:
    """
    EMA Slope 20d: Slope of 20-day EMA, normalized by price.
    
    Measures rate of change of 20-day EMA.
    Calculated as: ema20.diff(5) / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    ema20 = close.ewm(span=20, adjust=False).mean()
    slope = ema20.diff(5) / close
    slope.name = "ema_slope_20d"
    return slope


def feature_ema_slope_50d(df: DataFrame) -> Series:
    """
    EMA Slope 50d: Slope of 50-day EMA, normalized by price.
    
    Measures rate of change of 50-day EMA.
    Calculated as: ema50.diff(10) / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    ema50 = close.ewm(span=50, adjust=False).mean()
    slope = ema50.diff(10) / close
    slope.name = "ema_slope_50d"
    return slope


# ============================================================================
# BLOCK ML-5.4: Trend Quality Features (9 features)
# ============================================================================


def feature_trend_consistency(df: DataFrame) -> Series:
    """
    Trend Consistency: % of days price above/below MA over lookback.
    
    Measures how consistently price stays on one side of SMA50.
    Calculated as: % of days in last 20 where price > SMA50 (if bullish) or < SMA50 (if bearish).
    Values: 0.0 = inconsistent, 1.0 = very consistent.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Determine trend direction (price vs SMA50)
    above_ma = close > sma50
    
    # For each day, count consistency over last 20 days
    consistency = pd.Series(index=close.index, dtype=float)
    for i in range(20, len(close)):
        window_above = above_ma.iloc[i-20:i]
        if above_ma.iloc[i]:  # Currently above MA
            consistency.iloc[i] = window_above.sum() / 20.0
        else:  # Currently below MA
            consistency.iloc[i] = (~window_above).sum() / 20.0
    
    consistency = consistency.clip(0.0, 1.0)
    consistency.name = "trend_consistency"
    return consistency


def feature_trend_duration(df: DataFrame) -> Series:
    """
    Trend Duration: Days since last trend change, normalized.
    
    Counts consecutive days price has been on same side of SMA50.
    Normalized by dividing by 50 (max reasonable duration).
    Values: 0.0 = just changed, 1.0 = long trend.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Determine if price is above or below MA
    above_ma = close > sma50
    
    # Count consecutive days on same side
    duration = pd.Series(index=close.index, dtype=float)
    current_duration = 0
    current_side = None
    
    for i in range(len(close)):
        if pd.isna(above_ma.iloc[i]):
            duration.iloc[i] = 0
            continue
        
        if current_side is None:
            current_side = above_ma.iloc[i]
            current_duration = 1
        elif above_ma.iloc[i] == current_side:
            current_duration += 1
        else:
            current_side = above_ma.iloc[i]
            current_duration = 1
        
        duration.iloc[i] = current_duration / 50.0  # Normalize by 50
    
    duration = duration.clip(0.0, 1.0)
    duration.name = "trend_duration"
    return duration


def feature_trend_reversal_signal(df: DataFrame) -> Series:
    """
    Trend Reversal Signal: Potential trend reversal indicator.
    
    Detects when price crosses SMA50 with momentum.
    Calculated as: (close - sma50) * (close.shift(5) - sma50.shift(5)) < 0.
    Positive = bullish reversal, negative = bearish reversal.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Check for crossovers
    current_diff = close - sma50
    prev_diff = (close - sma50).shift(5)
    
    # Reversal: sign changed
    reversal = current_diff * prev_diff
    reversal = reversal.where(reversal < 0, 0)  # Only keep negative (sign change)
    reversal = reversal / close  # Normalize by price
    reversal.name = "trend_reversal_signal"
    return reversal


def feature_trend_acceleration(df: DataFrame) -> Series:
    """
    Trend Acceleration: Rate of change of trend slope.
    
    Second derivative of price trend.
    Calculated as: diff(sma50_slope).
    Positive = trend accelerating, negative = trend decelerating.
    No clipping - let ML learn the distribution.
    """
    sma50_slope = feature_sma_slope_50d(df)
    acceleration = sma50_slope.diff(5)
    acceleration.name = "trend_acceleration"
    return acceleration


def feature_trend_divergence(df: DataFrame) -> Series:
    """
    Trend Divergence: Price vs trend divergence.
    
    Measures correlation between price changes and trend (SMA50) changes.
    Calculated as: rolling correlation over 20 days.
    Negative = price diverging from trend.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    price_changes = close.pct_change()
    trend_changes = sma50.pct_change()
    
    # Rolling correlation
    # Optimized: Use numpy arrays for faster access
    def rolling_corr(series1, series2, window=20):
        # Convert to numpy arrays
        values1 = series1.values
        values2 = series2.values
        
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            # Use numpy array slicing (much faster than pandas iloc)
            window1 = values1[i-window:i]
            window2 = values2[i-window:i]
            
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            
            # Use numpy std (faster than pandas)
            std1 = np.std(window1, ddof=0)
            std2 = np.std(window2, ddof=0)
            if std1 == 0 or std2 == 0:
                result.iloc[i] = np.nan
                continue
            
            # Use numpy correlation (faster than pandas)
            corr = np.corrcoef(window1, window2)[0, 1]
            result.iloc[i] = corr if not np.isnan(corr) else np.nan
        return result
    
    divergence = rolling_corr(price_changes, trend_changes, window=20)
    divergence.name = "trend_divergence"
    return divergence


def feature_trend_pullback_strength(df: DataFrame) -> Series:
    """
    Trend Pullback Strength: Strength of pullback to trend.
    
    Measures how far price pulls back toward SMA50 during uptrend.
    Calculated as: (sma50 - close) / sma50 when in uptrend, normalized.
    Positive = pullback strength, negative = above trend.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Pullback strength (only when price is below SMA50 in uptrend)
    pullback = (sma50 - close) / sma50
    pullback = pullback.where(close < sma50, 0)  # Only when below MA
    pullback.name = "trend_pullback_strength"
    return pullback


def feature_trend_breakout_strength(df: DataFrame) -> Series:
    """
    Trend Breakout Strength: Strength of trend breakout.
    
    Measures how far price breaks above/below SMA50.
    Calculated as: (close - sma50) / sma50, normalized.
    Positive = breakout above, negative = breakdown below.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    breakout = (close - sma50) / sma50
    breakout.name = "trend_breakout_strength"
    return breakout


def feature_trend_following_strength(df: DataFrame) -> Series:
    """
    Trend Following Strength: How well price follows trend.
    
    Measures consistency of price direction with trend direction.
    Calculated as: correlation between price momentum and trend momentum.
    Higher values = price follows trend well.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    price_momentum = close.pct_change(5)
    trend_momentum = sma50.pct_change(5)
    
    # Rolling correlation
    def rolling_corr(series1, series2, window=20):
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            window1 = series1.iloc[i-window:i]
            window2 = series2.iloc[i-window:i]
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            if window1.std() == 0 or window2.std() == 0:
                result.iloc[i] = np.nan
                continue
            corr = window1.corr(window2)
            result.iloc[i] = corr if not pd.isna(corr) else np.nan
        return result
    
    following = rolling_corr(price_momentum, trend_momentum, window=20)
    following.name = "trend_following_strength"
    return following


def feature_trend_regime(df: DataFrame) -> Series:
    """
    Trend Regime: Trend regime classification.
    
    Combines ADX, EMA alignment, and price position to classify trend regime.
    Calculated as: weighted combination of ADX14, EMA alignment, and price vs MAs.
    Values: >0.5 = strong uptrend, <-0.5 = strong downtrend, else = ranging.
    No clipping - let ML learn the distribution.
    """
    adx14 = feature_adx14(df)
    ema_align = feature_ema_alignment(df)
    price_vs_mas = feature_price_vs_all_mas(df)
    
    # Normalize price_vs_mas to [-1, 1] range (currently [0, 1])
    price_vs_mas_norm = (price_vs_mas - 0.5) * 2
    
    # Weighted combination: ADX (40%), EMA alignment (30%), Price position (30%)
    regime = 0.4 * adx14 + 0.3 * ema_align + 0.3 * price_vs_mas_norm
    regime.name = "trend_regime"
    return regime


# ============================================================================
# BLOCK ML-6.1: Support/Resistance Levels (6 features)
# ============================================================================


def feature_resistance_level_20d(df: DataFrame) -> Series:
    """
    Resistance Level 20d: Nearest resistance (20-day high), normalized by price.
    
    Calculates the 20-day rolling maximum high as resistance level.
    Normalized by dividing by current price.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    close = _get_close_series(df)
    
    # 20-day rolling maximum high (resistance level)
    resistance = high.rolling(window=20, min_periods=1).max()
    
    # Normalize by current price
    resistance_norm = resistance / close
    resistance_norm.name = "resistance_level_20d"
    return resistance_norm


def feature_resistance_level_50d(df: DataFrame) -> Series:
    """
    Resistance Level 50d: Nearest resistance (50-day high), normalized by price.
    
    Calculates the 50-day rolling maximum high as resistance level.
    Normalized by dividing by current price.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    close = _get_close_series(df)
    
    # 50-day rolling maximum high (resistance level)
    resistance = high.rolling(window=50, min_periods=1).max()
    
    # Normalize by current price
    resistance_norm = resistance / close
    resistance_norm.name = "resistance_level_50d"
    return resistance_norm


def feature_resistance_level_100d(df: DataFrame) -> Series:
    """
    Resistance Level 100d: Nearest resistance (100-day high), normalized by price.
    
    Calculates the 100-day rolling maximum high as resistance level.
    Normalized by dividing by current price.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    close = _get_close_series(df)
    
    # 100-day rolling maximum high (resistance level)
    resistance = high.rolling(window=100, min_periods=1).max()
    
    # Normalize by current price
    resistance_norm = resistance / close
    resistance_norm.name = "resistance_level_100d"
    return resistance_norm


def feature_support_level_20d(df: DataFrame) -> Series:
    """
    Support Level 20d: Nearest support (20-day low), normalized by price.
    
    Calculates the 20-day rolling minimum low as support level.
    Normalized by dividing by current price.
    No clipping - let ML learn the distribution.
    """
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # 20-day rolling minimum low (support level)
    support = low.rolling(window=20, min_periods=1).min()
    
    # Normalize by current price
    support_norm = support / close
    support_norm.name = "support_level_20d"
    return support_norm


def feature_support_level_50d(df: DataFrame) -> Series:
    """
    Support Level 50d: Nearest support (50-day low), normalized by price.
    
    Calculates the 50-day rolling minimum low as support level.
    Normalized by dividing by current price.
    No clipping - let ML learn the distribution.
    """
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # 50-day rolling minimum low (support level)
    support = low.rolling(window=50, min_periods=1).min()
    
    # Normalize by current price
    support_norm = support / close
    support_norm.name = "support_level_50d"
    return support_norm


def feature_support_level_100d(df: DataFrame) -> Series:
    """
    Support Level 100d: Nearest support (100-day low), normalized by price.
    
    Calculates the 100-day rolling minimum low as support level.
    Normalized by dividing by current price.
    No clipping - let ML learn the distribution.
    """
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # 100-day rolling minimum low (support level)
    support = low.rolling(window=100, min_periods=1).min()
    
    # Normalize by current price
    support_norm = support / close
    support_norm.name = "support_level_100d"
    return support_norm


# ============================================================================
# BLOCK ML-6.2: Distance to S/R (4 features)
# ============================================================================


def feature_distance_to_resistance(df: DataFrame) -> Series:
    """
    Distance to Resistance: % distance to nearest resistance.
    
    Calculates minimum distance to any of the resistance levels (20d, 50d, 100d).
    Formula: min((resistance_20d - close)/close, (resistance_50d - close)/close, (resistance_100d - close)/close).
    Positive = distance above, negative = price above resistance (breakout).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    resistance_20d = feature_resistance_level_20d(df) * close
    resistance_50d = feature_resistance_level_50d(df) * close
    resistance_100d = feature_resistance_level_100d(df) * close
    
    # Calculate distances
    dist_20d = (resistance_20d - close) / close
    dist_50d = (resistance_50d - close) / close
    dist_100d = (resistance_100d - close) / close
    
    # Minimum distance (nearest resistance)
    distance = pd.concat([dist_20d, dist_50d, dist_100d], axis=1).min(axis=1)
    distance.name = "distance_to_resistance"
    return distance


def feature_distance_to_support(df: DataFrame) -> Series:
    """
    Distance to Support: % distance to nearest support.
    
    Calculates minimum distance to any of the support levels (20d, 50d, 100d).
    Formula: min((close - support_20d)/close, (close - support_50d)/close, (close - support_100d)/close).
    Positive = distance below, negative = price below support (breakdown).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    support_20d = feature_support_level_20d(df) * close
    support_50d = feature_support_level_50d(df) * close
    support_100d = feature_support_level_100d(df) * close
    
    # Calculate distances
    dist_20d = (close - support_20d) / close
    dist_50d = (close - support_50d) / close
    dist_100d = (close - support_100d) / close
    
    # Minimum distance (nearest support)
    distance = pd.concat([dist_20d, dist_50d, dist_100d], axis=1).min(axis=1)
    distance.name = "distance_to_support"
    return distance


def feature_price_near_resistance(df: DataFrame) -> Series:
    """
    Price Near Resistance: Binary flag (within 2% of resistance).
    
    Binary indicator: 1 if price is within 2% of nearest resistance, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    distance = feature_distance_to_resistance(df)
    
    # Within 2% of resistance (0 to 0.02)
    near_resistance = ((distance >= 0) & (distance <= 0.02)).astype(float)
    near_resistance.name = "price_near_resistance"
    return near_resistance


def feature_price_near_support(df: DataFrame) -> Series:
    """
    Price Near Support: Binary flag (within 2% of support).
    
    Binary indicator: 1 if price is within 2% of nearest support, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    distance = feature_distance_to_support(df)
    
    # Within 2% of support (0 to 0.02)
    near_support = ((distance >= 0) & (distance <= 0.02)).astype(float)
    near_support.name = "price_near_support"
    return near_support


# ============================================================================
# BLOCK ML-6.3: S/R Strength (4 features)
# ============================================================================


def feature_resistance_touches(df: DataFrame) -> Series:
    """
    Resistance Touches: Number of times price touched resistance in last 20 days.
    
    Counts how many times price (high) was within 1% of resistance level.
    Normalized by dividing by 20 (max possible touches).
    Already normalized to [0, 1]. Clipped for safety.
    """
    high = _get_high_series(df)
    close = _get_close_series(df)
    
    # Calculate resistance levels
    resistance_20d = high.rolling(window=20, min_periods=1).max()
    resistance_50d = high.rolling(window=50, min_periods=1).max()
    resistance_100d = high.rolling(window=100, min_periods=1).max()
    
    # Count touches over last 20 days
    # Optimized: Use numpy arrays for faster access
    high_values = high.values
    res_20d_values = resistance_20d.values
    res_50d_values = resistance_50d.values
    res_100d_values = resistance_100d.values
    
    touches = pd.Series(index=close.index, dtype=float)
    for i in range(20, len(close)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_high = high_values[i-20:i]
        window_res_20d = res_20d_values[i-20:i]
        window_res_50d = res_50d_values[i-20:i]
        window_res_100d = res_100d_values[i-20:i]
        
        # Vectorized numpy operations (faster than pandas)
        near_res_20d = np.sum(np.abs(window_high / window_res_20d - 1.0) <= 0.01)
        near_res_50d = np.sum(np.abs(window_high / window_res_50d - 1.0) <= 0.01)
        near_res_100d = np.sum(np.abs(window_high / window_res_100d - 1.0) <= 0.01)
        
        touches.iloc[i] = max(near_res_20d, near_res_50d, near_res_100d) / 20.0
    
    touches = touches.clip(0.0, 1.0)
    touches.name = "resistance_touches"
    return touches


def feature_support_touches(df: DataFrame) -> Series:
    """
    Support Touches: Number of times price touched support in last 20 days.
    
    Counts how many times price (low) was within 1% of support level.
    Normalized by dividing by 20 (max possible touches).
    Already normalized to [0, 1]. Clipped for safety.
    """
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Calculate support levels
    support_20d = low.rolling(window=20, min_periods=1).min()
    support_50d = low.rolling(window=50, min_periods=1).min()
    support_100d = low.rolling(window=100, min_periods=1).min()
    
    # Count touches over last 20 days
    # Optimized: Use numpy arrays for faster access
    low_values = low.values
    sup_20d_values = support_20d.values
    sup_50d_values = support_50d.values
    sup_100d_values = support_100d.values
    
    touches = pd.Series(index=close.index, dtype=float)
    for i in range(20, len(close)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_low = low_values[i-20:i]
        window_sup_20d = sup_20d_values[i-20:i]
        window_sup_50d = sup_50d_values[i-20:i]
        window_sup_100d = sup_100d_values[i-20:i]
        
        # Vectorized numpy operations (faster than pandas)
        near_sup_20d = np.sum(np.abs(window_low / window_sup_20d - 1.0) <= 0.01)
        near_sup_50d = np.sum(np.abs(window_low / window_sup_50d - 1.0) <= 0.01)
        near_sup_100d = np.sum(np.abs(window_low / window_sup_100d - 1.0) <= 0.01)
        
        touches.iloc[i] = max(near_sup_20d, near_sup_50d, near_sup_100d) / 20.0
    
    touches = touches.clip(0.0, 1.0)
    touches.name = "support_touches"
    return touches


def feature_support_resistance_strength(df: DataFrame) -> Series:
    """
    Support/Resistance Strength: Combined support/resistance strength score.
    
    Combines resistance touches and support touches into a strength score.
    Calculated as: (resistance_touches + support_touches) / 2.
    Higher values = stronger S/R levels (more touches).
    Already normalized to [0, 1]. Clipped for safety.
    """
    res_touches = feature_resistance_touches(df)
    sup_touches = feature_support_touches(df)
    
    strength = (res_touches + sup_touches) / 2.0
    strength = strength.clip(0.0, 1.0)
    strength.name = "support_resistance_strength"
    return strength


def feature_donchian_position(df: DataFrame) -> Series:
    """
    Donchian Position: Donchian channel position in [0, 1].
    
    Measures position within 20-period Donchian channel.
    Calculated as: (close - donchian_low_20) / (donchian_high_20 - donchian_low_20).
    Values: 0.0 = at lower channel, 1.0 = at upper channel.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Donchian channels (20-period)
    donchian_high_20 = high.rolling(window=20, min_periods=1).max()
    donchian_low_20 = low.rolling(window=20, min_periods=1).min()
    
    # Position within channel
    channel_range = (donchian_high_20 - donchian_low_20).replace(0, np.nan)
    donchian_pos = (close - donchian_low_20) / channel_range
    donchian_pos = donchian_pos.clip(0.0, 1.0)
    donchian_pos.name = "donchian_position"
    return donchian_pos


# ============================================================================
# BLOCK ML-6.4: Pivot & Fibonacci (4 features)
# ============================================================================


def feature_pivot_point(df: DataFrame) -> Series:
    """
    Pivot Point: Classic pivot point (H+L+C)/3, normalized by price.
    
    Calculates daily pivot point using high, low, close from previous day.
    Formula: (high_prev + low_prev + close_prev) / 3.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Previous day's HLC
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Normalize by current price
    pivot_norm = pivot / close
    pivot_norm.name = "pivot_point"
    return pivot_norm


def feature_pivot_resistance_1(df: DataFrame) -> Series:
    """
    Pivot Resistance 1: First pivot resistance level, normalized by price.
    
    Calculates R1 = 2 * pivot - low_prev.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Previous day's HLC
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # R1 = 2 * pivot - low_prev
    r1 = 2 * pivot - prev_low
    
    # Normalize by current price
    r1_norm = r1 / close
    r1_norm.name = "pivot_resistance_1"
    return r1_norm


def feature_pivot_support_1(df: DataFrame) -> Series:
    """
    Pivot Support 1: First pivot support level, normalized by price.
    
    Calculates S1 = 2 * pivot - high_prev.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Previous day's HLC
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # S1 = 2 * pivot - high_prev
    s1 = 2 * pivot - prev_high
    
    # Normalize by current price
    s1_norm = s1 / close
    s1_norm.name = "pivot_support_1"
    return s1_norm


def feature_fibonacci_retracement(df: DataFrame) -> Series:
    """
    Fibonacci Retracement: Current Fibonacci retracement level.
    
    Calculates which Fibonacci level (0.236, 0.382, 0.5, 0.618, 0.786) price is at.
    Uses 20-day high/low as swing points.
    Returns normalized level (0.0 to 1.0) indicating position in retracement.
    Already normalized to [0, 1]. Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # 20-day swing high and low
    swing_high = high.rolling(window=20, min_periods=1).max()
    swing_low = low.rolling(window=20, min_periods=1).min()
    
    # Range
    range_hl = swing_high - swing_low
    range_hl = range_hl.replace(0, np.nan)
    
    # Fibonacci retracement level
    # If price is in uptrend (close > midpoint), measure from low
    # If price is in downtrend (close < midpoint), measure from high
    midpoint = (swing_high + swing_low) / 2.0
    
    fib_level = pd.Series(index=close.index, dtype=float)
    uptrend = close > midpoint
    downtrend = close < midpoint
    
    # Uptrend: measure retracement from high
    fib_level[uptrend] = (swing_high[uptrend] - close[uptrend]) / range_hl[uptrend]
    
    # Downtrend: measure retracement from low
    fib_level[downtrend] = (close[downtrend] - swing_low[downtrend]) / range_hl[downtrend]
    
    # Neutral: at midpoint
    fib_level[~uptrend & ~downtrend] = 0.5
    
    fib_level = fib_level.clip(0.0, 1.0)
    fib_level.name = "fibonacci_retracement"
    return fib_level


# ============================================================================
# BLOCK ML-7.1: Volume Profile Approximations (6 features)
# ============================================================================


def feature_volume_profile_poc(df: DataFrame) -> Series:
    """
    Volume Profile POC: Point of Control (daily approximation), normalized by price.
    
    POC is the price level with highest volume. For daily data, we approximate using
    typical price (H+L+C)/3 weighted by volume over a rolling window.
    Calculated as: VWAP of typical price over 20 days.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # VWAP of typical price over 20 days (approximates POC)
    price_volume = typical_price * volume
    poc = price_volume.rolling(window=20, min_periods=1).sum() / volume.rolling(window=20, min_periods=1).sum()
    
    # Normalize by current price
    poc_norm = poc / close
    poc_norm.name = "volume_profile_poc"
    return poc_norm


def feature_volume_profile_vah(df: DataFrame) -> Series:
    """
    Volume Profile VAH: Value Area High (approximation), normalized by price.
    
    VAH is the upper bound of value area (70% of volume). We approximate using
    the 70th percentile of typical price weighted by volume over 20 days.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Approximate VAH as 70th percentile of typical price weighted by volume
    def weighted_percentile(window_data, percentile=0.7):
        if len(window_data) < 5:
            return np.nan
        prices = window_data['price'].values
        volumes = window_data['volume'].values
        # Calculate weighted percentile
        sorted_idx = np.argsort(prices)
        sorted_prices = prices[sorted_idx]
        sorted_volumes = volumes[sorted_idx]
        cumsum_vol = np.cumsum(sorted_volumes)
        total_vol = cumsum_vol[-1]
        target_vol = total_vol * percentile
        idx = np.searchsorted(cumsum_vol, target_vol)
        if idx >= len(sorted_prices):
            return sorted_prices[-1]
        return sorted_prices[idx]
    
    # Rolling weighted percentile
    # Optimized: Use numpy arrays for faster access
    price_values = typical_price.values
    vol_values = volume.values
    
    vah = pd.Series(index=close.index, dtype=float)
    for i in range(20, len(close)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_prices = price_values[i-20:i]
        window_volumes = vol_values[i-20:i]
        
        # Create window_data dict for weighted_percentile function
        window_data = pd.DataFrame({'price': window_prices, 'volume': window_volumes})
        vah.iloc[i] = weighted_percentile(window_data, percentile=0.7)
    
    # Normalize by current price
    vah_norm = vah / close
    vah_norm.name = "volume_profile_vah"
    return vah_norm


def feature_volume_profile_val(df: DataFrame) -> Series:
    """
    Volume Profile VAL: Value Area Low (approximation), normalized by price.
    
    VAL is the lower bound of value area (70% of volume). We approximate using
    the 30th percentile of typical price weighted by volume over 20 days.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Approximate VAL as 30th percentile of typical price weighted by volume
    def weighted_percentile(window_data, percentile=0.3):
        if len(window_data) < 5:
            return np.nan
        prices = window_data['price'].values
        volumes = window_data['volume'].values
        # Calculate weighted percentile
        sorted_idx = np.argsort(prices)
        sorted_prices = prices[sorted_idx]
        sorted_volumes = volumes[sorted_idx]
        cumsum_vol = np.cumsum(sorted_volumes)
        total_vol = cumsum_vol[-1]
        target_vol = total_vol * percentile
        idx = np.searchsorted(cumsum_vol, target_vol)
        if idx >= len(sorted_prices):
            return sorted_prices[-1]
        return sorted_prices[idx]
    
    # Rolling weighted percentile
    # Optimized: Use numpy arrays for faster access
    price_values = typical_price.values
    vol_values = volume.values
    
    val = pd.Series(index=close.index, dtype=float)
    for i in range(20, len(close)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_prices = price_values[i-20:i]
        window_volumes = vol_values[i-20:i]
        
        # Create window_data dict for weighted_percentile function
        window_data = pd.DataFrame({'price': window_prices, 'volume': window_volumes})
        val.iloc[i] = weighted_percentile(window_data, percentile=0.3)
    
    # Normalize by current price
    val_norm = val / close
    val_norm.name = "volume_profile_val"
    return val_norm


def feature_price_vs_poc(df: DataFrame) -> Series:
    """
    Price vs POC: Distance from current price to POC, normalized.
    
    Calculated as: (close - poc) / close.
    Positive = price above POC, negative = price below POC.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    poc = feature_volume_profile_poc(df) * close
    
    distance = (close - poc) / close
    distance.name = "price_vs_poc"
    return distance


def feature_price_vs_vah(df: DataFrame) -> Series:
    """
    Price vs VAH: Distance from price to Value Area High, normalized.
    
    Calculated as: (close - vah) / close.
    Positive = price above VAH, negative = price below VAH.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    vah = feature_volume_profile_vah(df) * close
    
    distance = (close - vah) / close
    distance.name = "price_vs_vah"
    return distance


def feature_price_vs_val(df: DataFrame) -> Series:
    """
    Price vs VAL: Distance from price to Value Area Low, normalized.
    
    Calculated as: (close - val) / close.
    Positive = price above VAL, negative = price below VAL.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    val = feature_volume_profile_val(df) * close
    
    distance = (close - val) / close
    distance.name = "price_vs_val"
    return distance


# ============================================================================
# BLOCK ML-7.2: VWAP Features (5 features)
# ============================================================================


def feature_volume_weighted_price(df: DataFrame) -> Series:
    """
    Volume Weighted Price: VWAP (Volume Weighted Average Price), normalized by price.
    
    VWAP = sum(price * volume) / sum(volume) over rolling 20-day window.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # VWAP over 20 days
    price_volume = close * volume
    vwap = price_volume.rolling(window=20, min_periods=1).sum() / volume.rolling(window=20, min_periods=1).sum()
    
    # Normalize by current price
    vwap_norm = vwap / close
    vwap_norm.name = "volume_weighted_price"
    return vwap_norm


def feature_price_vs_vwap(df: DataFrame) -> Series:
    """
    Price vs VWAP: Distance from price to VWAP, normalized.
    
    Calculated as: (close - vwap) / close.
    Positive = price above VWAP, negative = price below VWAP.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    vwap = feature_volume_weighted_price(df) * close
    
    distance = (close - vwap) / close
    distance.name = "price_vs_vwap"
    return distance


def feature_vwap_slope(df: DataFrame) -> Series:
    """
    VWAP Slope: VWAP trend direction, normalized by price.
    
    Measures rate of change of VWAP.
    Calculated as: vwap.diff(5) / close.
    Positive = VWAP rising, negative = VWAP falling.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    vwap = feature_volume_weighted_price(df) * close
    
    slope = vwap.diff(5) / close
    slope.name = "vwap_slope"
    return slope


def feature_vwap_distance_pct(df: DataFrame) -> Series:
    """
    VWAP Distance %: Percentage distance from VWAP.
    
    Calculated as: (close - vwap) / vwap * 100.
    Positive = % above VWAP, negative = % below VWAP.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    vwap = feature_volume_weighted_price(df) * close
    
    distance_pct = (close - vwap) / (vwap + 1e-10) * 100
    distance_pct.name = "vwap_distance_pct"
    return distance_pct


def feature_volume_profile_width(df: DataFrame) -> Series:
    """
    Volume Profile Width: Width of volume profile, normalized by price.
    
    Measures the spread between VAH and VAL.
    Calculated as: (vah - val) / close.
    Higher values = wider volume profile (more price range with volume).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    vah = feature_volume_profile_vah(df) * close
    val = feature_volume_profile_val(df) * close
    
    width = (vah - val) / close
    width.name = "volume_profile_width"
    return width


# ============================================================================
# BLOCK ML-7.3: Volume Analysis (9 features)
# ============================================================================


def feature_volume_distribution(df: DataFrame) -> Series:
    """
    Volume Distribution: Volume concentration metric.
    
    Measures how concentrated volume is around current price.
    Calculated as: 1 - (std of typical price weighted by volume) / mean(typical price).
    Higher values = more concentrated volume.
    Already normalized to [0, 1]. Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Weighted standard deviation over 20 days
    def weighted_std(window_data):
        if len(window_data) < 5:
            return np.nan
        prices = window_data['price'].values
        volumes = window_data['volume'].values
        total_vol = volumes.sum()
        if total_vol == 0:
            return np.nan
        mean_price = np.average(prices, weights=volumes)
        variance = np.average((prices - mean_price) ** 2, weights=volumes)
        return np.sqrt(variance)
    
    # Rolling weighted std
    # Optimized: Use numpy arrays for faster access
    price_values = typical_price.values
    vol_values = volume.values
    
    vol_std = pd.Series(index=close.index, dtype=float)
    for i in range(20, len(close)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_prices = price_values[i-20:i]
        window_volumes = vol_values[i-20:i]
        
        # Create window_data dict for weighted_std function
        window_data = pd.DataFrame({'price': window_prices, 'volume': window_volumes})
        vol_std.iloc[i] = weighted_std(window_data)
    
    # Mean typical price
    mean_price = typical_price.rolling(window=20, min_periods=1).mean()
    
    # Concentration metric
    concentration = 1 - (vol_std / (mean_price + 1e-10))
    concentration = concentration.clip(0.0, 1.0)
    concentration.name = "volume_distribution"
    return concentration


def feature_volume_climax(df: DataFrame) -> Series:
    """
    Volume Climax: Unusually high volume days (volume > 2x average), binary flag.
    
    Binary indicator: 1 if volume > 2x 20-day average, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    volume = _get_volume_series(df)
    
    avg_volume = volume.rolling(window=20, min_periods=1).mean()
    climax = (volume > 2 * avg_volume).astype(float)
    climax.name = "volume_climax"
    return climax


def feature_volume_dry_up(df: DataFrame) -> Series:
    """
    Volume Dry Up: Unusually low volume days (volume < 0.5x average), binary flag.
    
    Binary indicator: 1 if volume < 0.5x 20-day average, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    volume = _get_volume_series(df)
    
    avg_volume = volume.rolling(window=20, min_periods=1).mean()
    dry_up = (volume < 0.5 * avg_volume).astype(float)
    dry_up.name = "volume_dry_up"
    return dry_up


def feature_volume_breakout(df: DataFrame) -> Series:
    """
    Volume Breakout: Volume spike on price breakout, binary flag.
    
    Detects when volume > 1.5x average AND price breaks above 20-day high.
    Binary indicator: 1 if both conditions met, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    volume = _get_volume_series(df)
    
    # 20-day high (shifted by 1 to avoid lookahead)
    prev_20d_high = high.rolling(window=20, min_periods=1).max().shift(1)
    
    # Volume spike
    avg_volume = volume.rolling(window=20, min_periods=1).mean()
    volume_spike = volume > 1.5 * avg_volume
    
    # Price breakout
    price_breakout = close > prev_20d_high
    
    # Both conditions
    breakout = (volume_spike & price_breakout).astype(float)
    breakout.name = "volume_breakout"
    return breakout


def feature_volume_divergence(df: DataFrame) -> Series:
    """
    Volume Divergence: Volume vs price divergence.
    
    Measures correlation between volume changes and price changes over 20 days.
    Negative = volume diverging from price (potential reversal).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    price_changes = close.pct_change()
    volume_changes = volume.pct_change()
    
    # Rolling correlation
    def rolling_corr(series1, series2, window=20):
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            window1 = series1.iloc[i-window:i]
            window2 = series2.iloc[i-window:i]
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            if window1.std() == 0 or window2.std() == 0:
                result.iloc[i] = np.nan
                continue
            corr = window1.corr(window2)
            result.iloc[i] = corr if not pd.isna(corr) else np.nan
        return result
    
    divergence = rolling_corr(price_changes, volume_changes, window=20)
    divergence.name = "volume_divergence"
    return divergence


def feature_volume_autocorrelation(df: DataFrame) -> Series:
    """
    Volume Autocorrelation: Volume autocorrelation over 20 days.
    
    Measures correlation between current volume and lagged volume.
    Higher values = volume persistence (volume tends to cluster).
    No clipping - let ML learn the distribution.
    """
    volume = _get_volume_series(df)
    
    volume_lag = volume.shift(1)
    
    # Rolling correlation
    # Optimized: Use numpy arrays for faster access
    def rolling_corr(series1, series2, window=20):
        # Convert to numpy arrays
        values1 = series1.values
        values2 = series2.values
        
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            # Use numpy array slicing (much faster than pandas iloc)
            window1 = values1[i-window:i]
            window2 = values2[i-window:i]
            
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            
            # Use numpy std (faster than pandas)
            std1 = np.std(window1, ddof=0)
            std2 = np.std(window2, ddof=0)
            if std1 == 0 or std2 == 0:
                result.iloc[i] = np.nan
                continue
            
            # Use numpy correlation (faster than pandas)
            corr = np.corrcoef(window1, window2)[0, 1]
            result.iloc[i] = corr if not np.isnan(corr) else np.nan
        return result
    
    autocorr = rolling_corr(volume, volume_lag, window=20)
    autocorr.name = "volume_autocorrelation"
    return autocorr


def feature_volume_imbalance(df: DataFrame) -> Series:
    """
    Volume Imbalance: Buy vs sell volume imbalance proxy.
    
    Approximates buy/sell imbalance using price action and volume.
    Calculated as: (close - open) / (high - low) * volume, normalized.
    Positive = buying pressure, negative = selling pressure.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    # Price action component
    price_range = (high - low).replace(0, np.nan)
    price_action = (close - openp) / price_range
    
    # Volume imbalance
    imbalance = price_action * volume
    
    # Normalize by average volume
    avg_volume = volume.rolling(window=20, min_periods=1).mean()
    imbalance_norm = imbalance / (avg_volume + 1e-10)
    imbalance_norm.name = "volume_imbalance"
    return imbalance_norm


def feature_volume_at_price(df: DataFrame) -> Series:
    """
    Volume at Price: Volume at current price level, normalized.
    
    Approximates volume traded at current price level using typical price proximity.
    Calculated as: sum of volume where typical price is within 1% of current price.
    Normalized by dividing by total volume over 20 days.
    Already normalized to [0, 1]. Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # Volume at price (within 1% of current price)
    # Optimized: Use numpy arrays and vectorized operations inside the loop
    # Convert to numpy arrays for faster access
    tp_values = typical_price.values
    vol_values = volume.values
    close_values = close.values
    
    vol_at_price = pd.Series(index=close.index, dtype=float)
    for i in range(20, len(close)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_tp = tp_values[i-20:i]
        window_vol = vol_values[i-20:i]
        current_price = close_values[i]
        
        # Vectorized numpy operations (faster than pandas)
        within_range = np.abs(window_tp / current_price - 1.0) <= 0.01
        vol_at_price.iloc[i] = np.sum(window_vol[within_range])
    
    # Normalize by total volume
    total_vol = volume.rolling(window=20, min_periods=1).sum()
    vol_at_price_norm = vol_at_price / (total_vol + 1e-10)
    vol_at_price_norm = vol_at_price_norm.clip(0.0, 1.0)
    vol_at_price_norm.name = "volume_at_price"
    return vol_at_price_norm


def feature_volatility_ratio(df: DataFrame) -> Series:
    """
    Volatility Ratio: Volatility ratio (vol5/vol21).
    
    Measures short-term vs medium-term volatility.
    Calculated as: volatility_5d / volatility_21d.
    Values > 1.0 = short-term volatility higher (increasing volatility).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    vol_5d = close.pct_change().rolling(window=5, min_periods=1).std()
    vol_21d = close.pct_change().rolling(window=21, min_periods=1).std()
    
    ratio = vol_5d / (vol_21d + 1e-10)
    ratio.name = "volatility_ratio"
    return ratio


# ============================================================================
# BLOCK ML-8: Multi-Timeframe Weekly Features
# Helper function for weekly resampling
# ============================================================================


def _resample_to_weekly(df: DataFrame, column: str) -> Series:
    """
    Resample daily data to weekly (week ending Friday), then reindex to daily.
    
    Uses 'W-FRI' to resample to weekly data ending on Friday.
    Forward fills to daily frequency to avoid lookahead bias.
    """
    series = _get_column(df, column)
    weekly = series.resample('W-FRI').last()  # Week ending Friday
    daily = weekly.reindex(series.index, method='ffill')  # Forward fill to daily
    return daily


# ============================================================================
# BLOCK ML-8.1: Weekly Returns (3 features)
# ============================================================================


def feature_weekly_return_2w(df: DataFrame) -> Series:
    """
    Weekly Return 2w: 2-week log return.
    
    Calculates log return over 2 weeks (10 trading days).
    Formula: ln(close_t / close_{t-10}).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    log_ret_10d = np.log(close / close.shift(10))
    log_ret_10d.name = "weekly_return_2w"
    return log_ret_10d


def feature_weekly_return_4w(df: DataFrame) -> Series:
    """
    Weekly Return 4w: 4-week log return.
    
    Calculates log return over 4 weeks (20 trading days).
    Formula: ln(close_t / close_{t-20}).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    log_ret_20d = np.log(close / close.shift(20))
    log_ret_20d.name = "weekly_return_4w"
    return log_ret_20d


def feature_weekly_volume_ratio(df: DataFrame) -> Series:
    """
    Weekly Volume Ratio: Weekly volume vs 20-week average.
    
    Resamples volume to weekly, calculates 20-week average, then compares.
    Reindexed back to daily frequency.
    No clipping - let ML learn the distribution.
    """
    volume = _get_volume_series(df)
    
    # Resample to weekly (week ending Friday)
    weekly_volume = volume.resample('W-FRI').sum()
    
    # 20-week average
    weekly_avg_20w = weekly_volume.rolling(window=20, min_periods=1).mean()
    
    # Ratio
    weekly_ratio = weekly_volume / (weekly_avg_20w + 1e-10)
    
    # Reindex to daily and forward fill
    daily_ratio = weekly_ratio.reindex(volume.index, method='ffill')
    daily_ratio.name = "weekly_volume_ratio"
    return daily_ratio


# ============================================================================
# BLOCK ML-8.2: Weekly Moving Averages (6 features)
# ============================================================================


def feature_weekly_sma_5w(df: DataFrame) -> Series:
    """
    Weekly SMA 5w: 5-week SMA, normalized by price.
    
    Resamples close to weekly, calculates 5-week SMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 5-week SMA
    weekly_sma_5w = weekly_close.rolling(window=5, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_sma = weekly_sma_5w.reindex(close.index, method='ffill')
    
    # Normalize by current price
    sma_norm = daily_sma / close
    sma_norm.name = "weekly_sma_5w"
    return sma_norm


def feature_weekly_sma_10w(df: DataFrame) -> Series:
    """
    Weekly SMA 10w: 10-week SMA, normalized by price.
    
    Resamples close to weekly, calculates 10-week SMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 10-week SMA
    weekly_sma_10w = weekly_close.rolling(window=10, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_sma = weekly_sma_10w.reindex(close.index, method='ffill')
    
    # Normalize by current price
    sma_norm = daily_sma / close
    sma_norm.name = "weekly_sma_10w"
    return sma_norm


def feature_weekly_sma_20w(df: DataFrame) -> Series:
    """
    Weekly SMA 20w: 20-week SMA, normalized by price.
    
    Resamples close to weekly, calculates 20-week SMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 20-week SMA
    weekly_sma_20w = weekly_close.rolling(window=20, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_sma = weekly_sma_20w.reindex(close.index, method='ffill')
    
    # Normalize by current price
    sma_norm = daily_sma / close
    sma_norm.name = "weekly_sma_20w"
    return sma_norm


def feature_weekly_ema_5w(df: DataFrame) -> Series:
    """
    Weekly EMA 5w: 5-week EMA, normalized by price.
    
    Resamples close to weekly, calculates 5-week EMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 5-week EMA
    weekly_ema_5w = weekly_close.ewm(span=5, adjust=False).mean()
    
    # Reindex to daily and forward fill
    daily_ema = weekly_ema_5w.reindex(close.index, method='ffill')
    
    # Normalize by current price
    ema_norm = daily_ema / close
    ema_norm.name = "weekly_ema_5w"
    return ema_norm


def feature_weekly_ema_10w(df: DataFrame) -> Series:
    """
    Weekly EMA 10w: 10-week EMA, normalized by price.
    
    Resamples close to weekly, calculates 10-week EMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 10-week EMA
    weekly_ema_10w = weekly_close.ewm(span=10, adjust=False).mean()
    
    # Reindex to daily and forward fill
    daily_ema = weekly_ema_10w.reindex(close.index, method='ffill')
    
    # Normalize by current price
    ema_norm = daily_ema / close
    ema_norm.name = "weekly_ema_10w"
    return ema_norm


def feature_weekly_ema_20w(df: DataFrame) -> Series:
    """
    Weekly EMA 20w: 20-week EMA, normalized by price.
    
    Resamples close to weekly, calculates 20-week EMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 20-week EMA
    weekly_ema_20w = weekly_close.ewm(span=20, adjust=False).mean()
    
    # Reindex to daily and forward fill
    daily_ema = weekly_ema_20w.reindex(close.index, method='ffill')
    
    # Normalize by current price
    ema_norm = daily_ema / close
    ema_norm.name = "weekly_ema_20w"
    return ema_norm


# ============================================================================
# BLOCK ML-8.3: Weekly Indicators (6 features)
# ============================================================================


def feature_weekly_rsi_14w(df: DataFrame) -> Series:
    """
    Weekly RSI 14w: Weekly RSI (14-week), centered to [-1, +1].
    
    Resamples close to weekly, calculates 14-week RSI, reindexed to daily.
    Centered: (rsi - 50) / 50.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # Calculate RSI on weekly data
    delta = weekly_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_centered = (rsi - 50) / 50
    
    # Reindex to daily and forward fill
    daily_rsi = rsi_centered.reindex(close.index, method='ffill')
    daily_rsi.name = "weekly_rsi_14w"
    return daily_rsi


def feature_weekly_rsi_7w(df: DataFrame) -> Series:
    """
    Weekly RSI 7w: Weekly RSI (7-week), centered to [-1, +1].
    
    Resamples close to weekly, calculates 7-week RSI, reindexed to daily.
    Centered: (rsi - 50) / 50.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # Calculate RSI on weekly data
    delta = weekly_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=7, min_periods=1).mean()
    avg_loss = loss.rolling(window=7, min_periods=1).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_centered = (rsi - 50) / 50
    
    # Reindex to daily and forward fill
    daily_rsi = rsi_centered.reindex(close.index, method='ffill')
    daily_rsi.name = "weekly_rsi_7w"
    return daily_rsi


def feature_weekly_macd_histogram(df: DataFrame) -> Series:
    """
    Weekly MACD Histogram: Weekly MACD histogram, normalized by price.
    
    Resamples close to weekly, calculates MACD histogram, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # Calculate MACD on weekly data
    ema12 = weekly_close.ewm(span=12, adjust=False).mean()
    ema26 = weekly_close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # Reindex to daily and forward fill
    daily_hist = macd_hist.reindex(close.index, method='ffill')
    
    # Normalize by current price
    hist_norm = daily_hist / close
    hist_norm.name = "weekly_macd_histogram"
    return hist_norm


def feature_weekly_macd_signal(df: DataFrame) -> Series:
    """
    Weekly MACD Signal: Weekly MACD signal line, normalized by price.
    
    Resamples close to weekly, calculates MACD signal, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # Calculate MACD signal on weekly data
    ema12 = weekly_close.ewm(span=12, adjust=False).mean()
    ema26 = weekly_close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Reindex to daily and forward fill
    daily_signal = signal_line.reindex(close.index, method='ffill')
    
    # Normalize by current price
    signal_norm = daily_signal / close
    signal_norm.name = "weekly_macd_signal"
    return signal_norm


def feature_weekly_adx(df: DataFrame) -> Series:
    """
    Weekly ADX: Weekly ADX (14-week), normalized to [0, 1].
    
    Resamples OHLC to weekly, calculates 14-week ADX, reindexed to daily.
    Normalized: adx / 100.
    Already normalized to [0, 1]. Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_high = high.resample('W-FRI').max()
    weekly_low = low.resample('W-FRI').min()
    weekly_close = close.resample('W-FRI').last()
    
    # Calculate ADX on weekly data
    prev_high = weekly_high.shift(1)
    prev_low = weekly_low.shift(1)
    prev_close = weekly_close.shift(1)
    
    # True Range
    tr1 = weekly_high - weekly_low
    tr2 = (weekly_high - prev_close).abs()
    tr3 = (weekly_low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = weekly_high - prev_high
    minus_dm = prev_low - weekly_low
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    # Wilder's smoothing
    plus_dm_smooth = _wilder_smooth(plus_dm, period=14)
    minus_dm_smooth = _wilder_smooth(minus_dm, period=14)
    tr_smooth = _wilder_smooth(tr, period=14)
    
    # DI lines
    tr_safe = tr_smooth.replace(0, np.nan)
    plus_di = 100 * (plus_dm_smooth / tr_safe)
    minus_di = 100 * (minus_dm_smooth / tr_safe)
    
    # DX
    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum_safe
    
    # ADX
    adx = dx.ewm(span=14, adjust=False).mean()
    adx_norm = (adx / 100).clip(0.0, 1.0)
    
    # Reindex to daily and forward fill
    daily_adx = adx_norm.reindex(close.index, method='ffill')
    daily_adx.name = "weekly_adx"
    return daily_adx


def feature_weekly_stochastic(df: DataFrame) -> Series:
    """
    Weekly Stochastic: Weekly Stochastic (14-week), normalized to [0, 1].
    
    Resamples OHLC to weekly, calculates 14-week Stochastic %K, reindexed to daily.
    Already normalized to [0, 1]. Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_high = high.resample('W-FRI').max()
    weekly_low = low.resample('W-FRI').min()
    weekly_close = close.resample('W-FRI').last()
    
    # Calculate Stochastic on weekly data
    low_14 = weekly_low.rolling(window=14, min_periods=1).min()
    high_14 = weekly_high.rolling(window=14, min_periods=1).max()
    range_14 = (high_14 - low_14).replace(0, np.nan)
    stoch_k = (weekly_close - low_14) / range_14
    stoch_k = stoch_k.fillna(0.5).clip(0.0, 1.0)
    
    # Reindex to daily and forward fill
    daily_stoch = stoch_k.reindex(close.index, method='ffill')
    daily_stoch.name = "weekly_stochastic"
    return daily_stoch


# ============================================================================
# BLOCK ML-8.4: Weekly Price Comparisons (5 features)
# ============================================================================


def feature_close_vs_weekly_sma20(df: DataFrame) -> Series:
    """
    Close vs Weekly SMA20: Price vs 20-week SMA ratio.
    
    Calculates ratio of current close to 20-week SMA.
    Formula: close / weekly_sma20.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 20-week SMA
    weekly_sma_20w = weekly_close.rolling(window=20, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_sma = weekly_sma_20w.reindex(close.index, method='ffill')
    
    # Ratio
    ratio = close / (daily_sma + 1e-10)
    ratio.name = "close_vs_weekly_sma20"
    return ratio


def feature_close_vs_weekly_ema20(df: DataFrame) -> Series:
    """
    Close vs Weekly EMA20: Price vs 20-week EMA ratio.
    
    Calculates ratio of current close to 20-week EMA.
    Formula: close / weekly_ema20.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 20-week EMA
    weekly_ema_20w = weekly_close.ewm(span=20, adjust=False).mean()
    
    # Reindex to daily and forward fill
    daily_ema = weekly_ema_20w.reindex(close.index, method='ffill')
    
    # Ratio
    ratio = close / (daily_ema + 1e-10)
    ratio.name = "close_vs_weekly_ema20"
    return ratio


def feature_weekly_atr_pct(df: DataFrame) -> Series:
    """
    Weekly ATR %: Weekly ATR as percentage of price.
    
    Resamples OHLC to weekly, calculates ATR, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_high = high.resample('W-FRI').max()
    weekly_low = low.resample('W-FRI').min()
    weekly_close = close.resample('W-FRI').last()
    
    # True Range
    prev_close = weekly_close.shift(1)
    tr1 = weekly_high - weekly_low
    tr2 = (weekly_high - prev_close).abs()
    tr3 = (weekly_low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR (14-week)
    atr = tr.rolling(window=14, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_atr = atr.reindex(close.index, method='ffill')
    
    # Normalize by current price
    atr_pct = daily_atr / close
    atr_pct.name = "weekly_atr_pct"
    return atr_pct


def feature_weekly_trend_strength(df: DataFrame) -> Series:
    """
    Weekly Trend Strength: Slope of weekly SMA20, normalized by price.
    
    Measures rate of change of 20-week SMA.
    Calculated as: weekly_sma20.diff(5) / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last()
    
    # 20-week SMA
    weekly_sma_20w = weekly_close.rolling(window=20, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_sma = weekly_sma_20w.reindex(close.index, method='ffill')
    
    # Slope (5-week change)
    slope = daily_sma.diff(5) / close
    slope.name = "weekly_trend_strength"
    return slope


def feature_weekly_momentum(df: DataFrame) -> Series:
    """
    Weekly Momentum: Weekly momentum, normalized by price.
    
    Calculates momentum over 4 weeks (20 trading days).
    Formula: (close - close[20]) / close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = (close - close.shift(20)) / close
    momentum.name = "weekly_momentum"
    return momentum


# ============================================================================
# BLOCK ML-9: Multi-Timeframe Monthly Features
# ============================================================================


def feature_monthly_return_1m(df: DataFrame) -> Series:
    """
    Monthly Return 1m: 1-month log return.
    
    Calculates log return over 1 month (21 trading days).
    Formula: ln(close_t / close_{t-21}).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    log_ret_21d = np.log(close / close.shift(21))
    log_ret_21d.name = "monthly_return_1m"
    return log_ret_21d


def feature_monthly_return_3m(df: DataFrame) -> Series:
    """
    Monthly Return 3m: 3-month log return.
    
    Calculates log return over 3 months (63 trading days).
    Formula: ln(close_t / close_{t-63}).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    log_ret_63d = np.log(close / close.shift(63))
    log_ret_63d.name = "monthly_return_3m"
    return log_ret_63d


def feature_monthly_sma_3m(df: DataFrame) -> Series:
    """
    Monthly SMA 3m: 3-month SMA, normalized by price.
    
    Resamples close to monthly, calculates 3-month SMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to monthly (month-end)
    monthly_close = close.resample('ME').last()
    
    # 3-month SMA
    monthly_sma_3m = monthly_close.rolling(window=3, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_sma = monthly_sma_3m.reindex(close.index, method='ffill')
    
    # Normalize by current price
    sma_norm = daily_sma / close
    sma_norm.name = "monthly_sma_3m"
    return sma_norm


def feature_monthly_sma_6m(df: DataFrame) -> Series:
    """
    Monthly SMA 6m: 6-month SMA, normalized by price.
    
    Resamples close to monthly, calculates 6-month SMA, reindexed to daily.
    Normalized by dividing by current close.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to monthly (month-end)
    monthly_close = close.resample('ME').last()
    
    # 6-month SMA
    monthly_sma_6m = monthly_close.rolling(window=6, min_periods=1).mean()
    
    # Reindex to daily and forward fill
    daily_sma = monthly_sma_6m.reindex(close.index, method='ffill')
    
    # Normalize by current price
    sma_norm = daily_sma / close
    sma_norm.name = "monthly_sma_6m"
    return sma_norm


def feature_monthly_rsi(df: DataFrame) -> Series:
    """
    Monthly RSI: Monthly RSI (14-month), centered to [-1, +1].
    
    Resamples close to monthly, calculates 14-month RSI, reindexed to daily.
    Centered: (rsi - 50) / 50.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Resample to monthly (month-end)
    monthly_close = close.resample('ME').last()
    
    # Calculate RSI on monthly data
    delta = monthly_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_centered = (rsi - 50) / 50
    
    # Reindex to daily and forward fill
    daily_rsi = rsi_centered.reindex(close.index, method='ffill')
    daily_rsi.name = "monthly_rsi"
    return daily_rsi


# ============================================================================
# BLOCK ML-10.1: Candlestick Components (4 features)
# ============================================================================


def feature_candle_body_pct(df: DataFrame) -> Series:
    """
    Candle Body %: Candle body percentage (body/range) in [0, 1].
    
    Measures size of candle body relative to total candle range.
    Calculated as: abs(close - open) / (high - low).
    Values: 0.0 = doji (no body), 1.0 = marubozu (full body, no wicks).
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = (high - low).replace(0, np.nan)
    body_pct = body / range_
    body_pct = body_pct.clip(0.0, 1.0)
    body_pct.name = "candle_body_pct"
    return body_pct


def feature_candle_upper_wick_pct(df: DataFrame) -> Series:
    """
    Candle Upper Wick %: Upper wick percentage (upper/range) in [0, 1].
    
    Measures size of upper wick relative to total candle range.
    Calculated as: (high - max(close, open)) / (high - low).
    Values: 0.0 = no upper wick, 1.0 = full upper wick.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    upper_wick = high - pd.concat([close, openp], axis=1).max(axis=1)
    range_ = (high - low).replace(0, np.nan)
    upper_pct = upper_wick / range_
    upper_pct = upper_pct.clip(0.0, 1.0)
    upper_pct.name = "candle_upper_wick_pct"
    return upper_pct


def feature_candle_lower_wick_pct(df: DataFrame) -> Series:
    """
    Candle Lower Wick %: Lower wick percentage (lower/range) in [0, 1].
    
    Measures size of lower wick relative to total candle range.
    Calculated as: (min(close, open) - low) / (high - low).
    Values: 0.0 = no lower wick, 1.0 = full lower wick.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    lower_wick = pd.concat([close, openp], axis=1).min(axis=1) - low
    range_ = (high - low).replace(0, np.nan)
    lower_pct = lower_wick / range_
    lower_pct = lower_pct.clip(0.0, 1.0)
    lower_pct.name = "candle_lower_wick_pct"
    return lower_pct


def feature_candle_wicks_ratio(df: DataFrame) -> Series:
    """
    Candle Wicks Ratio: Upper wick / lower wick ratio.
    
    Measures relative size of upper vs lower wick.
    Calculated as: upper_wick / (lower_wick + 1e-10).
    Values: >1.0 = upper wick larger, <1.0 = lower wick larger, 1.0 = equal.
    No clipping - let ML learn the distribution.
    """
    upper_pct = feature_candle_upper_wick_pct(df)
    lower_pct = feature_candle_lower_wick_pct(df)
    
    ratio = upper_pct / (lower_pct + 1e-10)
    ratio.name = "candle_wicks_ratio"
    return ratio


# ============================================================================
# BLOCK ML-10.2: Reversal Patterns (8 features)
# ============================================================================


def feature_hanging_man(df: DataFrame) -> Series:
    """
    Hanging Man: Bearish reversal pattern, binary flag.
    
    Pattern: Small body at top, long lower wick, little/no upper wick.
    Conditions: body < 30% of range, lower wick > 2x body, upper wick < 0.5x body.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = high - low
    lower_wick = pd.concat([close, openp], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([close, openp], axis=1).max(axis=1)
    
    # Hanging man conditions
    body_small = (body / (range_ + 1e-10)) < 0.3
    lower_wick_long = lower_wick > (2 * body)
    upper_wick_small = upper_wick < (0.5 * body)
    
    pattern = (body_small & lower_wick_long & upper_wick_small).astype(float)
    pattern.name = "hanging_man"
    return pattern


def feature_inverted_hammer(df: DataFrame) -> Series:
    """
    Inverted Hammer: Bullish reversal pattern, binary flag.
    
    Pattern: Small body at bottom, long upper wick, little/no lower wick.
    Conditions: body < 30% of range, upper wick > 2x body, lower wick < 0.5x body.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = high - low
    lower_wick = pd.concat([close, openp], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([close, openp], axis=1).max(axis=1)
    
    # Inverted hammer conditions
    body_small = (body / (range_ + 1e-10)) < 0.3
    upper_wick_long = upper_wick > (2 * body)
    lower_wick_small = lower_wick < (0.5 * body)
    
    pattern = (body_small & upper_wick_long & lower_wick_small).astype(float)
    pattern.name = "inverted_hammer"
    return pattern


def feature_piercing_pattern(df: DataFrame) -> Series:
    """
    Piercing Pattern: Bullish 2-candle reversal pattern, binary flag.
    
    Pattern: First candle bearish, second candle opens below first close but closes above midpoint.
    Conditions: Candle 1 bearish, Candle 2 opens < Candle 1 close, Candle 2 close > (Candle 1 high + Candle 1 low) / 2.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Previous candle
    prev_close = close.shift(1)
    prev_open = openp.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    # Conditions
    prev_bearish = prev_close < prev_open
    opens_below = openp < prev_close
    closes_above_midpoint = close > ((prev_high + prev_low) / 2)
    
    pattern = (prev_bearish & opens_below & closes_above_midpoint).astype(float)
    pattern.name = "piercing_pattern"
    return pattern


def feature_dark_cloud_cover(df: DataFrame) -> Series:
    """
    Dark Cloud Cover: Bearish 2-candle reversal pattern, binary flag.
    
    Pattern: First candle bullish, second candle opens above first close but closes below midpoint.
    Conditions: Candle 1 bullish, Candle 2 opens > Candle 1 close, Candle 2 close < (Candle 1 high + Candle 1 low) / 2.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Previous candle
    prev_close = close.shift(1)
    prev_open = openp.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    # Conditions
    prev_bullish = prev_close > prev_open
    opens_above = openp > prev_close
    closes_below_midpoint = close < ((prev_high + prev_low) / 2)
    
    pattern = (prev_bullish & opens_above & closes_below_midpoint).astype(float)
    pattern.name = "dark_cloud_cover"
    return pattern


def feature_harami_bullish(df: DataFrame) -> Series:
    """
    Harami Bullish: Bullish 2-candle reversal pattern, binary flag.
    
    Pattern: First candle large bearish, second candle small and inside first candle's range.
    Conditions: Candle 1 bearish and large, Candle 2 body inside Candle 1 range.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Previous candle
    prev_close = close.shift(1)
    prev_open = openp.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    # Conditions
    prev_bearish = prev_close < prev_open
    prev_large = (prev_high - prev_low) > (high - low) * 1.5  # Previous candle much larger
    current_inside = (high < prev_high) & (low > prev_low)
    
    pattern = (prev_bearish & prev_large & current_inside).astype(float)
    pattern.name = "harami_bullish"
    return pattern


def feature_harami_bearish(df: DataFrame) -> Series:
    """
    Harami Bearish: Bearish 2-candle reversal pattern, binary flag.
    
    Pattern: First candle large bullish, second candle small and inside first candle's range.
    Conditions: Candle 1 bullish and large, Candle 2 body inside Candle 1 range.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Previous candle
    prev_close = close.shift(1)
    prev_open = openp.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    # Conditions
    prev_bullish = prev_close > prev_open
    prev_large = (prev_high - prev_low) > (high - low) * 1.5  # Previous candle much larger
    current_inside = (high < prev_high) & (low > prev_low)
    
    pattern = (prev_bullish & prev_large & current_inside).astype(float)
    pattern.name = "harami_bearish"
    return pattern


def feature_engulfing_bullish(df: DataFrame) -> Series:
    """
    Engulfing Bullish: Bullish engulfing pattern, binary flag.
    
    Pattern: First candle bearish, second candle bullish and completely engulfs first.
    Conditions: Candle 1 bearish, Candle 2 bullish, Candle 2 open < Candle 1 close, Candle 2 close > Candle 1 open.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    
    # Previous candle
    prev_close = close.shift(1)
    prev_open = openp.shift(1)
    
    # Conditions
    prev_bearish = prev_close < prev_open
    current_bullish = close > openp
    engulfs = (openp < prev_close) & (close > prev_open)
    
    pattern = (prev_bearish & current_bullish & engulfs).astype(float)
    pattern.name = "engulfing_bullish"
    return pattern


def feature_engulfing_bearish(df: DataFrame) -> Series:
    """
    Engulfing Bearish: Bearish engulfing pattern, binary flag.
    
    Pattern: First candle bullish, second candle bearish and completely engulfs first.
    Conditions: Candle 1 bullish, Candle 2 bearish, Candle 2 open > Candle 1 close, Candle 2 close < Candle 1 open.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    
    # Previous candle
    prev_close = close.shift(1)
    prev_open = openp.shift(1)
    
    # Conditions
    prev_bullish = prev_close > prev_open
    current_bearish = close < openp
    engulfs = (openp > prev_close) & (close < prev_open)
    
    pattern = (prev_bullish & current_bearish & engulfs).astype(float)
    pattern.name = "engulfing_bearish"
    return pattern


# ============================================================================
# BLOCK ML-10.3: Continuation Patterns (5 features)
# ============================================================================


def feature_three_white_soldiers(df: DataFrame) -> Series:
    """
    Three White Soldiers: Bullish 3-candle continuation pattern, binary flag.
    
    Pattern: Three consecutive bullish candles with higher closes.
    Conditions: Last 3 candles all bullish, each close > previous close.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    
    # Check last 3 candles
    c1_bullish = close.shift(2) > openp.shift(2)
    c2_bullish = close.shift(1) > openp.shift(1)
    c3_bullish = close > openp
    
    # Higher closes
    c2_higher = close.shift(1) > close.shift(2)
    c3_higher = close > close.shift(1)
    
    pattern = (c1_bullish & c2_bullish & c3_bullish & c2_higher & c3_higher).astype(float)
    pattern.name = "three_white_soldiers"
    return pattern


def feature_three_black_crows(df: DataFrame) -> Series:
    """
    Three Black Crows: Bearish 3-candle continuation pattern, binary flag.
    
    Pattern: Three consecutive bearish candles with lower closes.
    Conditions: Last 3 candles all bearish, each close < previous close.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    
    # Check last 3 candles
    c1_bearish = close.shift(2) < openp.shift(2)
    c2_bearish = close.shift(1) < openp.shift(1)
    c3_bearish = close < openp
    
    # Lower closes
    c2_lower = close.shift(1) < close.shift(2)
    c3_lower = close < close.shift(1)
    
    pattern = (c1_bearish & c2_bearish & c3_bearish & c2_lower & c3_lower).astype(float)
    pattern.name = "three_black_crows"
    return pattern


def feature_inside_bar(df: DataFrame) -> Series:
    """
    Inside Bar: Inside bar pattern, binary flag.
    
    Pattern: Current candle's range is completely inside previous candle's range.
    Conditions: Current high < previous high, current low > previous low.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    inside = (high < prev_high) & (low > prev_low)
    pattern = inside.astype(float)
    pattern.name = "inside_bar"
    return pattern


def feature_outside_bar(df: DataFrame) -> Series:
    """
    Outside Bar: Outside bar pattern, binary flag.
    
    Pattern: Current candle's range completely engulfs previous candle's range.
    Conditions: Current high > previous high, current low < previous low.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    
    outside = (high > prev_high) & (low < prev_low)
    pattern = outside.astype(float)
    pattern.name = "outside_bar"
    return pattern


def feature_engulfing_strength(df: DataFrame) -> Series:
    """
    Engulfing Strength: Strength of engulfing pattern, normalized.
    
    Measures how strongly current candle engulfs previous candle.
    Calculated as: (current_range - prev_range) / prev_range.
    Positive = strong engulfing, negative = weak/no engulfing.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    current_range = high - low
    prev_range = (high - low).shift(1)
    
    strength = (current_range - prev_range) / (prev_range + 1e-10)
    strength.name = "engulfing_strength"
    return strength


# ============================================================================
# BLOCK ML-10.4: Single Candle Patterns (5 features)
# ============================================================================


def feature_doji(df: DataFrame) -> Series:
    """
    Doji: Doji pattern, binary flag.
    
    Pattern: Open and close are very close (small body).
    Conditions: Body < 5% of range.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = high - low
    
    doji = (body / (range_ + 1e-10)) < 0.05
    pattern = doji.astype(float)
    pattern.name = "doji"
    return pattern


def feature_hammer(df: DataFrame) -> Series:
    """
    Hammer: Hammer pattern, binary flag.
    
    Pattern: Small body at top, long lower wick, little/no upper wick.
    Conditions: body < 30% of range, lower wick > 2x body, upper wick < 0.5x body, close > open (bullish).
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = high - low
    lower_wick = pd.concat([close, openp], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([close, openp], axis=1).max(axis=1)
    
    # Hammer conditions (bullish version)
    body_small = (body / (range_ + 1e-10)) < 0.3
    lower_wick_long = lower_wick > (2 * body)
    upper_wick_small = upper_wick < (0.5 * body)
    bullish = close > openp
    
    pattern = (body_small & lower_wick_long & upper_wick_small & bullish).astype(float)
    pattern.name = "hammer"
    return pattern


def feature_shooting_star(df: DataFrame) -> Series:
    """
    Shooting Star: Shooting star pattern, binary flag.
    
    Pattern: Small body at bottom, long upper wick, little/no lower wick.
    Conditions: body < 30% of range, upper wick > 2x body, lower wick < 0.5x body, close < open (bearish).
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = high - low
    lower_wick = pd.concat([close, openp], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([close, openp], axis=1).max(axis=1)
    
    # Shooting star conditions (bearish version)
    body_small = (body / (range_ + 1e-10)) < 0.3
    upper_wick_long = upper_wick > (2 * body)
    lower_wick_small = lower_wick < (0.5 * body)
    bearish = close < openp
    
    pattern = (body_small & upper_wick_long & lower_wick_small & bearish).astype(float)
    pattern.name = "shooting_star"
    return pattern


def feature_marubozu(df: DataFrame) -> Series:
    """
    Marubozu: Marubozu pattern (no wicks), binary flag.
    
    Pattern: Candle with no wicks (body equals range).
    Conditions: Body > 95% of range.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = high - low
    
    marubozu = (body / (range_ + 1e-10)) > 0.95
    pattern = marubozu.astype(float)
    pattern.name = "marubozu"
    return pattern


def feature_spinning_top(df: DataFrame) -> Series:
    """
    Spinning Top: Spinning top pattern, binary flag.
    
    Pattern: Small body with long wicks on both sides.
    Conditions: body < 30% of range, both wicks > 1.5x body.
    Binary indicator: 1 if pattern detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    body = (close - openp).abs()
    range_ = high - low
    lower_wick = pd.concat([close, openp], axis=1).min(axis=1) - low
    upper_wick = high - pd.concat([close, openp], axis=1).max(axis=1)
    
    # Spinning top conditions
    body_small = (body / (range_ + 1e-10)) < 0.3
    both_wicks_long = (lower_wick > 1.5 * body) & (upper_wick > 1.5 * body)
    
    pattern = (body_small & both_wicks_long).astype(float)
    pattern.name = "spinning_top"
    return pattern


# ============================================================================
# BLOCK ML-10.5: Pattern Quality (3 features)
# ============================================================================


def feature_pattern_strength(df: DataFrame) -> Series:
    """
    Pattern Strength: Overall pattern strength score.
    
    Combines multiple pattern signals into a strength score.
    Calculated as: weighted sum of bullish patterns - weighted sum of bearish patterns.
    Positive = bullish patterns stronger, negative = bearish patterns stronger.
    No clipping - let ML learn the distribution.
    """
    # Get pattern signals
    bullish_patterns = (
        feature_inverted_hammer(df) +
        feature_piercing_pattern(df) +
        feature_harami_bullish(df) +
        feature_engulfing_bullish(df) +
        feature_three_white_soldiers(df) +
        feature_hammer(df)
    )
    
    bearish_patterns = (
        feature_hanging_man(df) +
        feature_dark_cloud_cover(df) +
        feature_harami_bearish(df) +
        feature_engulfing_bearish(df) +
        feature_three_black_crows(df) +
        feature_shooting_star(df)
    )
    
    strength = bullish_patterns - bearish_patterns
    strength.name = "pattern_strength"
    return strength


def feature_pattern_confirmation(df: DataFrame) -> Series:
    """
    Pattern Confirmation: Pattern confirmation signal.
    
    Measures if patterns are confirmed by price action (volume, momentum).
    Calculated as: correlation between pattern signals and price momentum over 5 days.
    Higher values = patterns confirmed by price action.
    No clipping - let ML learn the distribution.
    """
    # Get pattern strength
    pattern_str = feature_pattern_strength(df)
    
    # Price momentum
    close = _get_close_series(df)
    price_momentum = close.pct_change(5)
    
    # Rolling correlation
    def rolling_corr(series1, series2, window=20):
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            window1 = series1.iloc[i-window:i]
            window2 = series2.iloc[i-window:i]
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            if window1.std() == 0 or window2.std() == 0:
                result.iloc[i] = np.nan
                continue
            corr = window1.corr(window2)
            result.iloc[i] = corr if not pd.isna(corr) else np.nan
        return result
    
    confirmation = rolling_corr(pattern_str, price_momentum, window=20)
    confirmation.name = "pattern_confirmation"
    return confirmation


def feature_pattern_divergence(df: DataFrame) -> Series:
    """
    Pattern Divergence: Pattern vs price divergence.
    
    Measures divergence between pattern signals and price direction.
    Calculated as: correlation between pattern strength and price changes over 10 days.
    Negative = patterns diverging from price (potential reversal).
    No clipping - let ML learn the distribution.
    """
    # Get pattern strength
    pattern_str = feature_pattern_strength(df)
    
    # Price changes
    close = _get_close_series(df)
    price_changes = close.pct_change(10)
    
    # Rolling correlation
    def rolling_corr(series1, series2, window=20):
        result = pd.Series(index=series1.index, dtype=float)
        for i in range(window, len(series1)):
            window1 = series1.iloc[i-window:i]
            window2 = series2.iloc[i-window:i]
            if len(window1) < 10:
                result.iloc[i] = np.nan
                continue
            if window1.std() == 0 or window2.std() == 0:
                result.iloc[i] = np.nan
                continue
            corr = window1.corr(window2)
            result.iloc[i] = corr if not pd.isna(corr) else np.nan
        return result
    
    divergence = rolling_corr(pattern_str, price_changes, window=20)
    divergence.name = "pattern_divergence"
    return divergence


# ============================================================================
# BLOCK ML-11: Price Action - Consecutive Days & Swings (6 features)
# ============================================================================


def feature_consecutive_green_days(df: DataFrame) -> Series:
    """
    Consecutive Green Days: Count of consecutive up days.
    
    Counts how many consecutive days the close has been higher than previous close.
    Calculated by iterating through price changes and counting consecutive positive returns.
    Normalized by dividing by 20 (max reasonable streak).
    Values: 0.0 = no streak, 1.0 = 20+ day streak.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Calculate daily returns
    returns = close.pct_change()
    
    # Count consecutive green days
    consecutive = pd.Series(index=close.index, dtype=float)
    current_streak = 0
    
    for i in range(len(close)):
        if pd.isna(returns.iloc[i]):
            consecutive.iloc[i] = 0
            current_streak = 0
            continue
        
        if returns.iloc[i] > 0:
            current_streak += 1
        else:
            current_streak = 0
        
        consecutive.iloc[i] = current_streak / 20.0  # Normalize by 20
    
    consecutive = consecutive.clip(0.0, 1.0)
    consecutive.name = "consecutive_green_days"
    return consecutive


def feature_consecutive_red_days(df: DataFrame) -> Series:
    """
    Consecutive Red Days: Count of consecutive down days.
    
    Counts how many consecutive days the close has been lower than previous close.
    Calculated by iterating through price changes and counting consecutive negative returns.
    Normalized by dividing by 20 (max reasonable streak).
    Values: 0.0 = no streak, 1.0 = 20+ day streak.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Calculate daily returns
    returns = close.pct_change()
    
    # Count consecutive red days
    consecutive = pd.Series(index=close.index, dtype=float)
    current_streak = 0
    
    for i in range(len(close)):
        if pd.isna(returns.iloc[i]):
            consecutive.iloc[i] = 0
            current_streak = 0
            continue
        
        if returns.iloc[i] < 0:
            current_streak += 1
        else:
            current_streak = 0
        
        consecutive.iloc[i] = current_streak / 20.0  # Normalize by 20
    
    consecutive = consecutive.clip(0.0, 1.0)
    consecutive.name = "consecutive_red_days"
    return consecutive


def feature_higher_high_10d(df: DataFrame) -> Series:
    """
    Higher High (10-day): Binary flag indicating if current close > previous 10-day max.
    
    Indicates if the current close is higher than the maximum close of the previous 10 days.
    A higher high suggests bullish momentum and potential trend continuation.
    Calculated as: (close > close.shift(1).rolling(10).max()).astype(int)
    Binary indicator: 1 if higher high detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Calculate max of previous 10 days (excluding current day)
    prev_10d_max = close.shift(1).rolling(window=10, min_periods=1).max()
    
    # Check if current close is higher than previous 10-day max
    hh = (close > prev_10d_max).astype(float)
    hh = hh.fillna(0.0).clip(0.0, 1.0)
    hh.name = "higher_high_10d"
    return hh


def feature_higher_low_10d(df: DataFrame) -> Series:
    """
    Higher Low (10-day): Binary flag indicating if current close > previous 10-day min.
    
    Indicates if the current close is higher than the minimum close of the previous 10 days.
    A higher low suggests bullish momentum and potential trend continuation.
    Calculated as: (close > close.shift(1).rolling(10).min()).astype(int)
    Binary indicator: 1 if higher low detected, else 0.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Calculate min of previous 10 days (excluding current day)
    prev_10d_min = close.shift(1).rolling(window=10, min_periods=1).min()
    
    # Check if current close is higher than previous 10-day min
    hl = (close > prev_10d_min).astype(float)
    hl = hl.fillna(0.0).clip(0.0, 1.0)
    hl.name = "higher_low_10d"
    return hl


def feature_swing_low_10d(df: DataFrame) -> Series:
    """
    Swing Low (10-day): Recent swing low price over previous 10 days.
    
    Identifies the most recent structural support level (swing low) which can be used
    for stop-loss placement. The swing low represents the lowest price point in the
    recent price action, indicating a support level.
    Calculated as: low.shift(1).rolling(10, min_periods=1).min()
    Excludes current day to avoid lookahead bias.
    Returns actual swing low price (not normalized).
    No clipping - let ML learn the distribution.
    """
    low = _get_low_series(df)
    
    # Calculate the minimum low over the previous 10 days (excluding current day)
    # shift(1) excludes current day to avoid lookahead bias
    swing_low = low.shift(1).rolling(window=10, min_periods=1).min()
    
    swing_low.name = "swing_low_10d"
    return swing_low


def feature_pattern_cluster(df: DataFrame) -> Series:
    """
    Pattern Cluster: Count of multiple patterns occurring simultaneously.
    
    Measures how many candlestick patterns from Block ML-10 are occurring at the same time.
    Calculated as: sum of all pattern detection flags (binary 0/1).
    Higher values = multiple patterns confirming each other (stronger signal).
    Normalized by dividing by 10 (max reasonable pattern count).
    Values: 0.0 = no patterns, 1.0 = 10+ patterns.
    Already normalized to [0, 1]. Clipped for safety.
    """
    # Get all pattern signals from Block ML-10
    patterns = [
        feature_hanging_man(df),
        feature_inverted_hammer(df),
        feature_piercing_pattern(df),
        feature_dark_cloud_cover(df),
        feature_harami_bullish(df),
        feature_harami_bearish(df),
        feature_engulfing_bullish(df),
        feature_engulfing_bearish(df),
        feature_three_white_soldiers(df),
        feature_three_black_crows(df),
        feature_inside_bar(df),
        feature_outside_bar(df),
        feature_doji(df),
        feature_hammer(df),
        feature_shooting_star(df),
        feature_marubozu(df),
        feature_spinning_top(df),
    ]
    
    # Sum all pattern flags
    cluster_count = pd.Series(0.0, index=df.index)
    for pattern in patterns:
        cluster_count += pattern.fillna(0.0)
    
    # Normalize by 10 (max reasonable pattern count)
    cluster_normalized = cluster_count / 10.0
    cluster_normalized = cluster_normalized.clip(0.0, 1.0)
    cluster_normalized.name = "pattern_cluster"
    return cluster_normalized


# ============================================================================
# BLOCK ML-12: Market Regime Indicators (SPY-based) (15 features)
# ============================================================================


def _get_spy_close_aligned(df: DataFrame) -> Optional[Series]:
    """
    Helper function to get SPY close price aligned with stock DataFrame dates.
    
    Returns SPY close Series aligned to stock dates, or None if SPY data unavailable.
    """
    spy_data = _load_spy_data()
    if spy_data is None:
        return None
    
    # Get SPY close price - try different column name variations
    if 'Close' in spy_data.columns:
        spy_close = spy_data['Close']
    elif 'close' in spy_data.columns:
        spy_close = spy_data['close']
    elif 'Adj Close' in spy_data.columns:
        spy_close = spy_data['Adj Close']
    else:
        numeric_cols = spy_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            spy_close = spy_data[numeric_cols[0]]
        else:
            return None
    
    if spy_close is None or len(spy_close) == 0:
        return None
    
    # Ensure SPY index is DatetimeIndex and sorted
    if not isinstance(spy_close.index, pd.DatetimeIndex):
        spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = spy_close.sort_index()
    
    # Get stock date index for alignment
    if isinstance(df.index, pd.DatetimeIndex):
        stock_dates = df.index
    else:
        stock_dates = pd.to_datetime(df.index)
    
    # Normalize dates to date-only for matching
    stock_dates_normalized = pd.to_datetime(stock_dates).normalize()
    spy_dates_normalized = pd.to_datetime(spy_close.index).normalize()
    
    # Reindex SPY to stock dates (forward fill for missing dates)
    spy_aligned = spy_close.reindex(stock_dates_normalized, method='ffill')
    
    return spy_aligned


def feature_market_regime(df: DataFrame) -> Series:
    """
    Market Regime: Bull/Bear/Sideways classification (0/1/2).
    
    Classifies market regime based on SPY price relative to SMA50 and SMA200.
    - 0 = Bull (SPY > SMA50 > SMA200)
    - 1 = Bear (SPY < SMA50 < SMA200)
    - 2 = Sideways (mixed signals)
    Binary/ternary indicator: 0/1/2.
    Already normalized (0/1/2). Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(2.0, index=df.index)  # Default to sideways
        result.name = "market_regime"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    
    # Classify regime
    bull = (spy_close > sma50) & (sma50 > sma200)
    bear = (spy_close < sma50) & (sma50 < sma200)
    
    regime = pd.Series(2.0, index=df.index)  # Default to sideways
    regime[bull] = 0.0  # Bull
    regime[bear] = 1.0  # Bear
    
    regime = regime.clip(0.0, 2.0)
    regime.name = "market_regime"
    return regime


def feature_regime_strength(df: DataFrame) -> Series:
    """
    Regime Strength: Strength of current regime (0-1).
    
    Measures how strongly the current regime is established.
    Calculated as: distance from SPY to SMA50, normalized by volatility.
    Higher values = stronger regime, lower = weaker/transitioning.
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "regime_strength"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    distance = (spy_close - sma50).abs()
    volatility = spy_close.rolling(window=20, min_periods=1).std()
    
    strength = distance / (volatility + 1e-10)
    strength = strength / (strength.rolling(window=252, min_periods=1).max() + 1e-10)  # Normalize by max
    strength = strength.clip(0.0, 1.0)
    strength.name = "regime_strength"
    return strength


def feature_regime_duration(df: DataFrame) -> Series:
    """
    Regime Duration: Days in current regime, normalized.
    
    Counts consecutive days market has been in same regime (bull/bear/sideways).
    Normalized by dividing by 252 (max reasonable duration ~1 year).
    Values: 0.0 = just changed, 1.0 = long regime.
    Already normalized to [0, 1]. Clipped for safety.
    """
    regime = feature_market_regime(df)
    
    # Count consecutive days in same regime
    duration = pd.Series(index=df.index, dtype=float)
    current_duration = 0
    current_regime = None
    
    for i in range(len(regime)):
        if pd.isna(regime.iloc[i]):
            duration.iloc[i] = 0
            current_duration = 0
            current_regime = None
            continue
        
        if current_regime is None:
            current_regime = regime.iloc[i]
            current_duration = 1
        elif regime.iloc[i] == current_regime:
            current_duration += 1
        else:
            current_regime = regime.iloc[i]
            current_duration = 1
        
        duration.iloc[i] = current_duration / 252.0  # Normalize by 252
    
    duration = duration.clip(0.0, 1.0)
    duration.name = "regime_duration"
    return duration


def feature_regime_change_probability(df: DataFrame) -> Series:
    """
    Regime Change Probability: Likelihood of regime change (0-1).
    
    Measures probability of regime change based on:
    - Distance from regime boundaries (SMA50, SMA200)
    - Volatility expansion
    - Momentum divergence
    Higher values = higher probability of change.
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "regime_change_probability"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    
    # Distance to boundaries
    dist_to_sma50 = (spy_close - sma50).abs() / (spy_close + 1e-10)
    dist_to_sma200 = (spy_close - sma200).abs() / (spy_close + 1e-10)
    
    # Volatility expansion
    volatility = spy_close.rolling(window=20, min_periods=1).std()
    vol_expansion = volatility / (volatility.rolling(window=60, min_periods=1).mean() + 1e-10)
    
    # Momentum divergence
    momentum = spy_close.pct_change(10)
    momentum_ma = momentum.rolling(window=20, min_periods=1).mean()
    momentum_div = (momentum - momentum_ma).abs()
    
    # Combine signals
    change_prob = (dist_to_sma50 + dist_to_sma200) / 2.0
    change_prob = change_prob + (vol_expansion - 1.0).clip(0, 1) * 0.3
    change_prob = change_prob + momentum_div.clip(0, 0.1) * 5.0
    
    change_prob = change_prob / (change_prob.rolling(window=252, min_periods=1).max() + 1e-10)
    change_prob = change_prob.clip(0.0, 1.0)
    change_prob.name = "regime_change_probability"
    return change_prob


def feature_trending_market_flag(df: DataFrame) -> Series:
    """
    Trending Market Flag: Binary flag for strong trend vs choppy.
    
    Indicates if market is in a strong trending phase (not choppy).
    Conditions: SPY > SMA50 > SMA200 OR SPY < SMA50 < SMA200, with strong momentum.
    Binary indicator: 1 if trending, 0 if choppy.
    Already normalized (binary 0/1). Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "trending_market_flag"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    momentum = spy_close.pct_change(20)
    
    # Strong uptrend
    uptrend = (spy_close > sma50) & (sma50 > sma200) & (momentum > 0.01)
    # Strong downtrend
    downtrend = (spy_close < sma50) & (sma50 < sma200) & (momentum < -0.01)
    
    trending = (uptrend | downtrend).astype(float)
    trending.name = "trending_market_flag"
    return trending


def feature_choppy_market_flag(df: DataFrame) -> Series:
    """
    Choppy Market Flag: Binary flag for sideways/choppy market.
    
    Indicates if market is in a choppy/sideways phase.
    Conditions: SPY oscillating around SMA50, low momentum, high volatility.
    Binary indicator: 1 if choppy, 0 if trending.
    Already normalized (binary 0/1). Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "choppy_market_flag"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    momentum = spy_close.pct_change(20).abs()
    volatility = spy_close.rolling(window=20, min_periods=1).std() / spy_close
    
    # Choppy conditions
    near_sma50 = (spy_close - sma50).abs() / spy_close < 0.02  # Within 2%
    low_momentum = momentum < 0.01  # Low momentum
    high_vol = volatility > volatility.rolling(window=60, min_periods=1).mean() * 1.2
    
    choppy = (near_sma50 & low_momentum & high_vol).astype(float)
    choppy.name = "choppy_market_flag"
    return choppy


def feature_bull_market_flag(df: DataFrame) -> Series:
    """
    Bull Market Flag: Binary flag for bullish regime.
    
    Indicates if market is in a bullish regime.
    Conditions: SPY > SMA50 > SMA200.
    Binary indicator: 1 if bull market, 0 otherwise.
    Already normalized (binary 0/1). Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "bull_market_flag"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    
    bull = ((spy_close > sma50) & (sma50 > sma200)).astype(float)
    bull.name = "bull_market_flag"
    return bull


def feature_bear_market_flag(df: DataFrame) -> Series:
    """
    Bear Market Flag: Binary flag for bearish regime.
    
    Indicates if market is in a bearish regime.
    Conditions: SPY < SMA50 < SMA200.
    Binary indicator: 1 if bear market, 0 otherwise.
    Already normalized (binary 0/1). Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "bear_market_flag"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    
    bear = ((spy_close < sma50) & (sma50 < sma200)).astype(float)
    bear.name = "bear_market_flag"
    return bear


def feature_market_volatility_regime(df: DataFrame) -> Series:
    """
    Market Volatility Regime: Market volatility regime (low/normal/high), normalized.
    
    Classifies volatility regime based on SPY volatility relative to historical.
    - 0.0 = Low volatility (bottom 33%)
    - 0.5 = Normal volatility (middle 33%)
    - 1.0 = High volatility (top 33%)
    Normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_volatility_regime"
        return result
    
    volatility = spy_close.rolling(window=20, min_periods=1).std() / spy_close
    # Optimized: Use vectorized rank instead of .apply() with lambda
    vol_percentile = volatility.rolling(window=252, min_periods=1).rank(pct=True, method='average')
    vol_percentile = vol_percentile.fillna(0.5)
    
    vol_regime = vol_percentile.clip(0.0, 1.0)
    vol_regime.name = "market_volatility_regime"
    return vol_regime


def feature_market_momentum_regime(df: DataFrame) -> Series:
    """
    Market Momentum Regime: Market momentum regime (strong/weak/neutral), normalized.
    
    Classifies momentum regime based on SPY momentum strength.
    - 0.0 = Weak momentum (negative/weak)
    - 0.5 = Neutral momentum (near zero)
    - 1.0 = Strong momentum (strong positive)
    Normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_momentum_regime"
        return result
    
    momentum = spy_close.pct_change(20)
    momentum_normalized = (momentum + 0.1) / 0.2  # Normalize: -0.1 to +0.1 -> 0 to 1
    momentum_normalized = momentum_normalized.clip(0.0, 1.0)
    momentum_normalized.name = "market_momentum_regime"
    return momentum_normalized


def feature_regime_transition(df: DataFrame) -> Series:
    """
    Regime Transition: Binary flag for regime transition occurring.
    
    Indicates if market is transitioning between regimes.
    Conditions: Regime changed recently, or regime strength weakening.
    Binary indicator: 1 if transitioning, 0 if stable.
    Already normalized (binary 0/1). Clipped for safety.
    """
    regime = feature_market_regime(df)
    regime_strength = feature_regime_strength(df)
    change_prob = feature_regime_change_probability(df)
    
    # Regime changed in last 5 days
    regime_changed = (regime != regime.shift(5)).astype(float)
    
    # Regime weakening
    strength_declining = (regime_strength < regime_strength.shift(5)).astype(float)
    
    # High change probability
    high_prob = (change_prob > 0.7).astype(float)
    
    transition = ((regime_changed > 0) | (strength_declining > 0) | (high_prob > 0)).astype(float)
    transition.name = "regime_transition"
    return transition


def feature_regime_stability(df: DataFrame) -> Series:
    """
    Regime Stability: Regime stability measure.
    
    Measures how stable the current regime is.
    Calculated as: inverse of regime change probability and volatility.
    Higher values = more stable, lower = less stable.
    Already normalized to [0, 1]. Clipped for safety.
    """
    change_prob = feature_regime_change_probability(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "regime_stability"
        return result
    
    volatility = spy_close.rolling(window=20, min_periods=1).std() / spy_close
    vol_normalized = volatility / (volatility.rolling(window=252, min_periods=1).max() + 1e-10)
    
    stability = (1.0 - change_prob) * (1.0 - vol_normalized)
    stability = stability.clip(0.0, 1.0)
    stability.name = "regime_stability"
    return stability


def feature_market_sentiment(df: DataFrame) -> Series:
    """
    Market Sentiment: Market sentiment indicator.
    
    Combines multiple signals to measure overall market sentiment.
    Factors: price momentum, volatility, regime strength, trend direction.
    Positive = bullish sentiment, negative = bearish sentiment.
    Normalized to [-1, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "market_sentiment"
        return result
    
    # Price momentum
    momentum = spy_close.pct_change(20)
    
    # Trend direction
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    trend = (spy_close > sma50).astype(float) * 2.0 - 1.0  # -1 to +1
    
    # Regime strength
    regime_strength = feature_regime_strength(df)
    
    # Combine signals
    sentiment = momentum * 10.0 + trend * 0.3 + (regime_strength - 0.5) * 0.2
    sentiment = sentiment / (sentiment.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    sentiment = sentiment.clip(-1.0, 1.0)
    sentiment.name = "market_sentiment"
    return sentiment


def feature_market_fear_greed(df: DataFrame) -> Series:
    """
    Market Fear/Greed: Market fear/greed index proxy.
    
    Proxy for fear/greed index based on:
    - Volatility (high vol = fear, low vol = greed)
    - Momentum (negative = fear, positive = greed)
    - Price position (low = fear, high = greed)
    Normalized to [0, 1]: 0 = extreme fear, 1 = extreme greed.
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_fear_greed"
        return result
    
    # Volatility component (inverse: high vol = fear)
    volatility = spy_close.rolling(window=20, min_periods=1).std() / spy_close
    # Optimized: Use vectorized rank instead of .apply() with lambda
    vol_percentile = volatility.rolling(window=252, min_periods=1).rank(pct=True, method='average')
    vol_percentile = vol_percentile.fillna(0.5)
    vol_component = 1.0 - vol_percentile  # Inverse
    
    # Momentum component
    momentum = spy_close.pct_change(20)
    momentum_normalized = (momentum + 0.1) / 0.2  # Normalize
    momentum_normalized = momentum_normalized.clip(0.0, 1.0)
    
    # Price position component
    sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    price_pos = (spy_close / sma200 - 0.9) / 0.2  # Normalize around 1.0
    price_pos = price_pos.clip(0.0, 1.0)
    
    # Combine (equal weights)
    fear_greed = (vol_component * 0.4 + momentum_normalized * 0.3 + price_pos * 0.3)
    fear_greed = fear_greed.clip(0.0, 1.0)
    fear_greed.name = "market_fear_greed"
    return fear_greed


def feature_regime_consistency(df: DataFrame) -> Series:
    """
    Regime Consistency: Consistency of regime signals.
    
    Measures how consistent different regime indicators are with each other.
    Calculated as: correlation between regime flags over rolling window.
    Higher values = more consistent signals, lower = conflicting signals.
    Already normalized to [0, 1]. Clipped for safety.
    """
    bull = feature_bull_market_flag(df)
    bear = feature_bear_market_flag(df)
    trending = feature_trending_market_flag(df)
    choppy = feature_choppy_market_flag(df)
    
    # Calculate consistency as agreement between signals
    # In bull market: bull=1, bear=0, trending should be high
    # In bear market: bull=0, bear=1, trending should be high
    # In choppy: choppy=1, trending should be low
    
    # Optimized: Use numpy arrays for faster access
    bull_values = bull.values
    bear_values = bear.values
    trending_values = trending.values
    choppy_values = choppy.values
    
    consistency = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_bull = bull_values[i-20:i]
        window_bear = bear_values[i-20:i]
        window_trending = trending_values[i-20:i]
        window_choppy = choppy_values[i-20:i]
        
        # Agreement: bull and bear should be opposite
        # Use numpy mean (faster than pandas)
        bull_bear_agreement = 1.0 - np.mean(window_bull * window_bear)
        
        # Trending should align with bull or bear (not choppy)
        trending_alignment = np.mean(window_trending * (window_bull + window_bear))
        choppy_alignment = np.mean(window_choppy * (1.0 - window_bull - window_bear))
        
        consistency.iloc[i] = (bull_bear_agreement + trending_alignment + choppy_alignment) / 3.0
    
    consistency = consistency.fillna(0.5).clip(0.0, 1.0)
    consistency.name = "regime_consistency"
    return consistency


# ============================================================================
# BLOCK ML-13: Market Direction Features (SPY-based) (12 features)
# ============================================================================


def feature_market_direction(df: DataFrame) -> Series:
    """
    Market Direction: Market direction (up/down/sideways), normalized.
    
    Classifies market direction based on SPY price momentum and trend.
    - 0.0 = Down (bearish)
    - 0.5 = Sideways (neutral)
    - 1.0 = Up (bullish)
    Normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_direction"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    momentum = spy_close.pct_change(20)
    
    # Classify direction
    up = (spy_close > sma50) & (momentum > 0.005)
    down = (spy_close < sma50) & (momentum < -0.005)
    
    direction = pd.Series(0.5, index=df.index)  # Default to sideways
    direction[up] = 1.0  # Up
    direction[down] = 0.0  # Down
    
    direction = direction.clip(0.0, 1.0)
    direction.name = "market_direction"
    return direction


def feature_market_direction_strength(df: DataFrame) -> Series:
    """
    Market Direction Strength: Strength of market direction.
    
    Measures how strong the current market direction is.
    Calculated as: normalized momentum and distance from SMA50.
    Higher values = stronger direction, lower = weaker/mixed.
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_direction_strength"
        return result
    
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    momentum = spy_close.pct_change(20).abs()
    distance = (spy_close - sma50).abs() / spy_close
    
    # Combine signals
    strength = momentum * 20.0 + distance * 5.0
    strength = strength / (strength.rolling(window=252, min_periods=1).max() + 1e-10)
    strength = strength.clip(0.0, 1.0)
    strength.name = "market_direction_strength"
    return strength


def feature_market_trend_alignment(df: DataFrame) -> Series:
    """
    Market Trend Alignment: Stock trend vs market trend alignment.
    
    Measures how aligned the stock's trend is with the market's trend.
    Calculated as: correlation between stock returns and SPY returns over 20 days.
    Positive = aligned (stock moving with market), negative = diverging.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "market_trend_alignment"
        return result
    
    stock_returns = close.pct_change()
    spy_returns = spy_close.pct_change()
    
    # Rolling correlation
    alignment = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        window_stock = stock_returns.iloc[i-20:i]
        window_spy = spy_returns.iloc[i-20:i]
        
        if len(window_stock) < 10 or len(window_spy) < 10:
            alignment.iloc[i] = 0.0
            continue
        
        if window_stock.std() == 0 or window_spy.std() == 0:
            alignment.iloc[i] = 0.0
            continue
        
        corr = window_stock.corr(window_spy)
        alignment.iloc[i] = corr if not pd.isna(corr) else 0.0
    
    alignment = alignment.fillna(0.0).clip(-1.0, 1.0)
    alignment.name = "market_trend_alignment"
    return alignment


def feature_market_momentum(df: DataFrame) -> Series:
    """
    Market Momentum: Market momentum indicator.
    
    Measures market momentum strength based on SPY price changes.
    Calculated as: normalized 20-day return with volatility adjustment.
    Positive = bullish momentum, negative = bearish momentum.
    Normalized to [-1, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "market_momentum"
        return result
    
    momentum = spy_close.pct_change(20)
    volatility = spy_close.rolling(window=20, min_periods=1).std() / spy_close
    
    # Normalize by volatility
    momentum_normalized = momentum / (volatility + 1e-10)
    momentum_normalized = momentum_normalized / (momentum_normalized.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    momentum_normalized = momentum_normalized.clip(-1.0, 1.0)
    momentum_normalized.name = "market_momentum"
    return momentum_normalized


def feature_market_volatility_context(df: DataFrame) -> Series:
    """
    Market Volatility Context: Market volatility context.
    
    Measures current market volatility relative to historical volatility.
    Calculated as: percentile rank of current volatility vs 252-day history.
    Higher values = higher volatility regime, lower = lower volatility.
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_volatility_context"
        return result
    
    volatility = spy_close.rolling(window=20, min_periods=1).std() / spy_close
    
    # Percentile rank over 252-day window
    # Optimized: Use vectorized rank instead of .apply() with lambda
    vol_context = volatility.rolling(window=252, min_periods=1).rank(pct=True, method='average')
    vol_context = vol_context.fillna(0.5)
    
    vol_context = vol_context.clip(0.0, 1.0)
    vol_context.name = "market_volatility_context"
    return vol_context


def feature_market_timing(df: DataFrame) -> Series:
    """
    Market Timing: Market timing indicator.
    
    Combines multiple signals to determine optimal market timing.
    Factors: regime, momentum, volatility, sentiment.
    Positive = good timing (bullish), negative = poor timing (bearish).
    Normalized to [-1, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "market_timing"
        return result
    
    # Get regime signals
    bull = feature_bull_market_flag(df)
    bear = feature_bear_market_flag(df)
    momentum = feature_market_momentum(df)
    sentiment = feature_market_sentiment(df)
    
    # Combine signals
    timing = (bull - bear) * 0.4 + momentum * 0.3 + sentiment * 0.3
    timing = timing.clip(-1.0, 1.0)
    timing.name = "market_timing"
    return timing


def feature_market_support_resistance(df: DataFrame) -> Series:
    """
    Market Support/Resistance: Market support/resistance levels.
    
    Measures how close SPY is to key support/resistance levels.
    Calculated as: distance to 20-day high/low, normalized.
    Higher values = near resistance, lower = near support.
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_support_resistance"
        return result
    
    high_20d = spy_close.rolling(window=20, min_periods=1).max()
    low_20d = spy_close.rolling(window=20, min_periods=1).min()
    
    # Position between support and resistance
    if (high_20d - low_20d).abs().max() < 1e-10:
        position = pd.Series(0.5, index=df.index)
    else:
        position = (spy_close - low_20d) / (high_20d - low_20d)
    
    position = position.clip(0.0, 1.0)
    position.name = "market_support_resistance"
    return position


def feature_market_regime_consistency(df: DataFrame) -> Series:
    """
    Market Regime Consistency: Market regime consistency.
    
    Measures how consistent the market regime has been over time.
    Calculated as: stability of regime signals over rolling window.
    Higher values = more consistent regime, lower = changing regime.
    Already normalized to [0, 1]. Clipped for safety.
    """
    regime = feature_market_regime(df)
    
    # Calculate consistency as inverse of regime changes
    consistency = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        window_regime = regime.iloc[i-20:i]
        # Count regime changes
        changes = (window_regime != window_regime.shift(1)).sum()
        # Consistency = 1 - (changes / window_size)
        consistency.iloc[i] = 1.0 - (changes / len(window_regime))
    
    consistency = consistency.fillna(0.5).clip(0.0, 1.0)
    consistency.name = "market_regime_consistency"
    return consistency


def feature_market_sector_rotation(df: DataFrame) -> Series:
    """
    Market Sector Rotation: Market sector rotation signal.
    
    Proxy for sector rotation based on market behavior patterns.
    Calculated as: momentum divergence and volatility patterns.
    Positive = rotation to risk-on, negative = rotation to risk-off.
    Normalized to [-1, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "market_sector_rotation"
        return result
    
    # Short-term vs long-term momentum
    momentum_short = spy_close.pct_change(5)
    momentum_long = spy_close.pct_change(20)
    
    # Volatility expansion
    volatility = spy_close.rolling(window=20, min_periods=1).std() / spy_close
    vol_expansion = volatility / (volatility.rolling(window=60, min_periods=1).mean() + 1e-10)
    
    # Rotation signal
    rotation = (momentum_short - momentum_long) * 10.0 + (vol_expansion - 1.0) * 0.5
    rotation = rotation / (rotation.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    rotation = rotation.clip(-1.0, 1.0)
    rotation.name = "market_sector_rotation"
    return rotation


def feature_market_breadth(df: DataFrame) -> Series:
    """
    Market Breadth: Market breadth indicator (advancing/declining), proxy.
    
    Proxy for market breadth based on SPY price action patterns.
    Calculated as: consistency of price moves and momentum distribution.
    Higher values = broad participation (bullish), lower = narrow (bearish).
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_breadth"
        return result
    
    # Price consistency (more consistent = broader participation)
    returns = spy_close.pct_change()
    returns_abs = returns.abs()
    
    # Consistency measure
    consistency = 1.0 - (returns_abs.rolling(window=20, min_periods=1).std() / (returns_abs.rolling(window=20, min_periods=1).mean() + 1e-10))
    
    # Momentum strength (stronger momentum = broader participation)
    momentum = spy_close.pct_change(20)
    momentum_strength = momentum.abs() / (momentum.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    
    # Combine
    breadth = (consistency * 0.5 + momentum_strength * 0.5).clip(0.0, 1.0)
    breadth.name = "market_breadth"
    return breadth


def feature_market_leadership(df: DataFrame) -> Series:
    """
    Market Leadership: Market leadership indicator.
    
    Measures market leadership strength (how strongly market is leading).
    Calculated as: momentum persistence and trend strength.
    Higher values = strong leadership, lower = weak leadership.
    Already normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "market_leadership"
        return result
    
    # Momentum persistence
    momentum = spy_close.pct_change(10)
    momentum_sign = np.sign(momentum)
    # Optimized: Vectorized version of momentum persistence
    # Count how many values equal the last value in the window
    momentum_persistence = momentum_sign.rolling(window=20, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0.5,
        raw=False
    )  # Note: This is harder to fully vectorize, but raw=True won't work with iloc[-1]
    # Keeping as-is for now - the performance gain from other optimizations should be sufficient
    
    # Trend strength
    sma50 = spy_close.rolling(window=50, min_periods=1).mean()
    trend_strength = (spy_close - sma50).abs() / spy_close
    
    # Combine
    leadership = (momentum_persistence * 0.6 + trend_strength.clip(0, 0.1) * 5.0 * 0.4).clip(0.0, 1.0)
    leadership.name = "market_leadership"
    return leadership


def feature_market_sentiment_alignment(df: DataFrame) -> Series:
    """
    Market Sentiment Alignment: Stock sentiment vs market sentiment.
    
    Measures how aligned stock sentiment is with market sentiment.
    Calculated as: correlation between stock momentum and market momentum.
    Positive = aligned, negative = diverging.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "market_sentiment_alignment"
        return result
    
    stock_momentum = close.pct_change(20)
    market_momentum = spy_close.pct_change(20)
    
    # Rolling correlation
    alignment = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        window_stock = stock_momentum.iloc[i-20:i]
        window_market = market_momentum.iloc[i-20:i]
        
        if len(window_stock) < 10 or len(window_market) < 10:
            alignment.iloc[i] = 0.0
            continue
        
        if window_stock.std() == 0 or window_market.std() == 0:
            alignment.iloc[i] = 0.0
            continue
        
        corr = window_stock.corr(window_market)
        alignment.iloc[i] = corr if not pd.isna(corr) else 0.0
    
    alignment = alignment.fillna(0.0).clip(-1.0, 1.0)
    alignment.name = "market_sentiment_alignment"
    return alignment


# ============================================================================
# BLOCK ML-14: Relative Strength (SPY-based) (12 features)
# ============================================================================


def feature_relative_strength_spy(df: DataFrame) -> Series:
    """
    Relative Strength vs SPY (20-day): Stock return vs SPY return over 20 days.
    
    Measures how well stock is performing relative to the market (SPY).
    Calculated as: (stock_return_20d - spy_return_20d).
    Positive = outperforming, negative = underperforming.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "relative_strength_spy"
        return result
    
    stock_return = (close - close.shift(20)) / close.shift(20)
    spy_return = (spy_close - spy_close.shift(20)) / spy_close.shift(20)
    
    rs = stock_return - spy_return
    rs.name = "relative_strength_spy"
    return rs


def feature_relative_strength_spy_50d(df: DataFrame) -> Series:
    """
    Relative Strength vs SPY (50-day): Stock return vs SPY return over 50 days.
    
    Measures longer-term relative strength vs market.
    Calculated as: (stock_return_50d - spy_return_50d).
    Positive = outperforming, negative = underperforming.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "relative_strength_spy_50d"
        return result
    
    stock_return = (close - close.shift(50)) / close.shift(50)
    spy_return = (spy_close - spy_close.shift(50)) / spy_close.shift(50)
    
    rs = stock_return - spy_return
    rs.name = "relative_strength_spy_50d"
    return rs


def feature_rs_rank_20d(df: DataFrame) -> Series:
    """
    Relative Strength Rank (20-day): Relative strength rank (0-100) over 20 days.
    
    Ranks current 20-day relative strength vs historical 20-day relative strength.
    Calculated as: percentile rank of current RS vs 252-day history.
    Higher values = stronger relative strength historically.
    Already normalized to [0, 1] (0-100 -> 0-1). Clipped for safety.
    """
    rs_20d = feature_relative_strength_spy(df)
    
    # Calculate percentile rank over 252-day window
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rs_rank = rs_20d.rolling(window=252, min_periods=20).rank(pct=True, method='average') * 100.0
    rs_rank = rs_rank.fillna(50.0)
    
    rs_rank = rs_rank / 100.0  # Normalize to [0, 1]
    rs_rank = rs_rank.clip(0.0, 1.0)
    rs_rank.name = "rs_rank_20d"
    return rs_rank


def feature_rs_rank_50d(df: DataFrame) -> Series:
    """
    Relative Strength Rank (50-day): Relative strength rank (0-100) over 50 days.
    
    Ranks current 50-day relative strength vs historical 50-day relative strength.
    Calculated as: percentile rank of current RS vs 252-day history.
    Higher values = stronger relative strength historically.
    Already normalized to [0, 1] (0-100 -> 0-1). Clipped for safety.
    """
    rs_50d = feature_relative_strength_spy_50d(df)
    
    # Calculate percentile rank over 252-day window
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rs_rank = rs_50d.rolling(window=252, min_periods=50).rank(pct=True, method='average') * 100.0
    rs_rank = rs_rank.fillna(50.0)
    
    rs_rank = rs_rank / 100.0  # Normalize to [0, 1]
    rs_rank = rs_rank.clip(0.0, 1.0)
    rs_rank.name = "rs_rank_50d"
    return rs_rank


def feature_rs_rank_100d(df: DataFrame) -> Series:
    """
    Relative Strength Rank (100-day): Relative strength rank (0-100) over 100 days.
    
    Ranks current 100-day relative strength vs historical 100-day relative strength.
    Calculated as: percentile rank of current RS vs 252-day history.
    Higher values = stronger relative strength historically.
    Already normalized to [0, 1] (0-100 -> 0-1). Clipped for safety.
    """
    close = _get_close_series(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None:
        result = pd.Series(0.5, index=df.index)
        result.name = "rs_rank_100d"
        return result
    
    stock_return = (close - close.shift(100)) / close.shift(100)
    spy_return = (spy_close - spy_close.shift(100)) / spy_close.shift(100)
    rs_100d = stock_return - spy_return
    
    # Calculate percentile rank over 252-day window
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rs_rank = rs_100d.rolling(window=252, min_periods=100).rank(pct=True, method='average') * 100.0
    rs_rank = rs_rank.fillna(50.0)
    
    rs_rank = rs_rank / 100.0  # Normalize to [0, 1]
    rs_rank = rs_rank.clip(0.0, 1.0)
    rs_rank.name = "rs_rank_100d"
    return rs_rank


def feature_rs_momentum(df: DataFrame) -> Series:
    """
    RS Momentum: Rate of change of relative strength.
    
    Measures how quickly relative strength is improving or deteriorating.
    Calculated as: change in 20-day RS over 5 days.
    Positive = RS improving, negative = RS deteriorating.
    No clipping - let ML learn the distribution.
    """
    rs_20d = feature_relative_strength_spy(df)
    
    rs_momentum = rs_20d - rs_20d.shift(5)
    rs_momentum.name = "rs_momentum"
    return rs_momentum


def feature_outperformance_flag(df: DataFrame) -> Series:
    """
    Outperformance Flag: Binary flag for outperforming market.
    
    Indicates if stock is currently outperforming the market.
    Conditions: 20-day RS > 0 AND 50-day RS > 0.
    Binary indicator: 1 if outperforming, 0 otherwise.
    Already normalized (binary 0/1). Clipped for safety.
    """
    rs_20d = feature_relative_strength_spy(df)
    rs_50d = feature_relative_strength_spy_50d(df)
    
    outperforming = ((rs_20d > 0) & (rs_50d > 0)).astype(float)
    outperforming.name = "outperformance_flag"
    return outperforming


def feature_rs_vs_price(df: DataFrame) -> Series:
    """
    RS vs Price: Relative strength vs price divergence.
    
    Measures divergence between relative strength and absolute price performance.
    Calculated as: correlation between RS and price momentum over 20 days.
    Positive = RS and price aligned, negative = diverging.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    rs_20d = feature_relative_strength_spy(df)
    
    price_momentum = close.pct_change(20)
    
    # Rolling correlation
    divergence = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        window_rs = rs_20d.iloc[i-20:i]
        window_price = price_momentum.iloc[i-20:i]
        
        if len(window_rs) < 10 or len(window_price) < 10:
            divergence.iloc[i] = 0.0
            continue
        
        if window_rs.std() == 0 or window_price.std() == 0:
            divergence.iloc[i] = 0.0
            continue
        
        corr = window_rs.corr(window_price)
        divergence.iloc[i] = corr if not pd.isna(corr) else 0.0
    
    divergence = divergence.fillna(0.0).clip(-1.0, 1.0)
    divergence.name = "rs_vs_price"
    return divergence


def feature_rs_consistency(df: DataFrame) -> Series:
    """
    RS Consistency: Consistency of relative strength.
    
    Measures how consistent relative strength has been over time.
    Calculated as: inverse of RS volatility over rolling window.
    Higher values = more consistent RS, lower = volatile RS.
    Already normalized to [0, 1]. Clipped for safety.
    """
    rs_20d = feature_relative_strength_spy(df)
    
    # Calculate consistency as inverse of volatility
    rs_volatility = rs_20d.rolling(window=20, min_periods=10).std()
    rs_mean_abs = rs_20d.abs().rolling(window=20, min_periods=10).mean()
    
    consistency = 1.0 - (rs_volatility / (rs_mean_abs + 1e-10))
    consistency = consistency.clip(0.0, 1.0)
    consistency.name = "rs_consistency"
    return consistency


def feature_rs_regime(df: DataFrame) -> Series:
    """
    RS Regime: Relative strength regime (strong/weak/neutral), normalized.
    
    Classifies relative strength regime based on RS magnitude and persistence.
    - 0.0 = Weak RS (negative, underperforming)
    - 0.5 = Neutral RS (near zero)
    - 1.0 = Strong RS (positive, outperforming)
    Normalized to [0, 1]. Clipped for safety.
    """
    rs_20d = feature_relative_strength_spy(df)
    rs_50d = feature_relative_strength_spy_50d(df)
    
    # Combine short and long-term RS
    rs_combined = (rs_20d * 0.6 + rs_50d * 0.4)
    
    # Normalize to [0, 1] based on historical distribution
    rs_regime = (rs_combined + 0.1) / 0.2  # Normalize: -0.1 to +0.1 -> 0 to 1
    rs_regime = rs_regime.clip(0.0, 1.0)
    rs_regime.name = "rs_regime"
    return rs_regime


def feature_rs_trend(df: DataFrame) -> Series:
    """
    RS Trend: Relative strength trend direction.
    
    Measures the trend direction of relative strength.
    Calculated as: slope of RS over 20 days, normalized.
    Positive = RS trending up, negative = RS trending down.
    Normalized to [-1, 1]. Clipped for safety.
    """
    rs_20d = feature_relative_strength_spy(df)
    
    # Calculate trend as change in RS over 20 days
    rs_trend = (rs_20d - rs_20d.shift(20)) / 0.2  # Normalize by 20 days
    rs_trend = rs_trend / (rs_trend.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    rs_trend = rs_trend.clip(-1.0, 1.0)
    rs_trend.name = "rs_trend"
    return rs_trend


def feature_relative_strength_sector(df: DataFrame) -> Series:
    """
    Relative Strength Sector: Stock vs sector return (proxy using SPY).
    
    Proxy for sector relative strength using SPY as sector proxy.
    Note: True sector data would require sector ETF data.
    Calculated as: stock return vs SPY return (same as relative_strength_spy).
    This is a placeholder that can be enhanced with actual sector data if available.
    No clipping - let ML learn the distribution.
    """
    # For now, use SPY as sector proxy
    # In future, this could use actual sector ETF data if available
    rs_sector = feature_relative_strength_spy(df)
    rs_sector.name = "relative_strength_sector"
    return rs_sector


# ============================================================================
# BLOCK ML-15: Time-Based Features (14 features, skipping days_since_earnings)
# ============================================================================


def feature_day_of_week(df: DataFrame) -> Series:
    """
    Day of Week: Day of week (cyclical encoding: sin/cos combined).
    
    Combines sine and cosine encoding for day of week (0=Monday, 6=Sunday).
    Calculated as: sin(2π * day_of_week / 7) + cos(2π * day_of_week / 7).
    Already normalized to cyclical range. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    day_of_week = dates.dayofweek  # 0=Monday, 6=Sunday
    
    # Cyclical encoding: sin + cos
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    day_combined = day_sin + day_cos
    
    result = pd.Series(day_combined, index=df.index)
    result.name = "day_of_week"
    return result


def feature_day_of_week_sin(df: DataFrame) -> Series:
    """
    Day of Week (Sine): Day of week sine encoding.
    
    Sine encoding for day of week (0=Monday, 6=Sunday).
    Calculated as: sin(2π * day_of_week / 7).
    Already normalized to [-1, 1]. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    day_of_week = dates.dayofweek  # 0=Monday, 6=Sunday
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    
    result = pd.Series(day_sin, index=df.index)
    result.name = "day_of_week_sin"
    return result


def feature_day_of_week_cos(df: DataFrame) -> Series:
    """
    Day of Week (Cosine): Day of week cosine encoding.
    
    Cosine encoding for day of week (0=Monday, 6=Sunday).
    Calculated as: cos(2π * day_of_week / 7).
    Already normalized to [-1, 1]. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    day_of_week = dates.dayofweek  # 0=Monday, 6=Sunday
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    result = pd.Series(day_cos, index=df.index)
    result.name = "day_of_week_cos"
    return result


def feature_day_of_month(df: DataFrame) -> Series:
    """
    Day of Month: Day of month (cyclical encoding: sin/cos combined).
    
    Combines sine and cosine encoding for day of month (1-31).
    Calculated as: sin(2π * day_of_month / 31) + cos(2π * day_of_month / 31).
    Already normalized to cyclical range. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    day_of_month = dates.day  # 1-31
    
    # Cyclical encoding: sin + cos
    day_sin = np.sin(2 * np.pi * day_of_month / 31)
    day_cos = np.cos(2 * np.pi * day_of_month / 31)
    day_combined = day_sin + day_cos
    
    result = pd.Series(day_combined, index=df.index)
    result.name = "day_of_month"
    return result


def feature_day_of_month_sin(df: DataFrame) -> Series:
    """
    Day of Month (Sine): Day of month sine encoding.
    
    Sine encoding for day of month (1-31).
    Calculated as: sin(2π * day_of_month / 31).
    Already normalized to [-1, 1]. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    day_of_month = dates.day  # 1-31
    day_sin = np.sin(2 * np.pi * day_of_month / 31)
    
    result = pd.Series(day_sin, index=df.index)
    result.name = "day_of_month_sin"
    return result


def feature_day_of_month_cos(df: DataFrame) -> Series:
    """
    Day of Month (Cosine): Day of month cosine encoding.
    
    Cosine encoding for day of month (1-31).
    Calculated as: cos(2π * day_of_month / 31).
    Already normalized to [-1, 1]. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    day_of_month = dates.day  # 1-31
    day_cos = np.cos(2 * np.pi * day_of_month / 31)
    
    result = pd.Series(day_cos, index=df.index)
    result.name = "day_of_month_cos"
    return result


def feature_month_of_year(df: DataFrame) -> Series:
    """
    Month of Year: Month of year (cyclical encoding: sin/cos combined).
    
    Combines sine and cosine encoding for month (1-12).
    Calculated as: sin(2π * month / 12) + cos(2π * month / 12).
    Already normalized to cyclical range. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    month = dates.month  # 1-12
    
    # Cyclical encoding: sin + cos
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    month_combined = month_sin + month_cos
    
    result = pd.Series(month_combined, index=df.index)
    result.name = "month_of_year"
    return result


def feature_month_of_year_sin(df: DataFrame) -> Series:
    """
    Month of Year (Sine): Month of year sine encoding.
    
    Sine encoding for month (1-12).
    Calculated as: sin(2π * month / 12).
    Already normalized to [-1, 1]. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    month = dates.month  # 1-12
    month_sin = np.sin(2 * np.pi * month / 12)
    
    result = pd.Series(month_sin, index=df.index)
    result.name = "month_of_year_sin"
    return result


def feature_month_of_year_cos(df: DataFrame) -> Series:
    """
    Month of Year (Cosine): Month of year cosine encoding.
    
    Cosine encoding for month (1-12).
    Calculated as: cos(2π * month / 12).
    Already normalized to [-1, 1]. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    month = dates.month  # 1-12
    month_cos = np.cos(2 * np.pi * month / 12)
    
    result = pd.Series(month_cos, index=df.index)
    result.name = "month_of_year_cos"
    return result


def feature_quarter(df: DataFrame) -> Series:
    """
    Quarter: Quarter (cyclical encoding: sin/cos combined).
    
    Combines sine and cosine encoding for quarter (1-4).
    Calculated as: sin(2π * quarter / 4) + cos(2π * quarter / 4).
    Already normalized to cyclical range. No clipping needed.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    quarter = dates.quarter  # 1-4
    
    # Cyclical encoding: sin + cos
    quarter_sin = np.sin(2 * np.pi * quarter / 4)
    quarter_cos = np.cos(2 * np.pi * quarter / 4)
    quarter_combined = quarter_sin + quarter_cos
    
    result = pd.Series(quarter_combined, index=df.index)
    result.name = "quarter"
    return result


def feature_is_month_end(df: DataFrame) -> Series:
    """
    Is Month End: Binary flag for last 3 days of month.
    
    Indicates if current date is within last 3 days of the month.
    Binary indicator: 1 if month end, 0 otherwise.
    Already normalized (binary 0/1). Clipped for safety.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    day_of_month = dates.day
    days_in_month = dates.days_in_month
    
    is_month_end = (day_of_month >= (days_in_month - 2)).astype(float)
    
    result = pd.Series(is_month_end, index=df.index)
    result.name = "is_month_end"
    return result


def feature_is_quarter_end(df: DataFrame) -> Series:
    """
    Is Quarter End: Binary flag for last week of quarter.
    
    Indicates if current date is within last 5 trading days of the quarter.
    Binary indicator: 1 if quarter end, 0 otherwise.
    Already normalized (binary 0/1). Clipped for safety.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    month = dates.month
    day_of_month = dates.day
    days_in_month = dates.days_in_month
    
    # Quarter ends: March (3), June (6), September (9), December (12)
    is_quarter_month = month.isin([3, 6, 9, 12])
    is_last_week = day_of_month >= (days_in_month - 4)
    
    is_quarter_end = (is_quarter_month & is_last_week).astype(float)
    
    result = pd.Series(is_quarter_end, index=df.index)
    result.name = "is_quarter_end"
    return result


def feature_is_year_end(df: DataFrame) -> Series:
    """
    Is Year End: Binary flag for December.
    
    Indicates if current date is in December (year-end period).
    Binary indicator: 1 if December, 0 otherwise.
    Already normalized (binary 0/1). Clipped for safety.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    month = dates.month
    is_year_end = (month == 12).astype(float)
    
    result = pd.Series(is_year_end, index=df.index)
    result.name = "is_year_end"
    return result


def feature_trading_day_of_year(df: DataFrame) -> Series:
    """
    Trading Day of Year: Trading day of year (1-252), normalized.
    
    Counts trading days from start of year, normalized to [0, 1].
    Calculated as: cumulative trading day count / 252.
    Already normalized to [0, 1]. Clipped for safety.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        dates = pd.to_datetime(df.index)
    else:
        dates = df.index
    
    # Group by year and count trading days
    trading_day = pd.Series(index=df.index, dtype=float)
    
    for year in dates.year.unique():
        year_mask = dates.year == year
        year_dates = dates[year_mask]
        
        # Count trading days (assuming all dates in index are trading days)
        day_count = pd.Series(range(1, len(year_dates) + 1), index=year_dates)
        trading_day.loc[year_mask] = day_count.values
    
    # Normalize by 252 (typical trading days per year)
    trading_day_normalized = trading_day / 252.0
    trading_day_normalized = trading_day_normalized.clip(0.0, 1.0)
    trading_day_normalized.name = "trading_day_of_year"
    return trading_day_normalized


# ============================================================================
# BLOCK ML-16: Statistical Features (15 features, volume_autocorrelation already exists)
# ============================================================================


def feature_price_skewness_20d(df: DataFrame) -> Series:
    """
    Price Skewness (20-day): Price distribution skewness over 20 days.
    
    Measures asymmetry of price distribution over rolling 20-day window.
    Positive = right-skewed (tail on right), negative = left-skewed (tail on left).
    Calculated using scipy.stats.skew or manual calculation.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Optimized: Use pandas' built-in skew() instead of .apply() with scipy
    # Note: pandas skew() uses Fisher's definition (unbiased), scipy uses Pearson's
    # For performance, we use pandas which is vectorized and much faster
    skewness = close.rolling(window=20, min_periods=10).skew()
    skewness = skewness.fillna(0.0)
    
    skewness.name = "price_skewness_20d"
    return skewness


def feature_price_kurtosis_20d(df: DataFrame) -> Series:
    """
    Price Kurtosis (20-day): Price distribution kurtosis over 20 days.
    
    Measures tail heaviness of price distribution over rolling 20-day window.
    Positive = heavy tails (more extreme values), negative = light tails.
    Calculated using scipy.stats.kurtosis or manual calculation.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Optimized: Use pandas' built-in kurt() instead of .apply() with scipy
    # Note: pandas uses .kurt() not .kurtosis(), and uses Fisher's definition (excess kurtosis)
    # For performance, we use pandas which is vectorized and much faster
    kurtosis = close.rolling(window=20, min_periods=10).kurt()
    kurtosis = kurtosis.fillna(0.0)
    
    kurtosis.name = "price_kurtosis_20d"
    return kurtosis


def feature_returns_skewness_20d(df: DataFrame) -> Series:
    """
    Returns Skewness (20-day): Return distribution skewness over 20 days.
    
    Measures asymmetry of return distribution over rolling 20-day window.
    Positive = right-skewed (more positive outliers), negative = left-skewed (more negative outliers).
    Calculated using scipy.stats.skew.
    No clipping - let ML learn the distribution.
    """
    from scipy import stats
    
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # Optimized: Use pandas' built-in skew() instead of .apply() with scipy
    skewness = returns.rolling(window=20, min_periods=10).skew()
    skewness = skewness.fillna(0.0)
    
    skewness.name = "returns_skewness_20d"
    return skewness


def feature_returns_kurtosis_20d(df: DataFrame) -> Series:
    """
    Returns Kurtosis (20-day): Return distribution kurtosis over 20 days.
    
    Measures tail heaviness of return distribution over rolling 20-day window.
    Positive = heavy tails (more extreme returns), negative = light tails.
    Calculated using scipy.stats.kurtosis.
    No clipping - let ML learn the distribution.
    """
    from scipy import stats
    
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # Optimized: Use pandas' built-in kurt() instead of .apply() with scipy
    # Note: pandas uses .kurt() not .kurtosis()
    kurtosis = returns.rolling(window=20, min_periods=10).kurt()
    kurtosis = kurtosis.fillna(0.0)
    
    kurtosis.name = "returns_kurtosis_20d"
    return kurtosis


def feature_price_autocorrelation_1d(df: DataFrame) -> Series:
    """
    Price Autocorrelation (1-day): Price autocorrelation with 1-day lag.
    
    Measures correlation between price and price lagged by 1 day.
    Positive = momentum, negative = mean-reversion.
    Calculated as: correlation(price[t], price[t-1]) over rolling window.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Optimized: Use numpy arrays for faster access
    close_values = close.values
    
    autocorr = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window = close_values[i-20:i]
        if len(window) < 10:
            autocorr.iloc[i] = 0.0
            continue
        
        window_std = np.std(window, ddof=0)
        if window_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag = close_values[i-21:i-1] if i >= 21 else close_values[i-20:i]
        if len(window_lag) < 10:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag_std = np.std(window_lag, ddof=0)
        if window_lag_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        # Align windows
        min_len = min(len(window), len(window_lag))
        window_aligned = window[-min_len:]
        window_lag_aligned = window_lag[-min_len:]
        
        # Use numpy correlation (faster than pandas)
        corr = np.corrcoef(window_aligned, window_lag_aligned)[0, 1]
        autocorr.iloc[i] = corr if not np.isnan(corr) else 0.0
    
    autocorr = autocorr.fillna(0.0).clip(-1.0, 1.0)
    autocorr.name = "price_autocorrelation_1d"
    return autocorr


def feature_price_autocorrelation_5d(df: DataFrame) -> Series:
    """
    Price Autocorrelation (5-day): Price autocorrelation with 5-day lag.
    
    Measures correlation between price and price lagged by 5 days.
    Positive = momentum, negative = mean-reversion.
    Calculated as: correlation(price[t], price[t-5]) over rolling window.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Optimized: Use numpy arrays for faster access
    close_values = close.values
    
    autocorr = pd.Series(index=df.index, dtype=float)
    
    for i in range(25, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window = close_values[i-20:i]
        if len(window) < 10:
            autocorr.iloc[i] = 0.0
            continue
        
        window_std = np.std(window, ddof=0)
        if window_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag = close_values[i-25:i-5] if i >= 25 else close_values[i-20:i]
        if len(window_lag) < 10:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag_std = np.std(window_lag, ddof=0)
        if window_lag_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        # Align windows
        min_len = min(len(window), len(window_lag))
        window_aligned = window[-min_len:]
        window_lag_aligned = window_lag[-min_len:]
        
        # Use numpy correlation (faster than pandas)
        corr = np.corrcoef(window_aligned, window_lag_aligned)[0, 1]
        autocorr.iloc[i] = corr if not np.isnan(corr) else 0.0
    
    autocorr = autocorr.fillna(0.0).clip(-1.0, 1.0)
    autocorr.name = "price_autocorrelation_5d"
    return autocorr


def feature_returns_autocorrelation(df: DataFrame) -> Series:
    """
    Returns Autocorrelation: Return autocorrelation over 20 days.
    
    Measures correlation between returns and lagged returns.
    Positive = momentum, negative = mean-reversion.
    Calculated as: correlation(returns[t], returns[t-1]) over rolling window.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # Optimized: Use numpy arrays for faster access
    returns_values = returns.values
    
    autocorr = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window = returns_values[i-20:i]
        if len(window) < 10:
            autocorr.iloc[i] = 0.0
            continue
        
        window_std = np.std(window)
        if window_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag = returns_values[i-21:i-1] if i >= 21 else returns_values[i-20:i]
        if len(window_lag) < 10:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag_std = np.std(window_lag)
        if window_lag_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        # Align windows
        min_len = min(len(window), len(window_lag))
        window_aligned = window[-min_len:]
        window_lag_aligned = window_lag[-min_len:]
        
        # Use numpy correlation (faster than pandas)
        corr = np.corrcoef(window_aligned, window_lag_aligned)[0, 1]
        if np.isnan(corr):
            autocorr.iloc[i] = 0.0
            continue
        
        autocorr.iloc[i] = corr
    
    autocorr = autocorr.fillna(0.0).clip(-1.0, 1.0)
    autocorr.name = "returns_autocorrelation"
    return autocorr


def feature_price_variance_ratio(df: DataFrame) -> Series:
    """
    Price Variance Ratio: Variance ratio test statistic.
    
    Tests for random walk vs mean-reversion.
    Calculated as: Var(k-period returns) / (k * Var(1-period returns)).
    VR > 1 = trending, VR < 1 = mean-reverting, VR ≈ 1 = random walk.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Calculate 1-period and 5-period returns
    returns_1d = close.pct_change()
    returns_5d = (close - close.shift(5)) / close.shift(5)
    
    # Optimized: Use numpy arrays for faster access
    returns_1d_values = returns_1d.values
    returns_5d_values = returns_5d.values
    
    variance_ratio = pd.Series(index=df.index, dtype=float)
    
    for i in range(50, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_1d = returns_1d_values[i-50:i]
        window_5d = returns_5d_values[i-50:i]
        
        if len(window_1d) < 20 or len(window_5d) < 20:
            variance_ratio.iloc[i] = 1.0
            continue
        
        # Use numpy variance (faster than pandas)
        var_1d = np.var(window_1d, ddof=0)
        var_5d = np.var(window_5d, ddof=0)
        
        if var_1d == 0:
            variance_ratio.iloc[i] = 1.0
            continue
        
        vr = var_5d / (5.0 * var_1d)
        variance_ratio.iloc[i] = vr if not pd.isna(vr) else 1.0
    
    variance_ratio = variance_ratio.fillna(1.0)
    variance_ratio.name = "price_variance_ratio"
    return variance_ratio


def feature_price_half_life(df: DataFrame) -> Series:
    """
    Price Half-Life: Price mean-reversion half-life.
    
    Measures speed of mean-reversion in price.
    Calculated using OLS regression: log(price) = α + β*log(price_lag).
    Half-life = -log(2) / log(β).
    Normalized by dividing by 252. No clipping - let ML learn the distribution.
    """
    from sklearn.linear_model import LinearRegression
    
    close = _get_close_series(df)
    log_price = np.log(close)
    
    # Optimized: Use numpy arrays for faster access
    log_price_values = log_price.values
    
    half_life = pd.Series(index=df.index, dtype=float)
    
    for i in range(50, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window = log_price_values[i-50:i].reshape(-1, 1)
        window_lag = log_price_values[i-51:i-1].reshape(-1, 1) if i >= 51 else log_price_values[i-50:i].reshape(-1, 1)
        
        if len(window) < 20 or len(window_lag) < 20:
            half_life.iloc[i] = 126.0 / 252.0  # Default to 0.5 (half year)
            continue
        
        # Align windows
        min_len = min(len(window), len(window_lag))
        window_aligned = window[-min_len:]
        window_lag_aligned = window_lag[-min_len:]
        
        try:
            reg = LinearRegression()
            reg.fit(window_lag_aligned, window_aligned)
            beta = reg.coef_[0][0]
            
            if beta >= 1.0 or beta <= 0:
                half_life.iloc[i] = 126.0 / 252.0
                continue
            
            hl = -np.log(2) / np.log(beta)
            half_life.iloc[i] = hl / 252.0  # Normalize by 252
        except:
            half_life.iloc[i] = 126.0 / 252.0
    
    half_life = half_life.fillna(126.0 / 252.0)
    half_life.name = "price_half_life"
    return half_life


def feature_returns_half_life(df: DataFrame) -> Series:
    """
    Returns Half-Life: Return mean-reversion half-life.
    
    Measures speed of mean-reversion in returns.
    Calculated using OLS regression: returns = α + β*returns_lag.
    Half-life = -log(2) / log(β).
    Normalized by dividing by 252. No clipping - let ML learn the distribution.
    """
    from sklearn.linear_model import LinearRegression
    
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # Optimized: Use numpy arrays for faster access
    returns_values = returns.values
    
    half_life = pd.Series(index=df.index, dtype=float)
    
    for i in range(50, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window = returns_values[i-50:i].reshape(-1, 1)
        window_lag = returns_values[i-51:i-1].reshape(-1, 1) if i >= 51 else returns_values[i-50:i].reshape(-1, 1)
        
        if len(window) < 20 or len(window_lag) < 20:
            half_life.iloc[i] = 126.0 / 252.0  # Default to 0.5
            continue
        
        # Align windows
        min_len = min(len(window), len(window_lag))
        window_aligned = window[-min_len:]
        window_lag_aligned = window_lag[-min_len:]
        
        try:
            reg = LinearRegression()
            reg.fit(window_lag_aligned, window_aligned)
            beta = reg.coef_[0][0]
            
            if beta >= 1.0 or beta <= 0:
                half_life.iloc[i] = 126.0 / 252.0
                continue
            
            hl = -np.log(2) / np.log(beta)
            half_life.iloc[i] = hl / 252.0  # Normalize by 252
        except:
            half_life.iloc[i] = 126.0 / 252.0
    
    half_life = half_life.fillna(126.0 / 252.0)
    half_life.name = "returns_half_life"
    return half_life


def feature_price_stationarity(df: DataFrame) -> Series:
    """
    Price Stationarity: Price stationarity test (ADF-like proxy).
    
    Proxy for Augmented Dickey-Fuller test using variance ratio.
    Calculated as: inverse of variance ratio (lower = more stationary).
    Normalized to [0, 1]. Clipped for safety.
    """
    variance_ratio = feature_price_variance_ratio(df)
    
    # Inverse of variance ratio as stationarity proxy
    stationarity = 1.0 / (variance_ratio + 1e-10)
    stationarity = stationarity / (stationarity.rolling(window=252, min_periods=1).max() + 1e-10)
    stationarity = stationarity.clip(0.0, 1.0)
    stationarity.name = "price_stationarity"
    return stationarity


def feature_returns_stationarity(df: DataFrame) -> Series:
    """
    Returns Stationarity: Return stationarity test (ADF-like proxy).
    
    Proxy for Augmented Dickey-Fuller test using return variance ratio.
    Returns are typically more stationary than prices.
    Normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # Calculate variance ratio for returns
    returns_1d = returns
    returns_5d = returns.rolling(window=5, min_periods=1).sum()
    
    variance_ratio = pd.Series(index=df.index, dtype=float)
    
    for i in range(50, len(df)):
        window_1d = returns_1d.iloc[i-50:i]
        window_5d = returns_5d.iloc[i-50:i]
        
        if len(window_1d) < 20 or len(window_5d) < 20:
            variance_ratio.iloc[i] = 1.0
            continue
        
        var_1d = window_1d.var()
        var_5d = window_5d.var()
        
        if var_1d == 0:
            variance_ratio.iloc[i] = 1.0
            continue
        
        vr = var_5d / (5.0 * var_1d)
        variance_ratio.iloc[i] = vr if not pd.isna(vr) else 1.0
    
    variance_ratio = variance_ratio.fillna(1.0)
    
    # Inverse as stationarity proxy
    stationarity = 1.0 / (variance_ratio + 1e-10)
    stationarity = stationarity / (stationarity.rolling(window=252, min_periods=1).max() + 1e-10)
    stationarity = stationarity.clip(0.0, 1.0)
    stationarity.name = "returns_stationarity"
    return stationarity


def feature_price_cointegration(df: DataFrame) -> Series:
    """
    Price Cointegration: Price cointegration with market (SPY).
    
    Measures long-term relationship between stock and market prices.
    Calculated as: correlation of price differences with SPY price differences.
    Higher values = more cointegrated (move together long-term).
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None:
        result = pd.Series(0.0, index=df.index)
        result.name = "price_cointegration"
        return result
    
    # Price differences (first differences)
    price_diff = close.diff()
    spy_diff = spy_close.diff()
    
    # Rolling correlation
    cointegration = pd.Series(index=df.index, dtype=float)
    
    for i in range(50, len(df)):
        window_price = price_diff.iloc[i-50:i]
        window_spy = spy_diff.iloc[i-50:i]
        
        if len(window_price) < 20 or len(window_spy) < 20:
            cointegration.iloc[i] = 0.0
            continue
        
        if window_price.std() == 0 or window_spy.std() == 0:
            cointegration.iloc[i] = 0.0
            continue
        
        corr = window_price.corr(window_spy)
        cointegration.iloc[i] = corr if not pd.isna(corr) else 0.0
    
    cointegration = cointegration.fillna(0.0).clip(-1.0, 1.0)
    cointegration.name = "price_cointegration"
    return cointegration


def feature_statistical_regime(df: DataFrame) -> Series:
    """
    Statistical Regime: Statistical regime (trending/mean-reverting/random), normalized.
    
    Classifies market regime based on statistical properties.
    - 0.0 = Mean-reverting (negative autocorrelation, VR < 1)
    - 0.5 = Random walk (low autocorrelation, VR ≈ 1)
    - 1.0 = Trending (positive autocorrelation, VR > 1)
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get autocorrelation and variance ratio
    autocorr = feature_returns_autocorrelation(df)
    variance_ratio = feature_price_variance_ratio(df)
    
    # Combine signals
    # Autocorrelation component: positive = trending, negative = mean-reverting
    autocorr_component = (autocorr + 1.0) / 2.0  # Normalize to [0, 1]
    
    # Variance ratio component: >1 = trending, <1 = mean-reverting
    vr_component = variance_ratio / (variance_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    vr_component = vr_component.clip(0.0, 1.0)
    
    # Combine (equal weights)
    regime = (autocorr_component * 0.5 + vr_component * 0.5)
    regime = regime.clip(0.0, 1.0)
    regime.name = "statistical_regime"
    return regime


# ============================================================================
# BLOCK ML-17: Advanced Volatility Features (12 features)
# ============================================================================


def feature_volatility_skewness(df: DataFrame) -> Series:
    """
    Volatility Skewness: Volatility distribution skewness.
    
    Measures asymmetry of volatility distribution over rolling window.
    Positive = right-skewed (more high volatility periods), negative = left-skewed.
    Calculated using scipy.stats.skew on volatility series.
    No clipping - let ML learn the distribution.
    """
    from scipy import stats
    
    close = _get_close_series(df)
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    # Optimized: Use pandas' built-in skew() instead of .apply() with scipy
    skewness = volatility.rolling(window=60, min_periods=20).skew()
    skewness = skewness.fillna(0.0)
    
    skewness.name = "volatility_skewness"
    return skewness


def feature_volatility_kurtosis(df: DataFrame) -> Series:
    """
    Volatility Kurtosis: Volatility distribution kurtosis.
    
    Measures tail heaviness of volatility distribution over rolling window.
    Positive = heavy tails (more extreme volatility), negative = light tails.
    Calculated using scipy.stats.kurtosis on volatility series.
    No clipping - let ML learn the distribution.
    """
    from scipy import stats
    
    close = _get_close_series(df)
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    # Optimized: Use pandas' built-in kurt() instead of .apply() with scipy
    # Note: pandas uses .kurt() not .kurtosis()
    kurtosis = volatility.rolling(window=60, min_periods=20).kurt()
    kurtosis = kurtosis.fillna(0.0)
    
    kurtosis.name = "volatility_kurtosis"
    return kurtosis


def feature_volatility_autocorrelation(df: DataFrame) -> Series:
    """
    Volatility Autocorrelation: Volatility autocorrelation.
    
    Measures correlation between volatility and lagged volatility.
    High autocorrelation = volatility clustering (GARCH effect).
    Calculated as: correlation(volatility[t], volatility[t-1]) over rolling window.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    # Optimized: Use numpy arrays for faster access
    volatility_values = volatility.values
    
    autocorr = pd.Series(index=df.index, dtype=float)
    
    for i in range(40, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window = volatility_values[i-40:i]
        if len(window) < 20:
            autocorr.iloc[i] = 0.0
            continue
        
        window_std = np.std(window, ddof=0)
        if window_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag = volatility_values[i-41:i-1] if i >= 41 else volatility_values[i-40:i]
        if len(window_lag) < 20:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag_std = np.std(window_lag, ddof=0)
        if window_lag_std == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        # Align windows
        min_len = min(len(window), len(window_lag))
        window_aligned = window[-min_len:]
        window_lag_aligned = window_lag[-min_len:]
        
        # Use numpy correlation (faster than pandas)
        corr = np.corrcoef(window_aligned, window_lag_aligned)[0, 1]
        autocorr.iloc[i] = corr if not np.isnan(corr) else 0.0
    
    autocorr = autocorr.fillna(0.0).clip(-1.0, 1.0)
    autocorr.name = "volatility_autocorrelation"
    return autocorr


def feature_volatility_mean_reversion(df: DataFrame) -> Series:
    """
    Volatility Mean-Reversion: Volatility mean-reversion speed.
    
    Measures how quickly volatility reverts to its mean.
    Calculated using OLS regression: volatility = α + β*volatility_lag.
    Half-life = -log(2) / log(β), normalized by 252.
    Higher values = faster mean-reversion.
    No clipping - let ML learn the distribution.
    """
    from sklearn.linear_model import LinearRegression
    
    close = _get_close_series(df)
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    # Optimized: Use numpy arrays for faster access
    volatility_values = volatility.values
    
    mean_reversion = pd.Series(index=df.index, dtype=float)
    
    for i in range(60, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window = volatility_values[i-60:i].reshape(-1, 1)
        window_lag = volatility_values[i-61:i-1].reshape(-1, 1) if i >= 61 else volatility_values[i-60:i].reshape(-1, 1)
        
        if len(window) < 30 or len(window_lag) < 30:
            mean_reversion.iloc[i] = 126.0 / 252.0  # Default to 0.5
            continue
        
        # Align windows
        min_len = min(len(window), len(window_lag))
        window_aligned = window[-min_len:]
        window_lag_aligned = window_lag[-min_len:]
        
        try:
            reg = LinearRegression()
            reg.fit(window_lag_aligned, window_aligned)
            beta = reg.coef_[0][0]
            
            if beta >= 1.0 or beta <= 0:
                mean_reversion.iloc[i] = 126.0 / 252.0
                continue
            
            hl = -np.log(2) / np.log(beta)
            mean_reversion.iloc[i] = hl / 252.0  # Normalize by 252
        except:
            mean_reversion.iloc[i] = 126.0 / 252.0
    
    mean_reversion = mean_reversion.fillna(126.0 / 252.0)
    mean_reversion.name = "volatility_mean_reversion"
    return mean_reversion


def feature_volatility_of_volatility(df: DataFrame) -> Series:
    """
    Volatility of Volatility (VoV): Improved VoV measure.
    
    Measures variability of volatility itself (second-order volatility).
    Calculated as: std(volatility) / mean(volatility) over rolling window.
    Higher values = more volatile volatility (regime changes).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    vov = volatility.rolling(window=60, min_periods=20).std() / (volatility.rolling(window=60, min_periods=20).mean() + 1e-10)
    
    vov.name = "volatility_of_volatility"
    return vov


def feature_volatility_regime_persistence(df: DataFrame) -> Series:
    """
    Volatility Regime Persistence: Volatility regime persistence.
    
    Measures how persistent volatility regimes are over time.
    Calculated as: consistency of volatility regime classification over rolling window.
    Higher values = more persistent regimes, lower = more switching.
    Already normalized to [0, 1]. Clipped for safety.
    """
    # Get volatility regime from existing feature
    volatility_regime = feature_volatility_regime(df)
    
    persistence = pd.Series(index=df.index, dtype=float)
    
    for i in range(40, len(df)):
        window_regime = volatility_regime.iloc[i-40:i]
        # Count regime changes
        changes = (window_regime != window_regime.shift(1)).sum()
        # Persistence = 1 - (changes / window_size)
        persistence.iloc[i] = 1.0 - (changes / len(window_regime))
    
    persistence = persistence.fillna(0.5).clip(0.0, 1.0)
    persistence.name = "volatility_regime_persistence"
    return persistence


def feature_volatility_shock(df: DataFrame) -> Series:
    """
    Volatility Shock: Recent volatility shock indicator.
    
    Measures if there was a recent significant volatility increase.
    Calculated as: current volatility / rolling mean volatility.
    Higher values = recent shock, lower = normal volatility.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    vol_mean = volatility.rolling(window=60, min_periods=20).mean()
    shock = volatility / (vol_mean + 1e-10)
    
    shock.name = "volatility_shock"
    return shock


def feature_volatility_normalized_returns(df: DataFrame) -> Series:
    """
    Volatility Normalized Returns: Returns normalized by volatility.
    
    Measures risk-adjusted returns (Sharpe-like).
    Calculated as: returns / volatility.
    Higher values = better risk-adjusted performance.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    returns = close.pct_change()
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    normalized = returns / (volatility + 1e-10)
    
    normalized.name = "volatility_normalized_returns"
    return normalized


def feature_volatility_forecast(df: DataFrame) -> Series:
    """
    Volatility Forecast: Volatility forecast (GARCH-like).
    
    Simple GARCH-like volatility forecast using EWMA.
    Calculated as: weighted average of past volatility and squared returns.
    Formula: σ²_t = α*σ²_{t-1} + (1-α)*r²_t, where α=0.94 (typical GARCH).
    Optimized: Uses vectorized pandas .ewm() instead of Python loop for better performance.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # GARCH-like forecast: EWMA of squared returns
    # alpha=0.94 means smoothing factor (1-alpha)=0.06 in EWMA formula
    squared_returns = returns ** 2
    
    # Vectorized EWMA: y_t = 0.94 * y_{t-1} + 0.06 * x_t
    # pandas ewm(alpha=0.06) gives: y_t = (1-0.06) * y_{t-1} + 0.06 * x_t = 0.94 * y_{t-1} + 0.06 * x_t
    # adjust=False means use the recursive formula directly (no bias correction)
    forecast_var = squared_returns.ewm(alpha=0.06, adjust=False, ignore_na=True).mean()
    
    # Convert to volatility (square root)
    forecast_vol = np.sqrt(forecast_var)
    forecast_vol.name = "volatility_forecast"
    return forecast_vol


def feature_volatility_term_structure(df: DataFrame) -> Series:
    """
    Volatility Term Structure: Volatility term structure (short vs long).
    
    Measures difference between short-term and long-term volatility.
    Calculated as: (short_vol - long_vol) / long_vol.
    Positive = short-term volatility higher (stress), negative = long-term higher (stability).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    short_vol = close.rolling(window=10, min_periods=1).std() / close
    long_vol = close.rolling(window=60, min_periods=1).std() / close
    
    term_structure = (short_vol - long_vol) / (long_vol + 1e-10)
    
    term_structure.name = "volatility_term_structure"
    return term_structure


def feature_volatility_risk_premium(df: DataFrame) -> Series:
    """
    Volatility Risk Premium: Volatility risk premium proxy.
    
    Proxy for volatility risk premium using realized vs forecast volatility.
    Calculated as: (realized_vol - forecast_vol) / forecast_vol.
    Positive = risk premium (realized > forecast), negative = risk discount.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # Realized volatility
    realized_vol = returns.rolling(window=20, min_periods=1).std()
    
    # Forecast volatility (simple EWMA) - vectorized
    squared_returns = returns ** 2
    # Vectorized EWMA: same as feature_volatility_forecast
    forecast_var = squared_returns.ewm(alpha=0.06, adjust=False, ignore_na=True).mean()
    
    forecast_vol = np.sqrt(forecast_var)
    
    # Risk premium
    premium = (realized_vol - forecast_vol) / (forecast_vol + 1e-10)
    
    premium.name = "volatility_risk_premium"
    return premium


def feature_volatility_smoothness(df: DataFrame) -> Series:
    """
    Volatility Smoothness: Volatility smoothness measure.
    
    Measures how smooth/stable volatility is (inverse of volatility of volatility).
    Calculated as: 1 / (1 + VoV), where VoV = std(volatility) / mean(volatility).
    Higher values = smoother volatility, lower = more erratic.
    Already normalized to [0, 1]. Clipped for safety.
    """
    vov = feature_volatility_of_volatility(df)
    
    smoothness = 1.0 / (1.0 + vov.abs())
    smoothness = smoothness.clip(0.0, 1.0)
    smoothness.name = "volatility_smoothness"
    return smoothness


# ============================================================================
# BLOCK ML-18: Advanced Technical Indicators (8 features)
# ============================================================================


def feature_bollinger_band_width(df: DataFrame) -> Series:
    """
    Bollinger Band Width: Bollinger Band Width (log normalized).
    
    Measures how "tight" or "expanded" price is relative to its recent volatility.
    BB Width is one of the best predictors of breakouts and volatility expansions.
    Calculated as: log1p((upper_band - lower_band) / mid_band).
    Log normalization compresses range while handling small values gracefully.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Calculate middle band (SMA20)
    mid = close.rolling(window=20, min_periods=1).mean()
    
    # Calculate standard deviation (20-period)
    std = close.rolling(window=20, min_periods=1).std()
    
    # Calculate upper and lower bands
    upper = mid + 2 * std
    lower = mid - 2 * std
    
    # Calculate Bollinger Band Width
    mid_safe = mid.replace(0, np.nan)
    bbw = (upper - lower) / mid_safe
    
    # Log normalization
    bbw_log = np.log1p(bbw)
    
    bbw_log.name = "bollinger_band_width"
    return bbw_log


def feature_fractal_dimension_index(df: DataFrame) -> Series:
    """
    Fractal Dimension Index: Fractal Dimension Index normalized to [0, 1].
    
    Measures complexity/roughness of price movement.
    Calculated using Higuchi's method or similar fractal dimension estimation.
    Higher values = more complex/rough, lower = smoother.
    Normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Simplified fractal dimension using price range vs price movement
    # More sophisticated methods exist but this is a good proxy
    window = 20
    
    fractal = pd.Series(index=df.index, dtype=float)
    
    for i in range(window, len(df)):
        window_prices = close.iloc[i-window:i]
        
        if len(window_prices) < window:
            fractal.iloc[i] = 0.5
            continue
        
        # Calculate total path length
        path_length = window_prices.diff().abs().sum()
        
        # Calculate straight-line distance
        straight_distance = abs(window_prices.iloc[-1] - window_prices.iloc[0])
        
        if straight_distance == 0:
            fractal.iloc[i] = 0.5
            continue
        
        # Fractal dimension proxy: path_length / straight_distance
        # Normalize to [0, 1] range
        fd_proxy = path_length / (straight_distance + 1e-10)
        fractal.iloc[i] = min(fd_proxy / 2.0, 1.0)  # Normalize (typical range 1-2)
    
    fractal = fractal.fillna(0.5).clip(0.0, 1.0)
    fractal.name = "fractal_dimension_index"
    return fractal


def feature_hurst_exponent(df: DataFrame) -> Series:
    """
    Hurst Exponent: Hurst Exponent in [0, 1].
    
    Measures long-term memory in price series.
    - H > 0.5 = trending (persistent)
    - H < 0.5 = mean-reverting (anti-persistent)
    - H ≈ 0.5 = random walk
    Calculated using rescaled range (R/S) analysis.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    
    # Optimized: Use numpy arrays for faster access
    close_values = close.values
    
    hurst = pd.Series(index=df.index, dtype=float)
    
    for i in range(100, len(df)):
        # Use numpy array slicing (much faster than pandas iloc)
        window_values = close_values[i-100:i]
        
        if len(window_values) < 50:
            hurst.iloc[i] = 0.5
            continue
        
        # Calculate returns directly with numpy (much faster than pandas pct_change)
        # returns = (window[1:] - window[:-1]) / window[:-1]
        returns_values = (window_values[1:] - window_values[:-1]) / window_values[:-1]
        
        # Remove NaN and inf values
        returns_values = returns_values[~np.isnan(returns_values)]
        returns_values = returns_values[np.isfinite(returns_values)]
        
        if len(returns_values) < 20:
            hurst.iloc[i] = 0.5
            continue
        
        # Rescaled Range (R/S) analysis
        # Use numpy for faster calculations
        mean_return = np.mean(returns_values)
        deviations = returns_values - mean_return
        cumulative_deviations = np.cumsum(deviations)
        
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)  # Range
        S = np.std(returns_values, ddof=0)  # Standard deviation (population std)
        
        if S == 0:
            hurst.iloc[i] = 0.5
            continue
        
        # R/S ratio
        rs_ratio = R / S if S > 0 else 1.0
        
        # Hurst exponent estimate (simplified)
        # H = log(R/S) / log(n), but we use a rolling window approach
        # For simplicity, use a proxy based on autocorrelation
        # Optimized: Use numpy correlation instead of pandas autocorr
        if len(returns_values) > 1:
            autocorr = np.corrcoef(returns_values[:-1], returns_values[1:])[0, 1]
        else:
            autocorr = 0.0
        
        if np.isnan(autocorr):
            hurst.iloc[i] = 0.5
            continue
        
        # Convert autocorrelation to Hurst-like measure
        # Positive autocorr -> H > 0.5, negative -> H < 0.5
        H = 0.5 + autocorr * 0.3  # Scale autocorr to Hurst range
        hurst.iloc[i] = np.clip(H, 0.0, 1.0)
    
    hurst = hurst.fillna(0.5).clip(0.0, 1.0)
    hurst.name = "hurst_exponent"
    return hurst


def feature_price_curvature(df: DataFrame) -> Series:
    """
    Price Curvature: Price Curvature (improved).
    
    Measures the second derivative of price (acceleration/deceleration).
    Calculated as: change in price momentum over time.
    Positive = accelerating, negative = decelerating.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # First derivative (momentum)
    momentum = close.pct_change(10)
    
    # Second derivative (curvature)
    curvature = momentum.diff(5)
    
    curvature.name = "price_curvature"
    return curvature


def feature_aroon_up(df: DataFrame) -> Series:
    """
    Aroon Up: Aroon Up in [0, 1].
    
    Measures how recently the highest high occurred within lookback period.
    Calculated as: (period - periods_since_highest) / period * 100, normalized to [0, 1].
    Higher values = recent new highs, lower = old highs.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    
    period = 25
    
    aroon_up = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(df)):
        window_high = high.iloc[i-period:i]
        
        if len(window_high) < period:
            aroon_up.iloc[i] = 0.0
            continue
        
        # Find highest high
        highest_high = window_high.max()
        
        # Find periods since highest high
        periods_since = period - 1
        for j in range(len(window_high) - 1, -1, -1):
            if window_high.iloc[j] == highest_high:
                periods_since = len(window_high) - 1 - j
                break
        
        # Aroon Up = (period - periods_since) / period
        aroon = (period - periods_since) / period
        aroon_up.iloc[i] = aroon
    
    aroon_up = aroon_up.fillna(0.0).clip(0.0, 1.0)
    aroon_up.name = "aroon_up"
    return aroon_up


def feature_aroon_down(df: DataFrame) -> Series:
    """
    Aroon Down: Aroon Down in [0, 1].
    
    Measures how recently the lowest low occurred within lookback period.
    Calculated as: (period - periods_since_lowest) / period * 100, normalized to [0, 1].
    Higher values = recent new lows, lower = old lows.
    Already normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    low = _get_low_series(df)
    
    period = 25
    
    aroon_down = pd.Series(index=df.index, dtype=float)
    
    for i in range(period, len(df)):
        window_low = low.iloc[i-period:i]
        
        if len(window_low) < period:
            aroon_down.iloc[i] = 0.0
            continue
        
        # Find lowest low
        lowest_low = window_low.min()
        
        # Find periods since lowest low
        periods_since = period - 1
        for j in range(len(window_low) - 1, -1, -1):
            if window_low.iloc[j] == lowest_low:
                periods_since = len(window_low) - 1 - j
                break
        
        # Aroon Down = (period - periods_since) / period
        aroon = (period - periods_since) / period
        aroon_down.iloc[i] = aroon
    
    aroon_down = aroon_down.fillna(0.0).clip(0.0, 1.0)
    aroon_down.name = "aroon_down"
    return aroon_down


def feature_aroon_oscillator(df: DataFrame) -> Series:
    """
    Aroon Oscillator: Aroon Oscillator normalized to [0, 1].
    
    Measures difference between Aroon Up and Aroon Down.
    Calculated as: Aroon Up - Aroon Down, normalized to [0, 1].
    Positive = bullish momentum, negative = bearish momentum.
    Normalized to [0, 1]. Clipped for safety.
    """
    aroon_up = feature_aroon_up(df)
    aroon_down = feature_aroon_down(df)
    
    oscillator = aroon_up - aroon_down
    # Normalize from [-1, 1] to [0, 1]
    oscillator_normalized = (oscillator + 1.0) / 2.0
    oscillator_normalized = oscillator_normalized.clip(0.0, 1.0)
    oscillator_normalized.name = "aroon_oscillator"
    return oscillator_normalized


def feature_donchian_breakout(df: DataFrame) -> Series:
    """
    Donchian Breakout: Donchian breakout binary flag.
    
    Indicates if price broke above the highest high of the previous N days.
    Calculated as: (close > high.rolling(N).max().shift(1)).astype(int).
    Binary indicator: 1 if breakout, 0 otherwise.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    
    period = 20
    
    # Calculate highest high of previous period (excluding current day)
    highest_high = high.shift(1).rolling(window=period, min_periods=1).max()
    
    # Check if current close is above previous period's highest high
    breakout = (close > highest_high).astype(float)
    breakout = breakout.fillna(0.0).clip(0.0, 1.0)
    breakout.name = "donchian_breakout"
    return breakout


# ============================================================================
# BLOCK ML-19: Market Microstructure Proxies (8 features)
# ============================================================================


def feature_liquidity_measure(df: DataFrame) -> Series:
    """
    Liquidity Measure: Volume/volatility ratio.
    
    Measures market liquidity as volume relative to price volatility.
    Higher values = more liquid (high volume, low volatility), lower = less liquid.
    Calculated as: volume / (volatility * price).
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    volatility = close.rolling(window=20, min_periods=1).std() / close
    
    liquidity = volume / (volatility * close + 1e-10)
    
    liquidity.name = "liquidity_measure"
    return liquidity


def feature_price_impact_proxy(df: DataFrame) -> Series:
    """
    Price Impact Proxy: Price impact proxy (volume/price change).
    
    Measures how much price moves per unit of volume (inverse of liquidity).
    Higher values = high price impact (illiquid), lower = low impact (liquid).
    Calculated as: abs(price_change) / volume.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    price_change = close.pct_change().abs()
    
    price_impact = price_change / (volume + 1e-10)
    
    price_impact.name = "price_impact_proxy"
    return price_impact


def feature_bid_ask_spread_proxy(df: DataFrame) -> Series:
    """
    Bid-Ask Spread Proxy: Bid-ask spread proxy (high-low/close).
    
    Measures effective bid-ask spread using high-low range relative to price.
    Higher values = wider spread (less liquid), lower = tighter spread (more liquid).
    Calculated as: (high - low) / close.
    Already normalized. No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    spread = (high - low) / close
    
    spread.name = "bid_ask_spread_proxy"
    return spread


def feature_order_flow_imbalance(df: DataFrame) -> Series:
    """
    Order Flow Imbalance: Order flow imbalance proxy.
    
    Measures buying vs selling pressure using price and volume.
    Calculated as: signed volume * price_change, normalized.
    Positive = buying pressure, negative = selling pressure.
    Normalized to [-1, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    price_change = close.pct_change()
    
    # Order flow = volume * price direction
    order_flow = volume * np.sign(price_change)
    
    # Normalize by rolling max
    order_flow_normalized = order_flow / (order_flow.abs().rolling(window=20, min_periods=1).max() + 1e-10)
    order_flow_normalized = order_flow_normalized.clip(-1.0, 1.0)
    order_flow_normalized.name = "order_flow_imbalance"
    return order_flow_normalized


def feature_market_depth_proxy(df: DataFrame) -> Series:
    """
    Market Depth Proxy: Market depth proxy.
    
    Measures market depth using volume relative to price range.
    Higher values = deeper market (more volume per price range), lower = shallow.
    Calculated as: volume / (high - low).
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    price_range = high - low
    
    depth = volume / (price_range + 1e-10)
    
    depth.name = "market_depth_proxy"
    return depth


def feature_price_efficiency(df: DataFrame) -> Series:
    """
    Price Efficiency: Price efficiency measure.
    
    Measures how efficiently price incorporates information.
    Calculated as: inverse of price autocorrelation (lower autocorr = more efficient).
    Higher values = more efficient, lower = less efficient.
    Normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    returns = close.pct_change()
    
    # Calculate autocorrelation
    autocorr = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        window = returns.iloc[i-20:i]
        if len(window) < 10 or window.std() == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        window_lag = returns.iloc[i-21:i-1] if i >= 21 else returns.iloc[i-20:i]
        if len(window_lag) < 10 or window_lag.std() == 0:
            autocorr.iloc[i] = 0.0
            continue
        
        min_len = min(len(window), len(window_lag))
        window_aligned = window.iloc[-min_len:]
        window_lag_aligned = window_lag.iloc[-min_len:]
        
        corr = window_aligned.corr(window_lag_aligned)
        autocorr.iloc[i] = corr if not pd.isna(corr) else 0.0
    
    # Efficiency = 1 - abs(autocorrelation)
    efficiency = 1.0 - autocorr.abs()
    efficiency = efficiency.clip(0.0, 1.0)
    efficiency.name = "price_efficiency"
    return efficiency


def feature_transaction_cost_proxy(df: DataFrame) -> Series:
    """
    Transaction Cost Proxy: Transaction cost proxy.
    
    Measures estimated transaction costs using bid-ask spread and price impact.
    Higher values = higher costs, lower = lower costs.
    Calculated as: (spread + price_impact) / price.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    # Spread component
    spread = (high - low) / close
    
    # Price impact component
    price_change = close.pct_change().abs()
    price_impact = price_change / (volume + 1e-10)
    
    # Combine
    transaction_cost = spread + price_impact
    
    transaction_cost.name = "transaction_cost_proxy"
    return transaction_cost


def feature_market_quality(df: DataFrame) -> Series:
    """
    Market Quality: Overall market quality measure.
    
    Combines multiple microstructure signals into overall quality score.
    Factors: liquidity, efficiency, depth, transaction costs.
    Higher values = better market quality, lower = worse.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get component features
    liquidity = feature_liquidity_measure(df)
    efficiency = feature_price_efficiency(df)
    depth = feature_market_depth_proxy(df)
    transaction_cost = feature_transaction_cost_proxy(df)
    
    # Normalize components
    liquidity_norm = liquidity / (liquidity.rolling(window=252, min_periods=1).max() + 1e-10)
    depth_norm = depth / (depth.rolling(window=252, min_periods=1).max() + 1e-10)
    cost_norm = 1.0 - (transaction_cost / (transaction_cost.rolling(window=252, min_periods=1).max() + 1e-10))
    
    # Combine (equal weights)
    quality = (liquidity_norm.clip(0, 1) * 0.3 + 
               efficiency * 0.3 + 
               depth_norm.clip(0, 1) * 0.2 + 
               cost_norm.clip(0, 1) * 0.2)
    
    quality = quality.clip(0.0, 1.0)
    quality.name = "market_quality"
    return quality


# ============================================================================
# BLOCK ML-20: Feature Interactions - Core (20 features)
# ============================================================================


def feature_rsi_x_volume(df: DataFrame) -> Series:
    """
    RSI × Volume: RSI × Volume ratio interaction.
    
    Combines momentum (RSI) with volume confirmation.
    Higher values = strong momentum with volume support.
    Calculated as: RSI14 × relative_volume.
    No clipping - let ML learn the distribution.
    """
    rsi = feature_rsi14(df)
    volume_ratio = feature_relative_volume(df)
    
    interaction = rsi * volume_ratio
    
    interaction.name = "rsi_x_volume"
    return interaction


def feature_rsi_x_atr(df: DataFrame) -> Series:
    """
    RSI × ATR: RSI × ATR pct interaction.
    
    Combines momentum (RSI) with volatility (ATR).
    Higher values = strong momentum in volatile conditions.
    Calculated as: RSI14 × ATR14_normalized.
    No clipping - let ML learn the distribution.
    """
    rsi = feature_rsi14(df)
    atr = feature_atr14_normalized(df)
    
    interaction = rsi * atr
    
    interaction.name = "rsi_x_atr"
    return interaction


def feature_macd_x_volume(df: DataFrame) -> Series:
    """
    MACD × Volume: MACD × Volume ratio interaction.
    
    Combines trend momentum (MACD) with volume confirmation.
    Higher values = strong trend with volume support.
    Calculated as: MACD_histogram × relative_volume.
    No clipping - let ML learn the distribution.
    """
    macd = feature_macd_histogram_normalized(df)
    volume_ratio = feature_relative_volume(df)
    
    interaction = macd * volume_ratio
    
    interaction.name = "macd_x_volume"
    return interaction


def feature_trend_x_momentum(df: DataFrame) -> Series:
    """
    Trend × Momentum: Trend strength × Momentum interaction.
    
    Combines trend strength with momentum.
    Higher values = strong trend with strong momentum.
    Calculated as: trend_strength_20d × momentum_20d.
    No clipping - let ML learn the distribution.
    """
    trend = feature_trend_strength_20d(df)
    momentum = feature_momentum_20d(df)
    
    interaction = trend * momentum
    
    interaction.name = "trend_x_momentum"
    return interaction


def feature_volume_x_momentum(df: DataFrame) -> Series:
    """
    Volume × Momentum: Volume × Momentum interaction.
    
    Combines volume with momentum.
    Higher values = strong momentum with volume support.
    Calculated as: relative_volume × momentum_20d.
    No clipping - let ML learn the distribution.
    """
    volume_ratio = feature_relative_volume(df)
    momentum = feature_momentum_20d(df)
    
    interaction = volume_ratio * momentum
    
    interaction.name = "volume_x_momentum"
    return interaction


def feature_rsi_x_trend_strength(df: DataFrame) -> Series:
    """
    RSI × Trend Strength: RSI × Trend strength interaction.
    
    Combines momentum (RSI) with trend strength.
    Higher values = strong momentum in strong trend.
    Calculated as: RSI14 × trend_strength_20d.
    No clipping - let ML learn the distribution.
    """
    rsi = feature_rsi14(df)
    trend = feature_trend_strength_20d(df)
    
    interaction = rsi * trend
    
    interaction.name = "rsi_x_trend_strength"
    return interaction


def feature_volatility_x_volume(df: DataFrame) -> Series:
    """
    Volatility × Volume: Volatility × Volume interaction.
    
    Combines volatility with volume.
    Higher values = high volatility with high volume.
    Calculated as: volatility_21d × relative_volume.
    No clipping - let ML learn the distribution.
    """
    volatility = feature_volatility_21d(df)
    volume_ratio = feature_relative_volume(df)
    
    interaction = volatility * volume_ratio
    
    interaction.name = "volatility_x_volume"
    return interaction


def feature_volatility_x_trend(df: DataFrame) -> Series:
    """
    Volatility × Trend: Volatility × Trend strength interaction.
    
    Combines volatility with trend strength.
    Higher values = high volatility in strong trend.
    Calculated as: volatility_21d × trend_strength_20d.
    No clipping - let ML learn the distribution.
    """
    volatility = feature_volatility_21d(df)
    trend = feature_trend_strength_20d(df)
    
    interaction = volatility * trend
    
    interaction.name = "volatility_x_trend"
    return interaction


def feature_trend_x_volatility(df: DataFrame) -> Series:
    """
    Trend × Volatility: Trend strength × Volatility interaction.
    
    Same as volatility_x_trend but emphasizes trend perspective.
    Calculated as: trend_strength_20d × volatility_21d.
    No clipping - let ML learn the distribution.
    """
    trend = feature_trend_strength_20d(df)
    volatility = feature_volatility_21d(df)
    
    interaction = trend * volatility
    
    interaction.name = "trend_x_volatility"
    return interaction


def feature_volatility_x_regime_strength(df: DataFrame) -> Series:
    """
    Volatility × Regime Strength: Volatility × Regime strength interaction.
    
    Combines volatility with market regime strength.
    Higher values = high volatility in strong regime.
    Calculated as: volatility_21d × regime_strength.
    No clipping - let ML learn the distribution.
    """
    volatility = feature_volatility_21d(df)
    regime_strength = feature_regime_strength(df)
    
    interaction = volatility * regime_strength
    
    interaction.name = "volatility_x_regime_strength"
    return interaction


def feature_price_position_x_rsi(df: DataFrame) -> Series:
    """
    Price Position × RSI: Price percentile × RSI interaction.
    
    Combines price position with momentum.
    Higher values = strong momentum at high price position.
    Calculated as: close_position_in_range × RSI14.
    No clipping - let ML learn the distribution.
    """
    price_pos = feature_close_position_in_range(df)
    rsi = feature_rsi14(df)
    
    interaction = price_pos * rsi
    
    interaction.name = "price_position_x_rsi"
    return interaction


def feature_support_resistance_x_momentum(df: DataFrame) -> Series:
    """
    Support/Resistance × Momentum: Support/resistance distance × Momentum interaction.
    
    Combines support/resistance position with momentum.
    Higher values = strong momentum near support/resistance.
    Calculated as: (distance_to_support + distance_to_resistance) × momentum_20d.
    No clipping - let ML learn the distribution.
    """
    support_dist = feature_distance_to_support(df)
    resistance_dist = feature_distance_to_resistance(df)
    momentum = feature_momentum_20d(df)
    
    # Combine support and resistance distances
    sr_distance = (support_dist + resistance_dist) / 2.0
    
    interaction = sr_distance * momentum
    
    interaction.name = "support_resistance_x_momentum"
    return interaction


def feature_rsi_x_support_distance(df: DataFrame) -> Series:
    """
    RSI × Support Distance: RSI × Distance to support interaction.
    
    Combines momentum with proximity to support.
    Higher values = strong momentum near support (potential bounce).
    Calculated as: RSI14 × distance_to_support.
    No clipping - let ML learn the distribution.
    """
    rsi = feature_rsi14(df)
    support_dist = feature_distance_to_support(df)
    
    interaction = rsi * support_dist
    
    interaction.name = "rsi_x_support_distance"
    return interaction


def feature_volume_x_breakout(df: DataFrame) -> Series:
    """
    Volume × Breakout: Volume × Breakout signal interaction.
    
    Combines volume with breakout signals.
    Higher values = high volume on breakout (strong signal).
    Calculated as: relative_volume × donchian_breakout.
    No clipping - let ML learn the distribution.
    """
    volume_ratio = feature_relative_volume(df)
    breakout = feature_donchian_breakout(df)
    
    interaction = volume_ratio * breakout
    
    interaction.name = "volume_x_breakout"
    return interaction


def feature_momentum_x_regime(df: DataFrame) -> Series:
    """
    Momentum × Regime: Momentum × Market regime interaction.
    
    Combines momentum with market regime.
    Higher values = strong momentum in bullish regime.
    Calculated as: momentum_20d × market_regime (normalized).
    No clipping - let ML learn the distribution.
    """
    momentum = feature_momentum_20d(df)
    regime = feature_market_regime(df)
    
    # Normalize regime to [-1, 1] range (0=bull, 1=bear, 2=sideways -> -1, 1, 0)
    regime_normalized = (regime - 1.0) / 1.0  # -1 to +1
    
    interaction = momentum * regime_normalized
    
    interaction.name = "momentum_x_regime"
    return interaction


def feature_volume_x_regime(df: DataFrame) -> Series:
    """
    Volume × Regime: Volume × Market regime interaction.
    
    Combines volume with market regime.
    Higher values = high volume in bullish regime.
    Calculated as: relative_volume × market_regime (normalized).
    No clipping - let ML learn the distribution.
    """
    volume_ratio = feature_relative_volume(df)
    regime = feature_market_regime(df)
    
    # Normalize regime to [-1, 1] range
    regime_normalized = (regime - 1.0) / 1.0
    
    interaction = volume_ratio * regime_normalized
    
    interaction.name = "volume_x_regime"
    return interaction


def feature_rsi_x_relative_strength(df: DataFrame) -> Series:
    """
    RSI × Relative Strength: RSI × Relative strength interaction.
    
    Combines momentum with relative strength vs market.
    Higher values = strong momentum with outperformance.
    Calculated as: RSI14 × relative_strength_spy (normalized).
    No clipping - let ML learn the distribution.
    """
    rsi = feature_rsi14(df)
    rs = feature_relative_strength_spy(df)
    
    # Normalize RS to [0, 1] range
    rs_normalized = (rs - rs.rolling(window=252, min_periods=1).min()) / (
        rs.rolling(window=252, min_periods=1).max() - rs.rolling(window=252, min_periods=1).min() + 1e-10
    )
    
    interaction = rsi * rs_normalized
    
    interaction.name = "rsi_x_relative_strength"
    return interaction


def feature_trend_x_relative_strength(df: DataFrame) -> Series:
    """
    Trend × Relative Strength: Trend × Relative strength interaction.
    
    Combines trend strength with relative strength vs market.
    Higher values = strong trend with outperformance.
    Calculated as: trend_strength_20d × relative_strength_spy (normalized).
    No clipping - let ML learn the distribution.
    """
    trend = feature_trend_strength_20d(df)
    rs = feature_relative_strength_spy(df)
    
    # Normalize RS to [0, 1] range
    rs_normalized = (rs - rs.rolling(window=252, min_periods=1).min()) / (
        rs.rolling(window=252, min_periods=1).max() - rs.rolling(window=252, min_periods=1).min() + 1e-10
    )
    
    interaction = trend * rs_normalized
    
    interaction.name = "trend_x_relative_strength"
    return interaction


def feature_momentum_x_time(df: DataFrame) -> Series:
    """
    Momentum × Time: Momentum × Time feature interaction.
    
    Combines momentum with time-based features (day of week).
    Captures time-of-week effects on momentum.
    Calculated as: momentum_20d × day_of_week_sin.
    No clipping - let ML learn the distribution.
    """
    momentum = feature_momentum_20d(df)
    time_feature = feature_day_of_week_sin(df)
    
    interaction = momentum * time_feature
    
    interaction.name = "momentum_x_time"
    return interaction


def feature_volume_x_time(df: DataFrame) -> Series:
    """
    Volume × Time: Volume × Time feature interaction.
    
    Combines volume with time-based features (day of week).
    Captures time-of-week effects on volume.
    Calculated as: relative_volume × day_of_week_sin.
    No clipping - let ML learn the distribution.
    """
    volume_ratio = feature_relative_volume(df)
    time_feature = feature_day_of_week_sin(df)
    
    interaction = volume_ratio * time_feature
    
    interaction.name = "volume_x_time"
    return interaction


# ============================================================================
# BLOCK ML-21: Predictive Gain Features (15 features)
# ============================================================================
# Note: These features predict probability of achieving target gain (e.g., 15% in 20 days)
# Target gain and horizon are configurable but default to typical swing trading values


def feature_historical_gain_probability(df: DataFrame, target_gain: float = 0.15, horizon: int = 20, cached_result: Optional[Series] = None) -> Series:
    """
    Historical Gain Probability: Historical probability of X% gain in Y days.
    
    Estimates probability of achieving target gain based on historical patterns.
    Calculated as: rolling percentage of past periods where target gain was achieved.
    Uses rolling window of 252 days to calculate historical success rate.
    NOTE: Uses historical lookback only - no future data (lookahead-safe).
    Already normalized to [0, 1]. Clipped for safety.
    """
    # Return cached result if provided (optimization: Strategy 1)
    if cached_result is not None:
        return cached_result.copy()
    
    close = _get_close_series(df)
    high = _get_high_series(df)
    
    # Optimized: Vectorized version using rolling windows
    # Pre-compute rolling max of high over horizon days (much faster than computing in loop)
    # For each position j, we want: max(high[j-horizon:j])
    rolling_max_high = high.rolling(window=horizon, min_periods=1).max()
    
    # Shift close by horizon days to get the starting price for each window
    # For position j, we want close[j-horizon] as the starting price
    close_shifted = close.shift(horizon)
    
    # Calculate historical returns: (max_high / start_close) - 1.0
    # This gives us the return achieved in each horizon-day window
    historical_returns = (rolling_max_high / close_shifted) - 1.0
    
    # Check if target gain was achieved (vectorized boolean array)
    achievements = (historical_returns >= target_gain).astype(float)
    
    # Calculate rolling probability: rolling mean of achievements over 252 days
    # This replaces the nested loop that was computing mean of historical_achievements
    # For position i, original code computes mean of achievements[j] for j in [i-252, i-horizon)
    # 
    # Optimized approach: Use rolling sum and count, then compute mean
    # We'll use numpy arrays for faster computation
    achievements_values = achievements.values
    probability_values = np.zeros(len(df), dtype=float)
    
    # For positions where we have enough history
    for i in range(horizon + 252, len(df)):
        # Get slice [i-252, i-horizon) - much faster with numpy array slicing
        window_achievements = achievements_values[i-252:i-horizon]
        if len(window_achievements) > 0:
            probability_values[i] = np.mean(window_achievements)
        else:
            probability_values[i] = 0.0
    
    probability = pd.Series(probability_values, index=df.index)
    
    # Set early values to 0.0 (before we have enough history) - already zeros from initialization
    
    probability = probability.fillna(0.0).clip(0.0, 1.0)
    probability.name = "historical_gain_probability"
    return probability


def feature_gain_probability_score(df: DataFrame, target_gain: float = 0.15, horizon: int = 20, cached_historical_prob: Optional[Series] = None, cached_result: Optional[Series] = None) -> Series:
    """
    Gain Probability Score: Combined score for gain probability.
    
    Combines multiple signals to estimate gain probability.
    Factors: historical probability, momentum, trend, volume.
    Higher values = higher probability of achieving target gain.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Return cached result if provided (optimization: Strategy 1)
    if cached_result is not None:
        return cached_result.copy()
    
    # Get component features
    if cached_historical_prob is not None:
        historical_prob = cached_historical_prob
    else:
        historical_prob = feature_historical_gain_probability(df, target_gain, horizon)
    momentum = feature_momentum_20d(df)
    trend = feature_trend_strength_20d(df)
    volume_ratio = feature_relative_volume(df)
    
    # Normalize components
    momentum_norm = (momentum + 0.1) / 0.2  # Normalize to [0, 1]
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    volume_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_norm = volume_norm.clip(0.0, 1.0)
    
    # Combine (weighted)
    score = (historical_prob * 0.4 + 
             momentum_norm * 0.3 + 
             trend * 0.2 + 
             volume_norm * 0.1)
    
    score = score.clip(0.0, 1.0)
    score.name = "gain_probability_score"
    return score


def feature_gain_regime(df: DataFrame, target_gain: float = 0.15, horizon: int = 20, cached_gain_prob_score: Optional[Series] = None) -> Series:
    """
    Gain Regime: Gain regime (high/low probability), normalized.
    
    Classifies current regime for achieving target gain.
    - 0.0 = Low probability regime
    - 0.5 = Medium probability regime
    - 1.0 = High probability regime
    Normalized to [0, 1]. Clipped for safety.
    """
    if cached_gain_prob_score is not None:
        probability_score = cached_gain_prob_score
    else:
        probability_score = feature_gain_probability_score(df, target_gain, horizon)
    
    # Classify regime based on percentile
    # Optimized: Use vectorized rank instead of .apply() with lambda
    regime = probability_score.rolling(window=252, min_periods=50).rank(pct=True, method='average')
    regime = regime.fillna(0.5).clip(0.0, 1.0)
    regime.name = "gain_regime"
    return regime


def feature_gain_consistency(df: DataFrame, target_gain: float = 0.15, horizon: int = 20, cached_historical_prob: Optional[Series] = None) -> Series:
    """
    Gain Consistency: Consistency of gain achievement.
    
    Measures how consistently target gain has been achieved historically.
    Calculated as: inverse of variance in historical gain achievement.
    Higher values = more consistent, lower = more variable.
    NOTE: Uses historical lookback only - no future data (lookahead-safe).
    Normalized to [0, 1]. Clipped for safety.
    """
    # Use historical probability to calculate consistency
    if cached_historical_prob is not None:
        historical_prob = cached_historical_prob
    else:
        historical_prob = feature_historical_gain_probability(df, target_gain, horizon)
    
    # Calculate consistency as inverse of variance in probability
    variance = historical_prob.rolling(window=252, min_periods=50).var()
    consistency = 1.0 / (variance + 1e-10)
    consistency = consistency / (consistency.rolling(window=252, min_periods=1).max() + 1e-10)
    consistency = consistency.clip(0.0, 1.0)
    consistency.name = "gain_consistency"
    return consistency


def feature_gain_timing(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Timing: Optimal timing for gain.
    
    Measures if current conditions are optimal for achieving target gain.
    Combines: momentum, trend, volume, volatility, support levels.
    Higher values = better timing, lower = worse timing.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get component features
    momentum = feature_momentum_20d(df)
    trend = feature_trend_strength_20d(df)
    volume_ratio = feature_relative_volume(df)
    volatility = feature_volatility_21d(df)
    support_dist = feature_distance_to_support(df)
    
    # Normalize components
    momentum_norm = (momentum + 0.1) / 0.2
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    volume_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_norm = volume_norm.clip(0.0, 1.0)
    volatility_norm = 1.0 - (volatility / (volatility.rolling(window=252, min_periods=1).max() + 1e-10))
    volatility_norm = volatility_norm.clip(0.0, 1.0)
    support_norm = (support_dist + 0.1) / 0.2  # Normalize support distance
    support_norm = support_norm.clip(0.0, 1.0)
    
    # Combine (weighted)
    timing = (momentum_norm * 0.3 + 
              trend * 0.25 + 
              volume_norm * 0.2 + 
              volatility_norm * 0.15 + 
              support_norm * 0.1)
    
    timing = timing.clip(0.0, 1.0)
    timing.name = "gain_timing"
    return timing


def feature_gain_momentum(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Momentum: Momentum toward target gain.
    
    Measures current momentum relative to target gain requirement.
    Calculated as: current momentum / required momentum for target.
    Higher values = strong momentum toward target, lower = weak.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    momentum = feature_momentum_20d(df)
    
    # Required momentum per day to achieve target gain
    required_momentum_per_day = target_gain / horizon
    
    # Current momentum (20-day)
    current_momentum_per_day = momentum / 20.0
    
    # Gain momentum = current / required
    gain_momentum = current_momentum_per_day / (required_momentum_per_day + 1e-10)
    
    gain_momentum.name = "gain_momentum"
    return gain_momentum


def feature_gain_acceleration(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Acceleration: Acceleration toward target gain.
    
    Measures if momentum is accelerating toward target gain.
    Calculated as: change in gain momentum over time.
    Positive = accelerating, negative = decelerating.
    No clipping - let ML learn the distribution.
    """
    gain_momentum = feature_gain_momentum(df, target_gain, horizon)
    
    acceleration = gain_momentum.diff(5)
    
    acceleration.name = "gain_acceleration"
    return acceleration


def feature_target_distance(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Target Distance: Distance to target gain %.
    
    Measures how far current price is from target gain level.
    Calculated as: (target_price - current_price) / current_price.
    Negative = already above target, positive = below target.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Target price
    target_price = close * (1.0 + target_gain)
    
    # Distance to target
    distance = (target_price - close) / close
    
    distance.name = "target_distance"
    return distance


def feature_gain_momentum_strength(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Momentum Strength: Strength of gain momentum.
    
    Measures strength of momentum toward target gain.
    Combines: gain momentum, volume confirmation, trend alignment.
    Higher values = stronger momentum, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    gain_momentum = feature_gain_momentum(df, target_gain, horizon)
    volume_ratio = feature_relative_volume(df)
    trend = feature_trend_strength_20d(df)
    
    # Normalize components
    momentum_norm = gain_momentum / (gain_momentum.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    volume_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_norm = volume_norm.clip(0.0, 1.0)
    
    # Combine
    strength = (momentum_norm * 0.5 + trend * 0.3 + volume_norm * 0.2)
    strength = strength.clip(0.0, 1.0)
    strength.name = "gain_momentum_strength"
    return strength


def feature_gain_breakout_signal(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Breakout Signal: Breakout signal for gain.
    
    Measures if price is breaking out toward target gain.
    Combines: breakout patterns, momentum, volume.
    Higher values = stronger breakout signal, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    breakout = feature_donchian_breakout(df)
    volume_ratio = feature_relative_volume(df)
    momentum = feature_momentum_20d(df)
    
    # Normalize components
    momentum_norm = (momentum + 0.1) / 0.2
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    volume_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_norm = volume_norm.clip(0.0, 1.0)
    
    # Combine
    signal = (breakout * 0.4 + momentum_norm * 0.35 + volume_norm * 0.25)
    signal = signal.clip(0.0, 1.0)
    signal.name = "gain_breakout_signal"
    return signal


def feature_gain_risk_ratio(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Risk Ratio: Gain potential vs risk ratio.
    
    Measures reward-to-risk ratio for target gain.
    Calculated as: target_gain / stop_loss_distance.
    Higher values = better risk-reward, lower = worse.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    low = _get_low_series(df)
    
    # Target gain
    target_price = close * (1.0 + target_gain)
    
    # Stop loss distance (using swing low as proxy)
    swing_low = feature_swing_low_10d(df)
    stop_distance = (close - swing_low) / close
    
    # Risk-reward ratio
    risk_reward = target_gain / (stop_distance + 1e-10)
    
    risk_reward.name = "gain_risk_ratio"
    return risk_reward


def feature_gain_volume_confirmation(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Volume Confirmation: Volume confirmation for gain.
    
    Measures if volume supports achieving target gain.
    Combines: volume ratio, volume trend, volume momentum.
    Higher values = strong volume confirmation, lower = weak.
    Normalized to [0, 1]. Clipped for safety.
    """
    volume_ratio = feature_relative_volume(df)
    volume_trend = feature_volume_trend(df)
    volume_momentum = feature_volume_momentum(df)
    
    # Normalize components
    volume_ratio_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_ratio_norm = volume_ratio_norm.clip(0.0, 1.0)
    volume_trend_norm = (volume_trend + 1.0) / 2.0
    volume_trend_norm = volume_trend_norm.clip(0.0, 1.0)
    volume_momentum_norm = (volume_momentum + 1.0) / 2.0
    volume_momentum_norm = volume_momentum_norm.clip(0.0, 1.0)
    
    # Combine
    confirmation = (volume_ratio_norm * 0.5 + volume_trend_norm * 0.3 + volume_momentum_norm * 0.2)
    confirmation = confirmation.clip(0.0, 1.0)
    confirmation.name = "gain_volume_confirmation"
    return confirmation


def feature_gain_trend_alignment(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Trend Alignment: Trend alignment for gain.
    
    Measures if trend is aligned with achieving target gain.
    Combines: trend strength, trend direction, trend consistency.
    Higher values = strong trend alignment, lower = weak.
    Normalized to [0, 1]. Clipped for safety.
    """
    trend_strength = feature_trend_strength_20d(df)
    trend_consistency = feature_trend_consistency(df)
    momentum = feature_momentum_20d(df)
    
    # Check if trend is bullish (momentum positive)
    trend_direction = (momentum > 0).astype(float)
    
    # Combine
    alignment = (trend_strength * 0.4 + trend_consistency * 0.3 + trend_direction * 0.3)
    alignment = alignment.clip(0.0, 1.0)
    alignment.name = "gain_trend_alignment"
    return alignment


def feature_gain_volatility_context(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Volatility Context: Volatility context for gain.
    
    Measures if volatility context supports achieving target gain.
    Moderate volatility is best (too low = no movement, too high = risk).
    Higher values = favorable volatility context, lower = unfavorable.
    Normalized to [0, 1]. Clipped for safety.
    """
    volatility = feature_volatility_21d(df)
    
    # Optimal volatility range (moderate)
    # Optimized: Use vectorized rank instead of .apply() with lambda
    vol_percentile = volatility.rolling(window=252, min_periods=1).rank(pct=True, method='average')
    vol_percentile = vol_percentile.fillna(0.5)
    
    # Best volatility is in middle range (0.3 to 0.7 percentile)
    # Create bell curve around 0.5
    vol_context = 1.0 - 4.0 * (vol_percentile - 0.5) ** 2
    vol_context = vol_context.clip(0.0, 1.0)
    
    vol_context.name = "gain_volatility_context"
    return vol_context


def feature_gain_support_level(df: DataFrame, target_gain: float = 0.15, horizon: int = 20) -> Series:
    """
    Gain Support Level: Support level for gain.
    
    Measures if support levels are strong enough for target gain.
    Combines: support distance, support strength, support touches.
    Higher values = strong support, lower = weak support.
    Normalized to [0, 1]. Clipped for safety.
    """
    support_dist = feature_distance_to_support(df)
    support_strength = feature_support_resistance_strength(df)
    support_touches = feature_support_touches(df)
    
    # Normalize components
    support_dist_norm = (support_dist + 0.1) / 0.2
    support_dist_norm = support_dist_norm.clip(0.0, 1.0)
    support_touches_norm = support_touches / (support_touches.rolling(window=252, min_periods=1).max() + 1e-10)
    support_touches_norm = support_touches_norm.clip(0.0, 1.0)
    
    # Combine
    support_level = (support_dist_norm * 0.3 + support_strength * 0.4 + support_touches_norm * 0.3)
    support_level = support_level.clip(0.0, 1.0)
    support_level.name = "gain_support_level"
    return support_level


# ============================================================================
# BLOCK ML-22: Accuracy-Enhancing Features (20 features)
# ============================================================================
# Note: These features combine multiple signals to assess overall signal quality,
# strength, consistency, and context to improve prediction accuracy


def feature_signal_quality_score(df: DataFrame, 
                                  cached_signal_consistency: Optional[Series] = None,
                                  cached_signal_confirmation: Optional[Series] = None,
                                  cached_signal_strength: Optional[Series] = None,
                                  cached_signal_timing: Optional[Series] = None,
                                  cached_signal_risk_reward: Optional[Series] = None,
                                  cached_result: Optional[Series] = None) -> Series:
    """
    Signal Quality Score: Overall signal quality score.
    
    Combines multiple quality metrics into overall signal quality.
    Factors: consistency, confirmation, strength, timing, risk-reward.
    Higher values = higher quality signal, lower = lower quality.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Return cached result if provided (optimization)
    if cached_result is not None:
        return cached_result.copy()
    
    # Get component features (use cached if available)
    if cached_signal_consistency is not None:
        consistency = cached_signal_consistency
    else:
        consistency = feature_signal_consistency(df)
    
    if cached_signal_confirmation is not None:
        confirmation = cached_signal_confirmation
    else:
        confirmation = feature_signal_confirmation(df)
    
    if cached_signal_strength is not None:
        strength = cached_signal_strength
    else:
        strength = feature_signal_strength(df)
    
    if cached_signal_timing is not None:
        timing = cached_signal_timing
    else:
        timing = feature_signal_timing(df)
    
    if cached_signal_risk_reward is not None:
        risk_reward = cached_signal_risk_reward
    else:
        risk_reward = feature_signal_risk_reward(df)
    
    # Normalize risk-reward to [0, 1]
    risk_reward_norm = risk_reward / (risk_reward.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    risk_reward_norm = (risk_reward_norm + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
    risk_reward_norm = risk_reward_norm.clip(0.0, 1.0)
    
    # Combine (weighted)
    quality = (consistency * 0.25 + 
               confirmation * 0.25 + 
               strength * 0.2 + 
               timing * 0.15 + 
               risk_reward_norm * 0.15)
    
    quality = quality.clip(0.0, 1.0)
    quality.name = "signal_quality_score"
    return quality


def feature_signal_strength(df: DataFrame) -> Series:
    """
    Signal Strength: Overall signal strength.
    
    Combines multiple strength indicators into overall signal strength.
    Factors: momentum strength, trend strength, volume strength, pattern strength.
    Higher values = stronger signal, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get component features
    momentum = feature_momentum_20d(df)
    trend = feature_trend_strength_20d(df)
    volume_ratio = feature_relative_volume(df)
    pattern = feature_pattern_strength(df)
    
    # Normalize components
    momentum_norm = (momentum + 0.1) / 0.2
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    volume_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_norm = volume_norm.clip(0.0, 1.0)
    pattern_norm = (pattern + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
    pattern_norm = pattern_norm.clip(0.0, 1.0)
    
    # Combine (weighted)
    strength = (momentum_norm * 0.3 + 
                trend * 0.3 + 
                volume_norm * 0.2 + 
                pattern_norm * 0.2)
    
    strength = strength.clip(0.0, 1.0)
    strength.name = "signal_strength"
    return strength


def feature_signal_consistency(df: DataFrame) -> Series:
    """
    Signal Consistency: Signal consistency across indicators.
    
    Measures how consistently different indicators agree on direction.
    Calculated as: inverse of variance in normalized indicator signals.
    Higher values = more consistent, lower = more divergent.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get multiple indicator signals
    rsi = feature_rsi14(df)
    macd = feature_macd_signal(df)
    momentum = feature_momentum_20d(df)
    trend = feature_trend_strength_20d(df)
    
    # Normalize to [0, 1]
    rsi_norm = rsi / 100.0
    macd_norm = (macd + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
    macd_norm = macd_norm.clip(0.0, 1.0)
    momentum_norm = (momentum + 0.1) / 0.2
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    
    # Calculate variance across indicators
    signals = pd.DataFrame({
        'rsi': rsi_norm,
        'macd': macd_norm,
        'momentum': momentum_norm,
        'trend': trend
    })
    
    # Variance across columns (axis=1)
    variance = signals.var(axis=1)
    
    # Consistency = inverse of variance
    consistency = 1.0 / (variance + 1e-10)
    consistency = consistency / (consistency.rolling(window=252, min_periods=1).max() + 1e-10)
    consistency = consistency.clip(0.0, 1.0)
    consistency.name = "signal_consistency"
    return consistency


def feature_signal_confirmation(df: DataFrame) -> Series:
    """
    Signal Confirmation: Signal confirmation strength.
    
    Measures how many indicators confirm the signal direction.
    Calculated as: percentage of indicators agreeing on direction.
    Higher values = stronger confirmation, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get multiple indicator signals
    rsi = feature_rsi14(df)
    macd = feature_macd_signal(df)
    momentum = feature_momentum_20d(df)
    volume_ratio = feature_relative_volume(df)
    breakout = feature_donchian_breakout(df)
    
    # Determine direction for each indicator
    rsi_bullish = (rsi > 50).astype(float)
    macd_bullish = (macd > 0).astype(float)
    momentum_bullish = (momentum > 0).astype(float)
    volume_high = (volume_ratio > 1.0).astype(float)
    breakout_bullish = (breakout > 0.5).astype(float)
    
    # Count bullish signals
    bullish_count = rsi_bullish + macd_bullish + momentum_bullish + volume_high + breakout_bullish
    
    # Confirmation = percentage of indicators agreeing
    confirmation = bullish_count / 5.0
    
    confirmation = confirmation.clip(0.0, 1.0)
    confirmation.name = "signal_confirmation"
    return confirmation


def feature_signal_timing(df: DataFrame) -> Series:
    """
    Signal Timing: Optimal signal timing.
    
    Measures if current conditions are optimal for signal entry.
    Combines: momentum, trend, volume, volatility, support levels.
    Higher values = better timing, lower = worse.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get component features
    momentum = feature_momentum_20d(df)
    trend = feature_trend_strength_20d(df)
    volume_ratio = feature_relative_volume(df)
    volatility = feature_volatility_21d(df)
    support_dist = feature_distance_to_support(df)
    
    # Normalize components
    momentum_norm = (momentum + 0.1) / 0.2
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    volume_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_norm = volume_norm.clip(0.0, 1.0)
    volatility_norm = 1.0 - (volatility / (volatility.rolling(window=252, min_periods=1).max() + 1e-10))
    volatility_norm = volatility_norm.clip(0.0, 1.0)
    support_norm = (support_dist + 0.1) / 0.2
    support_norm = support_norm.clip(0.0, 1.0)
    
    # Combine (weighted)
    timing = (momentum_norm * 0.3 + 
              trend * 0.25 + 
              volume_norm * 0.2 + 
              volatility_norm * 0.15 + 
              support_norm * 0.1)
    
    timing = timing.clip(0.0, 1.0)
    timing.name = "signal_timing"
    return timing


def feature_signal_risk_reward(df: DataFrame) -> Series:
    """
    Signal Risk-Reward: Signal risk-reward ratio.
    
    Measures reward-to-risk ratio for the signal.
    Calculated as: potential gain / stop loss distance.
    Higher values = better risk-reward, lower = worse.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Potential gain (using momentum as proxy)
    momentum = feature_momentum_20d(df)
    potential_gain = abs(momentum)
    
    # Stop loss distance (using swing low as proxy)
    swing_low = feature_swing_low_10d(df)
    stop_distance = (close - swing_low) / close
    
    # Risk-reward ratio
    risk_reward = potential_gain / (stop_distance + 1e-10)
    
    risk_reward.name = "signal_risk_reward"
    return risk_reward


def feature_signal_statistical_significance(df: DataFrame) -> Series:
    """
    Signal Statistical Significance: Statistical significance of signal.
    
    Measures if signal strength is statistically significant.
    Calculated as: z-score of signal strength relative to historical distribution.
    Higher values = more significant, lower = less significant.
    Normalized to [0, 1]. Clipped for safety.
    """
    signal_strength = feature_signal_strength(df)
    
    # Calculate z-score
    mean = signal_strength.rolling(window=252, min_periods=50).mean()
    std = signal_strength.rolling(window=252, min_periods=50).std()
    z_score = (signal_strength - mean) / (std + 1e-10)
    
    # Convert z-score to [0, 1] using sigmoid
    significance = 1.0 / (1.0 + np.exp(-z_score))
    significance = significance.clip(0.0, 1.0)
    significance.name = "signal_statistical_significance"
    return significance


def feature_signal_historical_success(df: DataFrame) -> Series:
    """
    Signal Historical Success: Historical success rate of similar signals.
    
    Estimates success rate based on historical patterns matching current conditions.
    Uses rolling window to calculate historical success rate.
    Higher values = higher historical success, lower = lower.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Use signal strength as proxy for signal quality
    signal_strength = feature_signal_strength(df)
    
    # Calculate historical success rate (simplified: use percentile rank)
    # In practice, this would match current conditions to historical periods
    # Optimized: Use vectorized rank instead of .apply() with lambda
    success_rate = signal_strength.rolling(window=252, min_periods=50).rank(pct=True, method='average')
    success_rate = success_rate.fillna(0.5).clip(0.0, 1.0)
    success_rate.name = "signal_historical_success"
    return success_rate


def feature_signal_ensemble_score(df: DataFrame,
                                   cached_signal_quality_score: Optional[Series] = None,
                                   cached_signal_strength: Optional[Series] = None,
                                   cached_signal_confirmation: Optional[Series] = None,
                                   cached_signal_timing: Optional[Series] = None,
                                   cached_signal_historical_success: Optional[Series] = None,
                                   cached_result: Optional[Series] = None) -> Series:
    """
    Signal Ensemble Score: Ensemble score from multiple models.
    
    Combines multiple signal sources into ensemble score.
    Uses: quality score, strength, confirmation, timing, historical success.
    Higher values = stronger ensemble signal, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Return cached result if provided (optimization)
    if cached_result is not None:
        return cached_result.copy()
    
    # Get component features (use cached if available)
    if cached_signal_quality_score is not None:
        quality = cached_signal_quality_score
    else:
        quality = feature_signal_quality_score(df)
    
    if cached_signal_strength is not None:
        strength = cached_signal_strength
    else:
        strength = feature_signal_strength(df)
    
    if cached_signal_confirmation is not None:
        confirmation = cached_signal_confirmation
    else:
        confirmation = feature_signal_confirmation(df)
    
    if cached_signal_timing is not None:
        timing = cached_signal_timing
    else:
        timing = feature_signal_timing(df)
    
    if cached_signal_historical_success is not None:
        historical = cached_signal_historical_success
    else:
        historical = feature_signal_historical_success(df)
    
    # Combine (weighted)
    ensemble = (quality * 0.3 + 
                strength * 0.25 + 
                confirmation * 0.2 + 
                timing * 0.15 + 
                historical * 0.1)
    
    ensemble = ensemble.clip(0.0, 1.0)
    ensemble.name = "signal_ensemble_score"
    return ensemble


def feature_false_positive_risk(df: DataFrame,
                                cached_signal_confirmation: Optional[Series] = None,
                                cached_signal_strength: Optional[Series] = None,
                                cached_signal_timing: Optional[Series] = None) -> Series:
    """
    False Positive Risk: False positive risk indicator.
    
    Measures risk of false positive signal.
    Factors: low confirmation, weak strength, poor timing, high volatility.
    Higher values = higher false positive risk, lower = lower.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get component features (use cached if available)
    if cached_signal_confirmation is not None:
        confirmation = cached_signal_confirmation
    else:
        confirmation = feature_signal_confirmation(df)
    
    if cached_signal_strength is not None:
        strength = cached_signal_strength
    else:
        strength = feature_signal_strength(df)
    
    if cached_signal_timing is not None:
        timing = cached_signal_timing
    else:
        timing = feature_signal_timing(df)
    
    volatility = feature_volatility_21d(df)
    
    # Normalize volatility (high volatility = higher risk)
    volatility_norm = volatility / (volatility.rolling(window=252, min_periods=1).max() + 1e-10)
    volatility_norm = volatility_norm.clip(0.0, 1.0)
    
    # False positive risk = inverse of quality factors
    risk = ((1.0 - confirmation) * 0.3 + 
            (1.0 - strength) * 0.3 + 
            (1.0 - timing) * 0.2 + 
            volatility_norm * 0.2)
    
    risk = risk.clip(0.0, 1.0)
    risk.name = "false_positive_risk"
    return risk


def feature_signal_divergence(df: DataFrame) -> Series:
    """
    Signal Divergence: Signal divergence indicator.
    
    Measures divergence between price and indicators.
    Calculated as: correlation between price momentum and indicator momentum.
    Lower values = more divergence (bearish), higher = less divergence (bullish).
    Normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    momentum = feature_momentum_20d(df)
    rsi = feature_rsi14(df)
    
    # Calculate momentum of price and RSI
    price_momentum = close.pct_change(5)
    rsi_momentum = rsi.diff(5)
    
    # Calculate rolling correlation
    correlation = price_momentum.rolling(window=20, min_periods=10).corr(rsi_momentum)
    
    # Convert correlation [-1, 1] to [0, 1] (1 = no divergence, 0 = high divergence)
    divergence = (correlation + 1.0) / 2.0
    divergence = divergence.fillna(0.5).clip(0.0, 1.0)
    divergence.name = "signal_divergence"
    return divergence


def feature_signal_volume_confirmation(df: DataFrame) -> Series:
    """
    Signal Volume Confirmation: Volume confirmation for signal.
    
    Measures if volume supports the signal direction.
    Combines: volume ratio, volume trend, volume momentum.
    Higher values = strong volume confirmation, lower = weak.
    Normalized to [0, 1]. Clipped for safety.
    """
    volume_ratio = feature_relative_volume(df)
    volume_trend = feature_volume_trend(df)
    volume_momentum = feature_volume_momentum(df)
    
    # Normalize components
    volume_ratio_norm = volume_ratio / (volume_ratio.rolling(window=252, min_periods=1).max() + 1e-10)
    volume_ratio_norm = volume_ratio_norm.clip(0.0, 1.0)
    volume_trend_norm = (volume_trend + 1.0) / 2.0
    volume_trend_norm = volume_trend_norm.clip(0.0, 1.0)
    volume_momentum_norm = (volume_momentum + 1.0) / 2.0
    volume_momentum_norm = volume_momentum_norm.clip(0.0, 1.0)
    
    # Combine
    confirmation = (volume_ratio_norm * 0.5 + volume_trend_norm * 0.3 + volume_momentum_norm * 0.2)
    confirmation = confirmation.clip(0.0, 1.0)
    confirmation.name = "signal_volume_confirmation"
    return confirmation


def feature_signal_trend_alignment(df: DataFrame) -> Series:
    """
    Signal Trend Alignment: Trend alignment for signal.
    
    Measures if trend is aligned with signal direction.
    Combines: trend strength, trend direction, trend consistency.
    Higher values = strong trend alignment, lower = weak.
    Normalized to [0, 1]. Clipped for safety.
    """
    trend_strength = feature_trend_strength_20d(df)
    trend_consistency = feature_trend_consistency(df)
    momentum = feature_momentum_20d(df)
    
    # Check if trend is bullish (momentum positive)
    trend_direction = (momentum > 0).astype(float)
    
    # Combine
    alignment = (trend_strength * 0.4 + trend_consistency * 0.3 + trend_direction * 0.3)
    alignment = alignment.clip(0.0, 1.0)
    alignment.name = "signal_trend_alignment"
    return alignment


def feature_signal_volatility_context(df: DataFrame) -> Series:
    """
    Signal Volatility Context: Volatility context for signal.
    
    Measures if volatility context supports the signal.
    Moderate volatility is best (too low = no movement, too high = risk).
    Higher values = favorable volatility context, lower = unfavorable.
    Normalized to [0, 1]. Clipped for safety.
    """
    volatility = feature_volatility_21d(df)
    
    # Optimal volatility range (moderate)
    # Optimized: Use vectorized rank instead of .apply() with lambda
    vol_percentile = volatility.rolling(window=252, min_periods=1).rank(pct=True, method='average')
    vol_percentile = vol_percentile.fillna(0.5)
    
    # Best volatility is in middle range (0.3 to 0.7 percentile)
    # Create bell curve around 0.5
    vol_context = 1.0 - 4.0 * (vol_percentile - 0.5) ** 2
    vol_context = vol_context.clip(0.0, 1.0)
    
    vol_context.name = "signal_volatility_context"
    return vol_context


def feature_signal_regime_alignment(df: DataFrame) -> Series:
    """
    Signal Regime Alignment: Regime alignment for signal.
    
    Measures if market regime is aligned with signal direction.
    Combines: market regime, regime strength, relative strength.
    Higher values = strong regime alignment, lower = weak.
    Normalized to [0, 1]. Clipped for safety.
    """
    regime = feature_market_regime(df)
    regime_strength = feature_regime_strength(df)
    relative_strength = feature_relative_strength_spy(df)
    
    # Fill NaN values with defaults (for initial periods or missing SPY data)
    regime = regime.fillna(2.0)  # Default to sideways
    regime_strength = regime_strength.fillna(0.5)  # Default to neutral
    relative_strength = relative_strength.fillna(0.0)  # Default to no relative strength
    
    # Normalize regime (0=bull, 1=bear, 2=sideways -> 1, 0, 0.5)
    regime_norm = pd.Series(index=regime.index, dtype=float)
    regime_norm[regime == 0] = 1.0  # Bull = aligned
    regime_norm[regime == 1] = 0.0  # Bear = not aligned
    regime_norm[regime == 2] = 0.5  # Sideways = neutral
    regime_norm = regime_norm.fillna(0.5)
    
    # Normalize relative strength
    relative_strength_norm = (relative_strength + 1.0) / 2.0
    relative_strength_norm = relative_strength_norm.clip(0.0, 1.0)
    
    # Combine
    alignment = (regime_norm * 0.4 + regime_strength * 0.3 + relative_strength_norm * 0.3)
    alignment = alignment.clip(0.0, 1.0)
    alignment.name = "signal_regime_alignment"
    return alignment


def feature_signal_momentum_strength(df: DataFrame) -> Series:
    """
    Signal Momentum Strength: Momentum strength for signal.
    
    Measures strength of momentum supporting the signal.
    Combines: momentum magnitude, momentum persistence, momentum acceleration.
    Higher values = stronger momentum, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    momentum = feature_momentum_20d(df)
    momentum_persistence = feature_momentum_persistence(df)
    momentum_acceleration = feature_momentum_acceleration(df)
    
    # Normalize components
    momentum_norm = (momentum.abs() + 0.1) / 0.2
    momentum_norm = momentum_norm.clip(0.0, 1.0)
    momentum_persistence_norm = momentum_persistence / (momentum_persistence.rolling(window=252, min_periods=1).max() + 1e-10)
    momentum_persistence_norm = momentum_persistence_norm.clip(0.0, 1.0)
    momentum_acceleration_norm = (momentum_acceleration + 1.0) / 2.0
    momentum_acceleration_norm = momentum_acceleration_norm.clip(0.0, 1.0)
    
    # Combine
    strength = (momentum_norm * 0.4 + momentum_persistence_norm * 0.35 + momentum_acceleration_norm * 0.25)
    strength = strength.clip(0.0, 1.0)
    strength.name = "signal_momentum_strength"
    return strength


def feature_signal_support_resistance(df: DataFrame) -> Series:
    """
    Signal Support/Resistance: Support/resistance context for signal.
    
    Measures if support/resistance levels support the signal.
    Combines: support distance, support strength, resistance distance.
    Higher values = favorable support/resistance context, lower = unfavorable.
    Normalized to [0, 1]. Clipped for safety.
    """
    support_dist = feature_distance_to_support(df)
    support_strength = feature_support_resistance_strength(df)
    resistance_dist = feature_distance_to_resistance(df)
    
    # Normalize components
    support_dist_norm = (support_dist + 0.1) / 0.2
    support_dist_norm = support_dist_norm.clip(0.0, 1.0)
    resistance_dist_norm = (resistance_dist + 0.1) / 0.2
    resistance_dist_norm = resistance_dist_norm.clip(0.0, 1.0)
    
    # Combine (closer to support = better for bullish, closer to resistance = worse)
    context = (support_dist_norm * 0.4 + support_strength * 0.4 + (1.0 - resistance_dist_norm) * 0.2)
    context = context.clip(0.0, 1.0)
    context.name = "signal_support_resistance"
    return context


def feature_signal_relative_strength(df: DataFrame) -> Series:
    """
    Signal Relative Strength: Relative strength for signal.
    
    Measures if relative strength supports the signal.
    Uses: relative strength vs SPY, relative strength momentum.
    Higher values = strong relative strength, lower = weak.
    Normalized to [0, 1]. Clipped for safety.
    """
    relative_strength = feature_relative_strength_spy(df)
    relative_momentum = feature_rs_momentum(df)
    
    # Fill NaN values with 0.0 before normalization (for initial periods)
    relative_strength = relative_strength.fillna(0.0)
    relative_momentum = relative_momentum.fillna(0.0)
    
    # Normalize components
    relative_strength_norm = (relative_strength + 1.0) / 2.0
    relative_strength_norm = relative_strength_norm.clip(0.0, 1.0)
    relative_momentum_norm = (relative_momentum + 1.0) / 2.0
    relative_momentum_norm = relative_momentum_norm.clip(0.0, 1.0)
    
    # Combine
    strength = (relative_strength_norm * 0.6 + relative_momentum_norm * 0.4)
    strength = strength.clip(0.0, 1.0)
    strength.name = "signal_relative_strength"
    return strength


def feature_signal_multi_timeframe(df: DataFrame) -> Series:
    """
    Signal Multi-Timeframe: Multi-timeframe confirmation.
    
    Measures if signal is confirmed across multiple timeframes.
    Combines: daily, weekly, monthly trend alignment.
    Higher values = stronger multi-timeframe confirmation, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get multi-timeframe trends
    daily_trend = feature_trend_strength_20d(df)
    weekly_trend = feature_weekly_trend_strength(df)
    
    # For monthly trend, use monthly SMA slope as proxy (since monthly_trend_strength doesn't exist)
    # Use monthly_sma_3m slope normalized
    close = _get_close_series(df)
    monthly_close = close.resample('ME').last()
    monthly_sma_3m = monthly_close.rolling(window=3, min_periods=1).mean()
    monthly_slope = monthly_sma_3m.diff(1)
    monthly_slope_daily = monthly_slope.reindex(close.index, method='ffill')
    # Normalize monthly trend to [0, 1] range
    monthly_trend = monthly_slope_daily / (monthly_slope_daily.abs().rolling(window=252, min_periods=1).max() + 1e-10)
    monthly_trend = (monthly_trend + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
    monthly_trend = monthly_trend.clip(0.0, 1.0)
    
    # Fill NaN values with defaults (for initial periods)
    daily_trend = daily_trend.fillna(0.5)  # Default to neutral trend
    weekly_trend = weekly_trend.fillna(0.5)  # Default to neutral trend
    monthly_trend = monthly_trend.fillna(0.5)  # Default to neutral trend
    
    # Combine (weighted by timeframe importance)
    multi_timeframe = (daily_trend * 0.5 + weekly_trend * 0.3 + monthly_trend * 0.2)
    multi_timeframe = multi_timeframe.clip(0.0, 1.0)
    multi_timeframe.name = "signal_multi_timeframe"
    return multi_timeframe


def feature_signal_pattern_confirmation(df: DataFrame) -> Series:
    """
    Signal Pattern Confirmation: Pattern confirmation for signal.
    
    Measures if price patterns confirm the signal.
    Uses: pattern strength, pattern confirmation, candlestick patterns.
    Higher values = stronger pattern confirmation, lower = weaker.
    Normalized to [0, 1]. Clipped for safety.
    """
    pattern_strength = feature_pattern_strength(df)
    pattern_conf = feature_pattern_confirmation(df)
    
    # Normalize pattern strength [-1, 1] to [0, 1]
    pattern_strength_norm = (pattern_strength + 1.0) / 2.0
    pattern_strength_norm = pattern_strength_norm.clip(0.0, 1.0)
    
    # Combine
    confirmation = (pattern_strength_norm * 0.6 + pattern_conf * 0.4)
    confirmation = confirmation.clip(0.0, 1.0)
    confirmation.name = "signal_pattern_confirmation"
    return confirmation


# ============================================================================
# BLOCK ML-23: Market Context (SPY-based) (3 features)
# ============================================================================
# Note: These features provide market context using SPY data


def feature_mkt_spy_dist_sma200(df: DataFrame) -> Series:
    """
    SPY Distance from SMA200: SPY Distance from SMA200 (z-score normalized).
    
    Measures how far SPY is from its 200-day SMA, indicating market extension.
    Calculated as: (SPY_close - SPY_SMA200) / SPY_SMA200, then z-score normalized.
    Positive = SPY above SMA200 (bullish), negative = below (bearish).
    Z-score normalized to standardize across different market conditions.
    No clipping - let ML learn the distribution.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None or len(spy_close) == 0:
        result = pd.Series(np.nan, index=df.index)
        result.name = "mkt_spy_dist_sma200"
        return result
    
    # Calculate SPY SMA200
    spy_sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    
    # Calculate distance as percentage
    dist_pct = (spy_close / spy_sma200) - 1.0
    
    # Z-score normalize
    mean = dist_pct.rolling(window=252, min_periods=50).mean()
    std = dist_pct.rolling(window=252, min_periods=50).std()
    z_score = (dist_pct - mean) / (std + 1e-10)
    
    z_score.name = "mkt_spy_dist_sma200"
    return z_score


def feature_mkt_spy_sma200_slope(df: DataFrame) -> Series:
    """
    SPY SMA200 Slope: SPY SMA200 Slope (percentile rank normalized).
    
    Measures the slope/trend of SPY's 200-day SMA, indicating long-term market trend.
    Calculated as: linear regression slope of SPY SMA200 over 20 days.
    Positive = rising trend, negative = falling trend.
    Percentile rank normalized to [0, 1] for consistency.
    Normalized to [0, 1]. Clipped for safety.
    """
    spy_close = _get_spy_close_aligned(df)
    if spy_close is None or len(spy_close) == 0:
        result = pd.Series(np.nan, index=df.index)
        result.name = "mkt_spy_sma200_slope"
        return result
    
    # Calculate SPY SMA200
    spy_sma200 = spy_close.rolling(window=200, min_periods=1).mean()
    
    # Calculate slope using linear regression over 20 days
    # Use vectorized approach with rolling apply for efficiency
    def calc_slope(window):
        if len(window) < 20 or np.isnan(window).any():
            return 0.0
        X = np.arange(len(window))
        y = window.values
        # Simple linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        n = len(X)
        sum_x = np.sum(X)
        sum_y = np.sum(y)
        sum_xy = np.sum(X * y)
        sum_x2 = np.sum(X * X)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0
        slope_val = (n * sum_xy - sum_x * sum_y) / denominator
        return slope_val
    
    slope = spy_sma200.rolling(window=20, min_periods=20).apply(calc_slope, raw=False)
    
    # Percentile rank normalize
    # Optimized: Use vectorized rank instead of .apply() with lambda
    percentile_rank = slope.rolling(window=252, min_periods=50).rank(pct=True, method='average')
    percentile_rank = percentile_rank.fillna(0.5).clip(0.0, 1.0)
    percentile_rank.name = "mkt_spy_sma200_slope"
    return percentile_rank


def feature_beta_spy_252d(df: DataFrame) -> Series:
    """
    Beta vs SPY: Rolling beta vs SPY (252-day) normalized to [0, 1].
    
    Measures stock's sensitivity to market movements (SPY).
    Calculated as: rolling covariance(stock_ret, spy_ret) / variance(spy_ret) over 252 days.
    Beta = 1 means stock moves with market, >1 means more volatile, <1 means less.
    Normalized by: ((beta + 1) / 4).clip(0, 1) to map [-1, 3] to [0, 1].
    Normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    spy_close = _get_spy_close_aligned(df)
    
    if spy_close is None or len(spy_close) == 0:
        result = pd.Series(np.nan, index=df.index)
        result.name = "beta_spy_252d"
        return result
    
    # Calculate log returns
    stock_ret = np.log(close / close.shift(1))
    spy_ret = np.log(spy_close / spy_close.shift(1))
    
    # Create DataFrame for rolling operations
    returns_df = pd.DataFrame({
        'stock_ret': stock_ret,
        'spy_ret': spy_ret
    })
    
    # Calculate rolling means
    stock_mean = returns_df['stock_ret'].rolling(window=252, min_periods=50).mean()
    spy_mean = returns_df['spy_ret'].rolling(window=252, min_periods=50).mean()
    
    # Calculate rolling covariance: E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
    stock_spy_product = (returns_df['stock_ret'] * returns_df['spy_ret']).rolling(window=252, min_periods=50).mean()
    cov = stock_spy_product - (stock_mean * spy_mean)
    
    # Calculate rolling variance of SPY returns
    var = returns_df['spy_ret'].rolling(window=252, min_periods=50).var()
    
    # Calculate beta: cov / var
    # Avoid division by zero
    var = var.replace(0, np.nan)
    beta = cov / var
    
    # Normalize: ((beta + 1) / 4).clip(0, 1)
    # This maps beta from [-1, 3] to [0, 1]
    # beta = -1 → normalized = 0
    # beta = 0 → normalized = 0.25
    # beta = 1 → normalized = 0.5
    # beta = 3 → normalized = 1
    beta_norm = ((beta + 1) / 4).clip(0, 1)
    
    beta_norm.name = "beta_spy_252d"
    return beta_norm


# ============================================================================
# BLOCK ML-24: Breakout & Channel Features (4 features)
# ============================================================================
# Note: donchian_position and donchian_breakout already exist (from ML-6.3)
# Adding TTM Squeeze features here


def feature_ttm_squeeze_on(df: DataFrame) -> Series:
    """
    TTM Squeeze On: TTM Squeeze binary flag.
    
    Detects volatility contraction (squeeze) by comparing Bollinger Bands to Keltner Channels.
    TTM Squeeze is a popular breakout indicator that identifies tight trading ranges.
    
    Calculation:
    1. Bollinger Bands: mid = SMA(20), std = rolling_std(20), upper_bb = mid + 2*std, lower_bb = mid - 2*std
    2. Keltner Channels: atr = ATR(20), upper_kc = mid + 1.5*atr, lower_kc = mid - 1.5*atr
    3. Squeeze condition: (lower_bb > lower_kc) & (upper_bb < upper_kc)
    
    Binary flag: 1 = squeeze active (BB inside KC), 0 = no squeeze.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Step 1: Calculate Bollinger Bands
    mid = close.rolling(window=20, min_periods=1).mean()  # SMA20
    std = close.rolling(window=20, min_periods=1).std()
    upper_bb = mid + 2 * std
    lower_bb = mid - 2 * std
    
    # Step 2: Calculate Keltner Channels
    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR(20)
    atr = tr.rolling(window=20, min_periods=1).mean()
    
    # Keltner Channels
    upper_kc = mid + 1.5 * atr
    lower_kc = mid - 1.5 * atr
    
    # Step 3: Squeeze condition
    # Squeeze is ON when Bollinger Bands are inside Keltner Channels
    squeeze_on = ((lower_bb > lower_kc) & (upper_bb < upper_kc)).astype(float)
    squeeze_on = squeeze_on.fillna(0.0).clip(0.0, 1.0)
    squeeze_on.name = "ttm_squeeze_on"
    return squeeze_on


def feature_ttm_squeeze_momentum(df: DataFrame) -> Series:
    """
    TTM Squeeze Momentum: TTM Squeeze momentum.
    
    Provides momentum direction during squeeze conditions.
    Measures how far price is from the 20-day moving average, normalized by price.
    
    Calculation:
    1. mid = SMA(close, 20)
    2. squeeze_momentum = close - mid
    3. squeeze_momentum_norm = squeeze_momentum / close
    
    Positive = price above SMA20 (bullish), negative = below (bearish).
    Normalized by price for comparability across different price ranges.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate SMA20 (mid)
    mid = close.rolling(window=20, min_periods=1).mean()
    
    # Step 2: Calculate momentum
    squeeze_momentum = close - mid
    
    # Step 3: Normalize by price
    # Handle division by zero (when close is 0, which shouldn't happen but safety first)
    close_safe = close.replace(0, np.nan)
    squeeze_momentum_norm = squeeze_momentum / close_safe
    
    squeeze_momentum_norm.name = "ttm_squeeze_momentum"
    return squeeze_momentum_norm


# ============================================================================
# BLOCK ML-25: Trend Exhaustion & Consolidation (4 features)
# ============================================================================


def feature_trend_consolidation(df: DataFrame) -> Series:
    """
    Trend Consolidation: Binary flag indicating price consolidating within trend.
    
    Detects when price is consolidating (moving sideways) within an overall trend.
    Consolidation is identified by:
    1. Price is in a trend (above/below SMA50)
    2. Price range is tight (low volatility relative to recent average)
    3. Price is not making new highs/lows (stuck in range)
    
    Binary flag: 1 = consolidating within trend, 0 = not consolidating.
    Already normalized (binary 0/1). Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate SMA50 to determine trend direction
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Determine if in uptrend or downtrend
    in_uptrend = close > sma50
    in_downtrend = close < sma50
    
    # Calculate price range (high - low) over 10 days
    price_range = (high.rolling(window=10, min_periods=1).max() - 
                   low.rolling(window=10, min_periods=1).min())
    
    # Calculate average price range over 50 days
    avg_range = price_range.rolling(window=50, min_periods=1).mean()
    
    # Consolidation: tight range relative to average
    tight_range = price_range < (avg_range * 0.7)
    
    # Check if price is stuck in range (not making new highs/lows)
    # For uptrend: not making new 20-day highs
    # For downtrend: not making new 20-day lows
    new_highs = high >= high.rolling(window=20, min_periods=1).max().shift(1)
    new_lows = low <= low.rolling(window=20, min_periods=1).min().shift(1)
    
    # Consolidation in uptrend: tight range and not making new highs
    consolidation_uptrend = in_uptrend & tight_range & (~new_highs)
    
    # Consolidation in downtrend: tight range and not making new lows
    consolidation_downtrend = in_downtrend & tight_range & (~new_lows)
    
    # Overall consolidation
    consolidation = (consolidation_uptrend | consolidation_downtrend).astype(float)
    consolidation = consolidation.fillna(0.0).clip(0.0, 1.0)
    consolidation.name = "trend_consolidation"
    return consolidation


def feature_trend_exhaustion(df: DataFrame) -> Series:
    """
    Trend Exhaustion: Trend exhaustion indicator.
    
    Detects when a trend is losing momentum and may be exhausted.
    Factors:
    1. Decreasing momentum (momentum slowing down)
    2. Decreasing volume (less participation)
    3. Divergence (price making new highs/lows but indicators not)
    4. Extreme price extension (far from moving averages)
    
    Higher values = more exhausted, lower = less exhausted.
    Normalized to [0, 1]. Clipped for safety.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    # Get component features
    momentum = feature_momentum_20d(df)
    momentum_accel = feature_momentum_acceleration(df)
    volume_ratio = feature_relative_volume(df)
    rsi = feature_rsi14(df)
    
    # Calculate price extension from SMA50
    sma50 = close.rolling(window=50, min_periods=1).mean()
    price_extension = abs((close - sma50) / sma50)
    
    # Normalize components
    # Decreasing momentum (negative acceleration)
    momentum_decreasing = (momentum_accel < 0).astype(float)
    
    # Low volume (below average)
    volume_low = (volume_ratio < 0.8).astype(float)
    
    # Extreme RSI (overbought/oversold)
    rsi_extreme = ((rsi > 70) | (rsi < 30)).astype(float)
    
    # Price extension (far from MA)
    price_extended = (price_extension > price_extension.rolling(window=252, min_periods=1).quantile(0.75)).astype(float)
    
    # Combine exhaustion signals
    exhaustion = (momentum_decreasing * 0.3 + 
                  volume_low * 0.25 + 
                  rsi_extreme * 0.25 + 
                  price_extended * 0.2)
    
    exhaustion = exhaustion.clip(0.0, 1.0)
    exhaustion.name = "trend_exhaustion"
    return exhaustion


def feature_trend_vs_mean_reversion(df: DataFrame) -> Series:
    """
    Trend vs Mean-Reversion: Trend vs mean-reversion signal.
    
    Determines if current conditions favor trend-following or mean-reversion strategies.
    Factors:
    1. Trend strength (strong trend = trend-following, weak = mean-reversion)
    2. Volatility regime (low volatility = mean-reversion, high = trend)
    3. Price position in range (extreme = mean-reversion, middle = trend)
    4. Momentum persistence (persistent = trend, choppy = mean-reversion)
    
    Higher values = trend-following favored, lower = mean-reversion favored.
    Normalized to [0, 1]. Clipped for safety.
    """
    # Get component features
    trend_strength = feature_trend_strength_20d(df)
    volatility = feature_volatility_21d(df)
    donchian_pos = feature_donchian_position(df)
    momentum_persistence = feature_momentum_persistence(df)
    
    # Normalize components
    # Trend strength (already [0, 1])
    trend_norm = trend_strength
    
    # Volatility (low = mean-reversion, high = trend)
    # Optimized: Use vectorized rank instead of .apply() with lambda
    vol_percentile = volatility.rolling(window=252, min_periods=1).rank(pct=True, method='average')
    vol_norm = vol_percentile.fillna(0.5)
    
    # Price position (extreme = mean-reversion, middle = trend)
    # Distance from 0.5 (middle)
    pos_distance = abs(donchian_pos - 0.5)
    pos_norm = 1.0 - (pos_distance * 2.0)  # Closer to middle = higher
    pos_norm = pos_norm.clip(0.0, 1.0)
    
    # Momentum persistence (already normalized)
    momentum_pers_norm = momentum_persistence / (momentum_persistence.rolling(window=252, min_periods=1).max() + 1e-10)
    momentum_pers_norm = momentum_pers_norm.clip(0.0, 1.0)
    
    # Combine (weighted)
    trend_vs_mean = (trend_norm * 0.35 + 
                     vol_norm * 0.25 + 
                     pos_norm * 0.2 + 
                     momentum_pers_norm * 0.2)
    
    trend_vs_mean = trend_vs_mean.clip(0.0, 1.0)
    trend_vs_mean.name = "trend_vs_mean_reversion"
    return trend_vs_mean


def feature_volatility_jump(df: DataFrame) -> Series:
    """
    Volatility Jump: Volatility jump detection.
    
    Detects sudden increases in volatility (volatility jumps).
    Calculated as: current volatility / average volatility over longer period.
    A jump occurs when current volatility is significantly higher than recent average.
    
    Higher values = larger volatility jump, lower = no jump.
    Normalized to [0, 1]. Clipped for safety.
    """
    volatility = feature_volatility_21d(df)
    
    # Calculate short-term and long-term average volatility
    vol_short = volatility.rolling(window=5, min_periods=1).mean()
    vol_long = volatility.rolling(window=50, min_periods=1).mean()
    
    # Volatility jump ratio
    vol_jump_ratio = vol_short / (vol_long + 1e-10)
    
    # Normalize to [0, 1] using percentile rank
    # Higher jump ratio = larger jump
    # Optimized: Use vectorized rank instead of .apply() with lambda
    vol_jump_norm = vol_jump_ratio.rolling(window=252, min_periods=50).rank(pct=True, method='average')
    vol_jump_norm = vol_jump_norm.fillna(0.5).clip(0.0, 1.0)
    vol_jump_norm.name = "volatility_jump"
    return vol_jump_norm


# ============================================================================
# BLOCK ML-26.1: Enhanced Volatility Features (10 features)
# ============================================================================


def feature_volatility_forecast_accuracy(df: DataFrame) -> Series:
    """
    Volatility Estimation Consistency: Consistency between EWMA and rolling sum volatility estimates.
    
    NOTE: Despite the name "accuracy", this does NOT measure true forecast accuracy (comparing
    past forecasts to present outcomes). Instead, it measures the consistency/divergence between
    two different volatility estimation methods:
    - EWMA (exponentially weighted, more reactive to recent data)
    - Rolling sum (equal weight, more stable)
    
    Both methods use current data, so this is a measure of estimation method divergence rather
    than forecast accuracy. High values indicate the two methods agree (stable volatility regime),
    low values indicate divergence (volatility regime change or clustering).
    
    Calculated as: 1 - |realized - forecast| / (realized + 1e-10) over 20-day rolling window.
    Higher values = methods agree (consistent estimation), lower = methods diverge (regime change).
    Normalized to [0, 1]. Clipped for safety.
    """
    forecast = feature_volatility_forecast(df)
    realized = feature_realized_volatility_20d(df)
    
    # Calculate consistency/divergence over rolling window (20 days)
    consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 20:
            consistency.iloc[i] = 0.5  # Default for early periods
            continue
        
        start_idx = max(0, i - 20)
        window_forecast = forecast.iloc[start_idx:i+1]
        window_realized = realized.iloc[start_idx:i+1]
        
        # Mean absolute error between the two estimation methods
        mae = (window_realized - window_forecast).abs().mean()
        mean_realized = window_realized.mean()
        
        # Consistency: 1 - normalized error (higher = methods agree)
        if mean_realized > 0:
            normalized_error = mae / (mean_realized + 1e-10)
            cons = 1.0 - normalized_error
        else:
            cons = 0.5
        
        consistency.iloc[i] = cons
    
    consistency = consistency.fillna(0.5).clip(0.0, 1.0)
    consistency.name = "volatility_forecast_accuracy"  # Keep name for backward compatibility
    return consistency


def feature_volatility_forecast_error(df: DataFrame) -> Series:
    """
    Volatility Forecast Error: Forecast error magnitude.
    
    NOTE: This compares current forecast to current realized volatility (both use current data).
    For true forecast error, see feature_volatility_forecast_error_true.
    
    Calculated as: |realized - forecast| / forecast.
    Higher values = larger divergence between EWMA and rolling sum methods.
    No clipping - let ML learn the distribution.
    """
    forecast = feature_volatility_forecast(df)
    realized = feature_realized_volatility_20d(df)
    
    error = (realized - forecast).abs() / (forecast + 1e-10)
    error.name = "volatility_forecast_error"
    return error


def feature_volatility_forecast_error_true(df: DataFrame) -> Series:
    """
    True Volatility Forecast Error: Actual forecast error (past forecast vs present outcome).
    
    This is a TRUE forecast accuracy measure: compares the forecast made at time t-1
    to the realized volatility outcome at time t. This measures how well the forecast
    actually predicted future volatility.
    
    Calculated as: |realized[t] - forecast[t-1]| / forecast[t-1].
    Higher values = larger forecast errors (forecast was less accurate).
    Lower values = smaller forecast errors (forecast was more accurate).
    No clipping - let ML learn the distribution.
    """
    forecast = feature_volatility_forecast(df)
    realized = feature_realized_volatility_20d(df)
    
    # Shift forecast by 1 to use past forecast
    forecast_shifted = forecast.shift(1)
    
    # Compare past forecast to current realized outcome
    error = (realized - forecast_shifted).abs() / (forecast_shifted + 1e-10)
    error = error.fillna(0.0)  # Fill NaN from shift
    error.name = "volatility_forecast_error_true"
    return error


def feature_volatility_forecast_trend(df: DataFrame) -> Series:
    """
    Volatility Forecast Trend: Trend in forecast (slope over 5-10 days).
    
    Measures whether volatility forecast is increasing or decreasing.
    Calculated as: slope of forecast over 7-day window.
    Positive = increasing forecast, negative = decreasing.
    No clipping - let ML learn the distribution.
    """
    forecast = feature_volatility_forecast(df)
    
    # Calculate slope over 7-day window
    trend = forecast.rolling(window=7, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x) >= 2 else 0.0,
        raw=False
    )
    
    trend = trend.fillna(0.0)
    trend.name = "volatility_forecast_trend"
    return trend


def feature_volatility_clustering_strength(df: DataFrame) -> Series:
    """
    Volatility Clustering Strength: Strength of volatility clustering (autocorrelation).
    
    Measures how much volatility clusters (high vol followed by high vol).
    Calculated as: autocorrelation of volatility at lag 1.
    Higher values = stronger clustering.
    Normalized to [0, 1]. Clipped for safety.
    """
    volatility = feature_volatility_21d(df)
    
    # Calculate autocorrelation at lag 1 over rolling window
    clustering = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 50:
            clustering.iloc[i] = 0.5
            continue
        
        start_idx = max(0, i - 50)
        window_vol = volatility.iloc[start_idx:i+1]
        
        if len(window_vol) >= 2 and window_vol.std() > 0:
            # Autocorrelation at lag 1
            vol_shifted = window_vol.shift(1)
            corr = window_vol.corr(vol_shifted)
            clustering.iloc[i] = corr if not pd.isna(corr) else 0.5
        else:
            clustering.iloc[i] = 0.5
    
    # Normalize from [-1, 1] to [0, 1]
    clustering = (clustering + 1.0) / 2.0
    clustering = clustering.fillna(0.5).clip(0.0, 1.0)
    clustering.name = "volatility_clustering_strength"
    return clustering


def feature_volatility_regime_forecast(df: DataFrame) -> Series:
    """
    Volatility Regime Forecast: Forecast of volatility regime (high/low based on trajectory).
    
    Predicts whether volatility regime will be high or low based on forecast trajectory.
    Calculated as: combination of forecast trend and current regime.
    Values: 0.0 = low regime forecast, 1.0 = high regime forecast.
    Normalized to [0, 1]. Clipped for safety.
    """
    forecast = feature_volatility_forecast(df)
    current_regime = feature_volatility_regime(df)
    forecast_trend = feature_volatility_forecast_trend(df)
    
    # Normalize forecast trend to [0, 1]
    trend_norm = (forecast_trend - forecast_trend.rolling(window=252, min_periods=1).min()) / (
        forecast_trend.rolling(window=252, min_periods=1).max() - 
        forecast_trend.rolling(window=252, min_periods=1).min() + 1e-10
    )
    trend_norm = trend_norm.fillna(0.5).clip(0.0, 1.0)
    
    # Normalize current regime to [0, 1] (it's already 0-100)
    regime_norm = current_regime / 100.0
    
    # Combine: 60% current regime, 40% trend
    regime_forecast = (regime_norm * 0.6 + trend_norm * 0.4)
    regime_forecast = regime_forecast.clip(0.0, 1.0)
    regime_forecast.name = "volatility_regime_forecast"
    return regime_forecast


def feature_volatility_surprise(df: DataFrame) -> Series:
    """
    Volatility Surprise: Volatility surprise (realized vs expected).
    
    Measures how much realized volatility differs from forecast (surprise).
    Calculated as: (realized - forecast) / forecast.
    Positive = realized higher than expected (positive surprise), negative = lower.
    No clipping - let ML learn the distribution.
    """
    forecast = feature_volatility_forecast(df)
    realized = feature_realized_volatility_20d(df)
    
    surprise = (realized - forecast) / (forecast + 1e-10)
    surprise.name = "volatility_surprise"
    return surprise


def feature_volatility_realized_forecast_ratio(df: DataFrame) -> Series:
    """
    Volatility Realized Forecast Ratio: Realized / forecast ratio.
    
    Compares realized volatility to forecast.
    Calculated as: realized_volatility_20d / volatility_forecast.
    >1 = realized higher than forecast, <1 = realized lower.
    No clipping - let ML learn the distribution.
    """
    realized = feature_realized_volatility_20d(df)
    forecast = feature_volatility_forecast(df)
    
    ratio = realized / (forecast + 1e-10)
    ratio.name = "volatility_realized_forecast_ratio"
    return ratio


# ============================================================================
# BLOCK ML-26.2: Enhanced Gain Probability Features (5 features)
# ============================================================================

def feature_gain_probability_trend(df: DataFrame, cached_gain_prob_score: Optional[Series] = None) -> Series:
    """
    Gain Probability Rank: Percentile rank of gain probability (like momentum_rank).
    
    Percentile rank of gain_probability_score vs last 252 days.
    Values: 0.0 = lowest probability, 1.0 = highest probability.
    Normalized to [0, 1]. Clipped for safety.
    """
    if cached_gain_prob_score is not None:
        gain_prob = cached_gain_prob_score
    else:
        gain_prob = feature_gain_probability_score(df)
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rank = gain_prob.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    rank = rank.fillna(0.5)
    
    rank = rank.fillna(0.5).clip(0.0, 1.0)
    rank.name = "gain_probability_rank"
    return rank


def feature_gain_probability_trend(df: DataFrame, cached_gain_prob_score: Optional[Series] = None) -> Series:
    """
    Gain Probability Trend: Trend in gain probability (increasing/decreasing).
    
    Measures whether gain probability is increasing or decreasing.
    Calculated as: slope of gain_probability_score over 10-day window.
    Positive = increasing, negative = decreasing.
    Normalized to [-1, 1] then to [0, 1]. Clipped for safety.
    """
    if cached_gain_prob_score is not None:
        gain_prob = cached_gain_prob_score
    else:
        gain_prob = feature_gain_probability_score(df)
    
    # Calculate slope over 10-day window
    trend = gain_prob.rolling(window=10, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x) >= 2 else 0.0,
        raw=False
    )
    
    # Normalize to [0, 1]
    trend_norm = (trend - trend.rolling(window=252, min_periods=1).min()) / (
        trend.rolling(window=252, min_periods=1).max() - 
        trend.rolling(window=252, min_periods=1).min() + 1e-10
    )
    trend_norm = trend_norm.fillna(0.5).clip(0.0, 1.0)
    trend_norm.name = "gain_probability_trend"
    return trend_norm


def feature_gain_probability_consistency_rank(df: DataFrame, cached_gain_consistency: Optional[Series] = None) -> Series:
    """
    Gain Probability Consistency Rank: Rank of gain consistency.
    
    Percentile rank of gain_consistency vs last 252 days.
    Values: 0.0 = lowest consistency, 1.0 = highest consistency.
    Normalized to [0, 1]. Clipped for safety.
    """
    if cached_gain_consistency is not None:
        gain_consistency = cached_gain_consistency
    else:
        gain_consistency = feature_gain_consistency(df)
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rank = gain_consistency.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    rank = rank.fillna(0.5)
    
    rank = rank.fillna(0.5).clip(0.0, 1.0)
    rank.name = "gain_probability_consistency_rank"
    return rank


def feature_gain_probability_volatility_adjusted(df: DataFrame, cached_gain_prob_score: Optional[Series] = None) -> Series:
    """
    Gain Probability Volatility Adjusted: Risk-adjusted opportunity.
    
    Calculated as: gain_probability / volatility_forecast.
    Higher values = better risk-adjusted opportunity.
    No clipping - let ML learn the distribution.
    """
    if cached_gain_prob_score is not None:
        gain_prob = cached_gain_prob_score
    else:
        gain_prob = feature_gain_probability_score(df)
    vol_forecast = feature_volatility_forecast(df)
    
    adjusted = gain_prob / (vol_forecast + 1e-10)
    adjusted.name = "gain_probability_volatility_adjusted"
    return adjusted


# ============================================================================
# BLOCK ML-26.3: Distance-Based Rank Features (5 features)
# ============================================================================


def feature_dist_support_rank(df: DataFrame) -> Series:
    """
    Distance to Support Rank: Percentile rank of distance to support.
    
    Percentile rank of distance_to_support vs last 252 days.
    Values: 0.0 = closest to support, 1.0 = farthest from support.
    Normalized to [0, 1]. Clipped for safety.
    """
    dist = feature_distance_to_support(df)
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rank = dist.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    rank = rank.fillna(0.5)
    
    rank = rank.fillna(0.5).clip(0.0, 1.0)
    rank.name = "dist_support_rank"
    return rank


# ============================================================================
# BLOCK ML-26.4: Percentile/Rank Features (8 features)
# ============================================================================


def feature_volatility_rank(df: DataFrame) -> Series:
    """
    Volatility Rank: Percentile rank of volatility (volatility_21d vs historical).
    
    Percentile rank of volatility_21d vs last 252 days.
    Values: 0.0 = lowest volatility, 1.0 = highest volatility.
    Normalized to [0, 1]. Clipped for safety.
    """
    volatility = feature_volatility_21d(df)
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rank = volatility.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    rank = rank.fillna(0.5)
    
    rank = rank.fillna(0.5).clip(0.0, 1.0)
    rank.name = "volatility_rank"
    return rank


def feature_return_rank_3m(df: DataFrame) -> Series:
    """
    Return Rank 3m: Percentile rank of 3-month return.
    
    Percentile rank of monthly_return_3m vs last 252 days.
    Values: 0.0 = lowest return, 1.0 = highest return.
    Normalized to [0, 1]. Clipped for safety.
    """
    return_3m = feature_monthly_return_3m(df)
    
    # Percentile rank over 252 days
    # Optimized: Use vectorized rank instead of .apply() with lambda
    rank = return_3m.rolling(window=252, min_periods=20).rank(pct=True, method='average')
    rank = rank.fillna(0.5)
    
    rank = rank.fillna(0.5).clip(0.0, 1.0)
    rank.name = "return_rank_3m"
    return rank


def feature_momentum_rank_trend(df: DataFrame) -> Series:
    """
    Momentum Rank Trend: Trend in momentum rank (improving/deteriorating).
    
    Measures whether momentum rank is improving or deteriorating.
    Calculated as: slope of momentum_rank over 10-day window.
    Positive = improving, negative = deteriorating.
    Normalized to [0, 1]. Clipped for safety.
    """
    momentum_rank = feature_momentum_rank(df)
    
    # Calculate slope over 10-day window
    trend = momentum_rank.rolling(window=10, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x.values, 1)[0] if len(x) >= 2 else 0.0,
        raw=False
    )
    
    # Normalize to [0, 1]
    trend_norm = (trend - trend.rolling(window=252, min_periods=1).min()) / (
        trend.rolling(window=252, min_periods=1).max() - 
        trend.rolling(window=252, min_periods=1).min() + 1e-10
    )
    trend_norm = trend_norm.fillna(0.5).clip(0.0, 1.0)
    trend_norm.name = "momentum_rank_trend"
    return trend_norm


# ============================================================================
# BLOCK ML-26.5: Volatility-Volume Interactions (4 features)
# ============================================================================


def feature_volatility_volume_correlation(df: DataFrame) -> Series:
    """
    Volatility Volume Correlation: Rolling correlation between volatility and volume.
    
    Measures correlation between volatility and volume over rolling window.
    Calculated as: rolling correlation(volatility_21d, volume) over 20 days.
    Values: -1 to 1, normalized to [0, 1]. Clipped for safety.
    """
    volatility = feature_volatility_21d(df)
    volume = _get_volume_series(df)
    
    # Rolling correlation over 20 days
    correlation = volatility.rolling(window=20, min_periods=5).corr(volume)
    
    # Normalize from [-1, 1] to [0, 1]
    correlation_norm = (correlation + 1.0) / 2.0
    correlation_norm = correlation_norm.fillna(0.5).clip(0.0, 1.0)
    correlation_norm.name = "volatility_volume_correlation"
    return correlation_norm


def feature_volatility_volume_divergence(df: DataFrame) -> Series:
    """
    Volatility Volume Divergence: Divergence between volatility and volume.
    
    Measures when volatility and volume move in opposite directions.
    Calculated as: -correlation (negative correlation = divergence).
    Higher values = more divergence.
    Normalized to [0, 1]. Clipped for safety.
    """
    correlation = feature_volatility_volume_correlation(df)
    
    # Divergence = 1 - correlation (inverse)
    divergence = 1.0 - correlation
    divergence = divergence.clip(0.0, 1.0)
    divergence.name = "volatility_volume_divergence"
    return divergence


def feature_volatility_forecast_volume_confirmation(df: DataFrame) -> Series:
    """
    Volatility Forecast Volume Confirmation: Volume confirms volatility forecast (interaction).
    
    Interaction feature: volatility_forecast × volume_ratio.
    Higher values = volume confirms high volatility forecast.
    No clipping - let ML learn the distribution.
    """
    vol_forecast = feature_volatility_forecast(df)
    volume_ratio = feature_relative_volume(df)
    
    # Interaction: forecast × volume ratio
    interaction = vol_forecast * volume_ratio
    interaction.name = "volatility_forecast_volume_confirmation"
    return interaction


# ============================================================================
# BLOCK ML-26.6: Return-Volatility Interactions (3 features)
# ============================================================================


def feature_return_volatility_ratio(df: DataFrame) -> Series:
    """
    Return Volatility Ratio: Return / volatility ratio (risk-adjusted return).
    
    Calculated as: monthly_return_3m / volatility_21d.
    Higher values = better risk-adjusted returns.
    No clipping - let ML learn the distribution.
    """
    return_3m = feature_monthly_return_3m(df)
    volatility = feature_volatility_21d(df)
    
    ratio = return_3m / (volatility + 1e-10)
    ratio.name = "return_volatility_ratio"
    return ratio


# ============================================================================
# BLOCK ML-26.7: Regime Transition Features (3 features)
# ============================================================================


def feature_volatility_regime_transition_probability(df: DataFrame) -> Series:
    """
    Volatility Regime Transition Probability: Probability of volatility regime change.
    
    Estimates probability that volatility regime will change soon.
    Calculated as: combination of regime duration and recent volatility changes.
    Values: 0.0 = low probability, 1.0 = high probability.
    Normalized to [0, 1]. Clipped for safety.
    """
    regime = feature_volatility_regime(df)
    regime_duration = pd.Series(index=df.index, dtype=float)
    
    # Calculate regime duration (days in current regime)
    current_regime = None
    duration = 0
    for i in range(len(df)):
        if pd.isna(regime.iloc[i]):
            regime_duration.iloc[i] = 0
            continue
        
        if current_regime is None or abs(regime.iloc[i] - current_regime) > 10:  # Regime change threshold
            current_regime = regime.iloc[i]
            duration = 1
        else:
            duration += 1
        
        regime_duration.iloc[i] = duration
    
    # Recent volatility change
    vol_change = regime.diff().abs()
    
    # Transition probability: longer duration + recent changes = higher probability
    duration_norm = regime_duration / (regime_duration.rolling(window=252, min_periods=1).max() + 1e-10)
    change_norm = vol_change / (vol_change.rolling(window=252, min_periods=1).max() + 1e-10)
    
    # Combine: 60% duration, 40% recent change
    transition_prob = (duration_norm * 0.6 + change_norm * 0.4)
    transition_prob = transition_prob.fillna(0.5).clip(0.0, 1.0)
    transition_prob.name = "volatility_regime_transition_probability"
    return transition_prob


def feature_gain_regime_transition_probability(df: DataFrame, cached_gain_regime: Optional[Series] = None, cached_gain_prob_score: Optional[Series] = None) -> Series:
    """
    Gain Regime Transition Probability: Probability of gain regime change.
    
    Estimates probability that gain regime will change soon.
    Calculated as: combination of regime duration and recent gain probability changes.
    Values: 0.0 = low probability, 1.0 = high probability.
    Normalized to [0, 1]. Clipped for safety.
    """
    if cached_gain_regime is not None:
        gain_regime = cached_gain_regime
    else:
        gain_regime = feature_gain_regime(df)
    
    if cached_gain_prob_score is not None:
        gain_prob = cached_gain_prob_score
    else:
        gain_prob = feature_gain_probability_score(df)
    
    # Calculate regime duration
    regime_duration = pd.Series(index=df.index, dtype=float)
    current_regime = None
    duration = 0
    for i in range(len(df)):
        if pd.isna(gain_regime.iloc[i]):
            regime_duration.iloc[i] = 0
            continue
        
        if current_regime is None or abs(gain_regime.iloc[i] - current_regime) > 0.2:  # Regime change threshold
            current_regime = gain_regime.iloc[i]
            duration = 1
        else:
            duration += 1
        
        regime_duration.iloc[i] = duration
    
    # Recent gain probability change
    prob_change = gain_prob.diff().abs()
    
    # Transition probability
    duration_norm = regime_duration / (regime_duration.rolling(window=252, min_periods=1).max() + 1e-10)
    change_norm = prob_change / (prob_change.rolling(window=252, min_periods=1).max() + 1e-10)
    
    transition_prob = (duration_norm * 0.6 + change_norm * 0.4)
    transition_prob = transition_prob.fillna(0.5).clip(0.0, 1.0)
    transition_prob.name = "gain_regime_transition_probability"
    return transition_prob


def feature_momentum_regime_transition(df: DataFrame) -> Series:
    """
    Momentum Regime Transition: Momentum regime transition detection.
    
    Detects when momentum regime is transitioning.
    Calculated as: binary signal when momentum_regime changes.
    Values: 0.0 = no transition, 1.0 = transition detected.
    Normalized to [0, 1]. Clipped for safety.
    """
    momentum_regime = feature_momentum_regime(df)
    
    # Detect regime changes (significant change in regime value)
    regime_change = momentum_regime.diff().abs()
    transition = (regime_change > 0.3).astype(float)  # Threshold for transition
    
    transition = transition.fillna(0.0).clip(0.0, 1.0)
    transition.name = "momentum_regime_transition"
    return transition


# ============================================================================
# BLOCK ML-26.8: Composite/Ensemble Features (6 features)
# ============================================================================


def feature_top_features_ensemble(df: DataFrame, cached_gain_prob_score: Optional[Series] = None) -> Series:
    """
    Top Features Ensemble: Weighted ensemble of top 5 features.
    
    Combines top-performing features based on analysis:
    - volatility_forecast (18.1%)
    - volatility_21d (13.5%)
    - volume_imbalance (2.8%)
    - gain_probability_score (1.5%)
    - momentum_rank (estimated)
    
    Weighted combination normalized to [0, 1]. Clipped for safety.
    """
    vol_forecast = feature_volatility_forecast(df)
    vol_21d = feature_volatility_21d(df)
    vol_imbalance = feature_volume_imbalance(df)
    if cached_gain_prob_score is not None:
        gain_prob = cached_gain_prob_score
    else:
        gain_prob = feature_gain_probability_score(df)
    momentum = feature_momentum_rank(df)
    
    # Normalize each feature to [0, 1] for combination
    vol_forecast_norm = (vol_forecast - vol_forecast.rolling(window=252, min_periods=1).min()) / (
        vol_forecast.rolling(window=252, min_periods=1).max() - 
        vol_forecast.rolling(window=252, min_periods=1).min() + 1e-10
    )
    vol_21d_norm = (vol_21d - vol_21d.rolling(window=252, min_periods=1).min()) / (
        vol_21d.rolling(window=252, min_periods=1).max() - 
        vol_21d.rolling(window=252, min_periods=1).min() + 1e-10
    )
    vol_imbalance_norm = (vol_imbalance - vol_imbalance.rolling(window=252, min_periods=1).min()) / (
        vol_imbalance.rolling(window=252, min_periods=1).max() - 
        vol_imbalance.rolling(window=252, min_periods=1).min() + 1e-10
    )
    
    # Weighted combination (based on importance percentages)
    ensemble = (vol_forecast_norm * 0.40 +  # volatility_forecast: 18.1% / 45% total
                vol_21d_norm * 0.30 +        # volatility_21d: 13.5% / 45%
                vol_imbalance_norm * 0.15 +  # volume_imbalance: 2.8% / 45%
                gain_prob * 0.10 +           # gain_probability_score: 1.5% / 45%
                momentum * 0.05)             # momentum_rank: estimated
    
    ensemble = ensemble.fillna(0.5).clip(0.0, 1.0)
    ensemble.name = "top_features_ensemble"
    return ensemble


def feature_volatility_forecast_accuracy_weighted(df: DataFrame) -> Series:
    """
    Volatility Forecast Consistency Weighted: Consistency-weighted volatility forecast.
    
    Adjusts volatility forecast by the consistency between EWMA and rolling sum methods.
    Calculated as: volatility_forecast × volatility_forecast_accuracy.
    Higher consistency = more weight on forecast (when methods agree, forecast is more reliable).
    Lower consistency = less weight (when methods diverge, indicates regime uncertainty).
    No clipping - let ML learn the distribution.
    """
    vol_forecast = feature_volatility_forecast(df)
    accuracy = feature_volatility_forecast_accuracy(df)
    
    weighted = vol_forecast * accuracy
    weighted.name = "volatility_forecast_accuracy_weighted"  # Keep name for backward compatibility
    return weighted


def feature_gain_probability_volatility_regime_interaction(df: DataFrame, cached_gain_prob_score: Optional[Series] = None) -> Series:
    """
    Gain Probability Volatility Regime Interaction: gain_probability × volatility_regime.
    
    Interaction feature combining gain probability and volatility regime.
    Higher values = high gain probability AND high volatility regime.
    Normalized to [0, 1]. Clipped for safety.
    """
    if cached_gain_prob_score is not None:
        gain_prob = cached_gain_prob_score
    else:
        gain_prob = feature_gain_probability_score(df)
    vol_regime = feature_volatility_regime(df) / 100.0  # Normalize from 0-100 to 0-1
    
    interaction = gain_prob * vol_regime
    interaction = interaction.clip(0.0, 1.0)
    interaction.name = "gain_probability_volatility_regime_interaction"
    return interaction


# ============================================================================
# BLOCK ML-27: Precision-Enhancing Features (5 features)
# ============================================================================

def feature_adr_percentage(df: DataFrame) -> Series:
    """
    ADR (Average Daily Range) Percentage: Normal intraday swing of a stock.
    
    Measures the "normal" intraday price movement, distinct from volatility (standard deviation).
    ADR provides a "Reality Check" for target expectations - if a stock normally moves 3% a day
    but is trying to hit a 15% target in a low-volatility environment, the math doesn't work.
    
    Calculated as: 14-period SMA of ((High - Low) / Close).
    Higher values = stock is more "active" (normal high intraday swings).
    Lower values = stock is less active (tighter intraday ranges).
    
    Already in percentage terms (from division by Close). No normalization needed.
    No clipping - let ML learn the distribution.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Daily range as percentage of close
    daily_range_pct = (high - low) / close
    
    # 14-period SMA of daily range percentage
    adr = daily_range_pct.rolling(window=14, min_periods=1).mean()
    
    adr.name = "adr_percentage"
    return adr


def feature_distance_from_monthly_vwap(df: DataFrame) -> Series:
    """
    Distance from Monthly VWAP: Distance from Volume Weighted Average Price over ~21 trading days.
    
    For a 20-30 day trading horizon, monthly VWAP is more "institutional" than 200-day SMA.
    Stocks trading significantly above their monthly VWAP are often "overextended" - the model
    may be buying late. This feature tells the model: "Yes, volatility is there, but the price
    is too far from the average entry price of other buyers."
    
    Calculated as: (Close / Monthly_VWAP) - 1.
    Positive values = trading above monthly VWAP (overextended).
    Negative values = trading below monthly VWAP (potential opportunity).
    Zero = trading at monthly VWAP.
    
    Monthly VWAP = sum(price * volume) / sum(volume) over ~21 trading days.
    No clipping - let ML learn the distribution.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Monthly VWAP over ~21 trading days (approximately 1 month)
    price_volume = close * volume
    monthly_vwap = price_volume.rolling(window=21, min_periods=1).sum() / volume.rolling(window=21, min_periods=1).sum()
    
    # Distance as percentage: (Close / Monthly_VWAP) - 1
    distance = (close / (monthly_vwap + 1e-10)) - 1.0
    
    distance.name = "distance_from_monthly_vwap"
    return distance


def feature_atr_channel_position(df: DataFrame) -> Series:
    """
    ATR Channel Position: Position within Keltner Channel (volatility envelope).
    
    Contextualizes volatility_forecast by showing where price is relative to volatility bands.
    If a stock is at the +3 ATR level, its probability of an immediate 15% further gain is low
    (it's overbought). If it's at the -1 ATR level and starting to turn, the 15% upside is
    much more likely. This will directly improve Precision.
    
    Calculated as: (Close - Keltner_Lower_Band) / (Keltner_Upper_Band - Keltner_Lower_Band).
    Values: 0.0 = at lower band, 0.5 = at middle (EMA), 1.0 = at upper band.
    Values > 1.0 = above upper band (overbought), < 0.0 = below lower band (oversold).
    
    Keltner Channel:
    - Mid-line: EMA(20)
    - Upper Band: EMA(20) + 1.5 * ATR(20)
    - Lower Band: EMA(20) - 1.5 * ATR(20)
    
    Normalized to [0, 1] range for typical positions, but allows >1 and <0 for extreme cases.
    Clipped to [-1, 2] for safety (allows some extreme values but prevents outliers).
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Mid-line: EMA(20)
    mid = close.ewm(span=20, adjust=False).mean()
    
    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR(20)
    atr = tr.rolling(window=20, min_periods=1).mean()
    
    # Keltner Channels
    upper_kc = mid + 1.5 * atr
    lower_kc = mid - 1.5 * atr
    
    # Channel position: (Close - Lower) / (Upper - Lower)
    channel_width = upper_kc - lower_kc
    position = (close - lower_kc) / (channel_width + 1e-10)
    
    # Clip to reasonable range but allow some extremes
    position = position.clip(-1.0, 2.0)
    position.name = "atr_channel_position"
    return position
