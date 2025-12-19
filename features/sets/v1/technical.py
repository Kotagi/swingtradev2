# src/features/technical.py

"""
technical.py

Defines feature-extraction functions for a financial dataset, using pandas-ta
for optimized indicator computations. Each function takes a DataFrame with
columns ['open', 'high', 'low', 'close', 'volume'] (case-insensitive) and
returns a pandas Series named for downstream pipeline use.
"""

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from pandas import Series, DataFrame
from typing import Optional
from sklearn.linear_model import LinearRegression
from pathlib import Path
import logging

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
    _trend_residual_window
)

logger = logging.getLogger(__name__)

def feature_log_return_1d(df: DataFrame) -> Series:
    """
    Compute 1-day log return: ln(close_t / close_{t-1}).

    Steps:
      1. Retrieve the closing price series (handles upper/lower case).
      2. Compute the ratio close_t / close_{t-1} and take the natural log.
      3. Name the resulting Series for downstream use.

    Args:
        df: Input DataFrame with a 'close' price column.

    Returns:
        Series named 'log_return_1d' of 1-day log returns.
    """
    close = _get_close_series(df)
    # log-return: ln(P_t / P_{t-1})
    lr = np.log(close / close.shift(1))
    lr.name = "log_return_1d"
    return lr

def feature_log_return_5d(df: DataFrame) -> Series:
    """
    Compute 5-day log return: ln(close_t / close_{t-5}).

    Steps:
      1. Retrieve closing price series (handles both 'close' and 'Close').
      2. Compute ratio close_t / close_{t-5} and take natural log.
      3. Name the resulting Series for downstream use.

    Args:
        df: Input DataFrame with a 'close' or 'Close' column.

    Returns:
        Series named 'log_return_5d' of 5-day log returns.
    """
    close = _get_close_series(df)
    lr5 = np.log(close / close.shift(5))
    lr5.name = "log_return_5d"
    return lr5


def feature_price(df: DataFrame) -> Series:
    """
    Return the current closing price (raw price value).
    
    This is the base price feature, useful for filtering by price ranges
    (e.g., $1-$5, >$5, >$10) similar to Finviz filters.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'price' containing closing prices.
    """
    close = _get_close_series(df)
    price = close.copy()
    price.name = "price"
    return price


def feature_price_log(df: DataFrame) -> Series:
    """
    Compute log price: ln(close).
    
    Log price squashes huge differences between high and low priced stocks,
    making it more suitable for ML models that need normalized inputs.
    Prefer this over raw price for ML applications.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'price_log' containing natural log of closing prices.
    """
    close = _get_close_series(df)
    price_log = np.log(close)
    price_log.name = "price_log"
    return price_log


def feature_price_vs_ma200(df: DataFrame) -> Series:
    """
    Compute price normalized relative to 200-day moving average: close / SMA(200).
    
    This normalizes price relative to a long-term baseline, making it comparable
    across different price ranges. Values > 1.0 indicate price above long-term average,
    values < 1.0 indicate price below long-term average.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'price_vs_ma200' containing price / SMA(200) ratios.
    """
    close = _get_close_series(df)
    sma200 = close.rolling(window=200, min_periods=1).mean()
    ratio = close / sma200
    ratio.name = "price_vs_ma200"
    return ratio


def feature_daily_return(df: DataFrame) -> Series:
    """
    Compute daily return as percentage: (close_t / close_{t-1} - 1) * 100.
    
    Normalized by clipping to [-0.2, 0.2] to cap extreme moves at ±20%.
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'daily_return' containing clipped daily percentage returns.
    """
    close = _get_close_series(df)
    daily_ret = close.pct_change()
    # Clip to ±20% to normalize extreme moves
    daily_ret = daily_ret.clip(-0.2, 0.2)
    daily_ret.name = "daily_return"
    return daily_ret


def feature_gap_pct(df: DataFrame) -> Series:
    """
    Compute gap percentage: (open_t - close_{t-1}) / close_{t-1}.
    
    This measures the gap between today's open and yesterday's close.
    Positive values indicate gap-up, negative values indicate gap-down.
    
    Normalized by clipping to [-0.2, 0.2] to cap extreme gaps at ±20%.
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with 'open' and 'close' columns.
    
    Returns:
        Series named 'gap_pct' containing clipped gap percentages.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    prev_close = close.shift(1)
    gap_pct = (openp - prev_close) / prev_close
    # Clip to ±20% to normalize extreme gaps
    gap_pct = gap_pct.clip(-0.2, 0.2)
    gap_pct.name = "gap_pct"
    return gap_pct


def feature_weekly_return_5d(df: DataFrame) -> Series:
    """
    Compute 5-day (weekly) return: close.pct_change(5).
    
    This measures the percentage return over 5 trading days (approximately one week).
    Calculated as: (close_t / close_{t-5} - 1).
    
    Normalized by clipping to [-0.3, 0.3] to cap extreme moves at ±30%.
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'weekly_return_5d' containing clipped 5-day percentage returns.
    """
    close = _get_close_series(df)
    ret_5 = close.pct_change(5)
    # Clip to ±30% to normalize extreme moves
    ret_5 = ret_5.clip(-0.3, 0.3)
    ret_5.name = "weekly_return_5d"
    return ret_5


def feature_monthly_return_21d(df: DataFrame) -> Series:
    """
    Compute 21-day (monthly) return: close.pct_change(21).
    
    This measures the percentage return over 21 trading days (approximately one month).
    Calculated as: (close_t / close_{t-21} - 1).
    
    Normalized by clipping to [-0.5, 0.5] to cap extreme moves at ±50%.
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'monthly_return_21d' containing clipped 21-day percentage returns.
    """
    close = _get_close_series(df)
    ret_21 = close.pct_change(21)
    # Clip to ±50% to normalize extreme moves
    ret_21 = ret_21.clip(-0.5, 0.5)
    ret_21.name = "monthly_return_21d"
    return ret_21


def feature_quarterly_return_63d(df: DataFrame) -> Series:
    """
    Compute 63-day (quarterly) return: close.pct_change(63).
    
    This measures the percentage return over 63 trading days (approximately one quarter).
    Calculated as: (close_t / close_{t-63} - 1).
    
    Normalized by clipping to [-1.0, 1.0] to cap extreme moves at ±100%.
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'quarterly_return_63d' containing clipped 63-day percentage returns.
    """
    close = _get_close_series(df)
    ret_63 = close.pct_change(63)
    # Clip to ±100% to normalize extreme moves
    ret_63 = ret_63.clip(-1.0, 1.0)
    ret_63.name = "quarterly_return_63d"
    return ret_63


def feature_ytd_return(df: DataFrame) -> Series:
    """
    Compute Year-to-Date (YTD) return: close / first_close_of_year - 1.
    
    This measures the percentage return from the first trading day of the year
    to the current date. Calculated as: close / close.groupby(year).transform('first') - 1.
    
    Normalized by clipping to [-1.0, 2.0] to cap extreme moves:
    - Minimum: -100% (total loss)
    - Maximum: +200% (triple the value)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column and DatetimeIndex.
    
    Returns:
        Series named 'ytd_return' containing clipped YTD percentage returns.
    """
    close = _get_close_series(df)
    
    # Ensure index is DatetimeIndex
    if not isinstance(close.index, pd.DatetimeIndex):
        close.index = pd.to_datetime(close.index)
    
    # Get first close price of each year
    first_close_of_year = close.groupby(close.index.year).transform('first')
    
    # Calculate YTD return: (current_close / first_close_of_year) - 1
    ytd = (close / first_close_of_year) - 1
    
    # Clip to (-1, +2) to normalize extreme moves
    ytd = ytd.clip(-1.0, 2.0)
    ytd.name = "ytd_return"
    return ytd


def feature_dist_52w_high(df: DataFrame) -> Series:
    """
    Compute 52-week high distance: (close / high_52w) - 1.
    
    This measures how far the current price is from the 52-week (252 trading days) high.
    Calculated as: close / close.rolling(252).max() - 1.
    
    Values:
    - 0.0: Price is at the 52-week high
    - Negative: Price is below the 52-week high (more negative = further below)
    - Positive: Price is above the 52-week high (rare, but possible with new highs)
    
    Normalized by clipping to [-1.0, 0.5] to cap extreme values:
    - Minimum: -100% (price is half of 52-week high)
    - Maximum: +50% (price is 1.5x the 52-week high)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'dist_52w_high' containing clipped 52-week high distance.
    """
    close = _get_close_series(df)
    
    # Calculate 52-week (252 trading days) high
    high_52 = close.rolling(window=252, min_periods=1).max()
    
    # Calculate distance: (current_price / 52w_high) - 1
    dist_52_high = (close / high_52) - 1
    
    # Clip to (-1, 0.5) to normalize extreme values
    dist_52_high = dist_52_high.clip(-1.0, 0.5)
    dist_52_high.name = "dist_52w_high"
    return dist_52_high


def feature_dist_52w_low(df: DataFrame) -> Series:
    """
    Compute 52-week low distance: (close / low_52w) - 1.
    
    This measures how far the current price is from the 52-week (252 trading days) low.
    Calculated as: close / close.rolling(252).min() - 1.
    
    Values:
    - 0.0: Price is at the 52-week low
    - Positive: Price is above the 52-week low (more positive = further above)
    - Negative: Price is below the 52-week low (rare, but possible with new lows)
    
    Normalized by clipping to [-0.5, 2.0] to cap extreme values:
    - Minimum: -50% (price is half of 52-week low)
    - Maximum: +200% (price is 3x the 52-week low)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'dist_52w_low' containing clipped 52-week low distance.
    """
    close = _get_close_series(df)
    
    # Calculate 52-week (252 trading days) low
    low_52 = close.rolling(window=252, min_periods=1).min()
    
    # Calculate distance: (current_price / 52w_low) - 1
    dist_52_low = (close / low_52) - 1
    
    # Clip to (-0.5, 2) to normalize extreme values
    dist_52_low = dist_52_low.clip(-0.5, 2.0)
    dist_52_low.name = "dist_52w_low"
    return dist_52_low


def feature_pos_52w(df: DataFrame) -> Series:
    """
    Compute 52-week position: (close - low_52) / (high_52 - low_52).
    
    This measures the normalized position of the current price within the 52-week range.
    Calculated as: (close - low_52) / (high_52 - low_52).
    
    Values:
    - 0.0: Price is at the 52-week low
    - 1.0: Price is at the 52-week high
    - 0.5: Price is at the midpoint of the 52-week range
    - Values between 0 and 1 represent the position within the range
    
    Normalized by clipping to [0.0, 1.0] to ensure values stay within bounds.
    This prevents division by zero errors and makes the feature suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'pos_52w' containing clipped 52-week position (0=low, 1=high).
    """
    close = _get_close_series(df)
    
    # Calculate 52-week (252 trading days) high and low
    high_52 = close.rolling(window=252, min_periods=1).max()
    low_52 = close.rolling(window=252, min_periods=1).min()
    
    # Calculate position: (current_price - low) / (high - low)
    range_52 = high_52 - low_52
    # Avoid division by zero (when high == low, position is 0.5 or use close position)
    pos_52 = (close - low_52) / range_52.replace(0, 1)  # Replace 0 with 1 to avoid division by zero
    
    # Clip to [0, 1] to normalize and handle edge cases
    pos_52 = pos_52.clip(0.0, 1.0)
    pos_52.name = "pos_52w"
    return pos_52


def feature_sma20_ratio(df: DataFrame) -> Series:
    """
    Compute SMA20 ratio: close / SMA(20).
    
    This measures the current price relative to the 20-day simple moving average.
    Calculated as: close / close.rolling(20).mean().
    
    Values:
    - 1.0: Price equals the SMA20
    - > 1.0: Price is above the SMA20 (bullish)
    - < 1.0: Price is below the SMA20 (bearish)
    
    Normalized by clipping to [0.5, 1.5] to cap extreme values:
    - Minimum: 0.5 (price is half of SMA20)
    - Maximum: 1.5 (price is 1.5x the SMA20)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'sma20_ratio' containing clipped SMA20 ratios.
    """
    close = _get_close_series(df)
    
    # Calculate 20-day SMA
    sma20 = close.rolling(window=20, min_periods=1).mean()
    
    # Calculate ratio: current_price / SMA20
    feat = close / sma20
    
    # Clip to [0.5, 1.5] to normalize extreme values
    feat = feat.clip(0.5, 1.5)
    feat.name = "sma20_ratio"
    return feat


def feature_sma50_ratio(df: DataFrame) -> Series:
    """
    Compute SMA50 ratio: close / SMA(50).
    
    This measures the current price relative to the 50-day simple moving average.
    Calculated as: close / close.rolling(50).mean().
    
    Values:
    - 1.0: Price equals the SMA50
    - > 1.0: Price is above the SMA50 (bullish)
    - < 1.0: Price is below the SMA50 (bearish)
    
    Normalized by clipping to [0.5, 1.5] to cap extreme values:
    - Minimum: 0.5 (price is half of SMA50)
    - Maximum: 1.5 (price is 1.5x the SMA50)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'sma50_ratio' containing clipped SMA50 ratios.
    """
    close = _get_close_series(df)
    
    # Calculate 50-day SMA
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Calculate ratio: current_price / SMA50
    feat = close / sma50
    
    # Clip to [0.5, 1.5] to normalize extreme values
    feat = feat.clip(0.5, 1.5)
    feat.name = "sma50_ratio"
    return feat


def feature_sma200_ratio(df: DataFrame) -> Series:
    """
    Compute SMA200 ratio: close / SMA(200).
    
    This measures the current price relative to the 200-day simple moving average.
    Calculated as: close / close.rolling(200).mean().
    
    Values:
    - 1.0: Price equals the SMA200
    - > 1.0: Price is above the SMA200 (bullish, long-term uptrend)
    - < 1.0: Price is below the SMA200 (bearish, long-term downtrend)
    
    Normalized by clipping to [0.5, 2.0] to cap extreme values:
    - Minimum: 0.5 (price is half of SMA200)
    - Maximum: 2.0 (price is 2x the SMA200)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'sma200_ratio' containing clipped SMA200 ratios.
    """
    close = _get_close_series(df)
    
    # Calculate 200-day SMA
    sma200 = close.rolling(window=200, min_periods=1).mean()
    
    # Calculate ratio: current_price / SMA200
    feat = close / sma200
    
    # Clip to [0.5, 2.0] to normalize extreme values
    feat = feat.clip(0.5, 2.0)
    feat.name = "sma200_ratio"
    return feat


def feature_sma20_sma50_ratio(df: DataFrame) -> Series:
    """
    Compute SMA20/SMA50 ratio: SMA(20) / SMA(50).
    
    This measures the relationship between short-term (20-day) and medium-term (50-day)
    moving averages. It's a moving average crossover indicator.
    
    Values:
    - 1.0: SMA20 equals SMA50 (neutral)
    - > 1.0: SMA20 above SMA50 (bullish crossover, uptrend)
    - < 1.0: SMA20 below SMA50 (bearish crossover, downtrend)
    
    Normalized by clipping to [0.8, 1.2] to cap extreme values:
    - Minimum: 0.8 (SMA20 is 80% of SMA50)
    - Maximum: 1.2 (SMA20 is 120% of SMA50)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'sma20_sma50_ratio' containing clipped SMA20/SMA50 ratios.
    """
    close = _get_close_series(df)
    
    # Calculate 20-day and 50-day SMAs
    sma20 = close.rolling(window=20, min_periods=1).mean()
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Calculate ratio: SMA20 / SMA50
    feat = sma20 / sma50
    
    # Clip to [0.8, 1.2] to normalize extreme values
    feat = feat.clip(0.8, 1.2)
    feat.name = "sma20_sma50_ratio"
    return feat


def feature_sma50_sma200_ratio(df: DataFrame) -> Series:
    """
    Compute SMA50/SMA200 ratio: SMA(50) / SMA(200).
    
    This measures the relationship between medium-term (50-day) and long-term (200-day)
    moving averages. It's a classic moving average crossover indicator (Golden Cross/Death Cross).
    
    Values:
    - 1.0: SMA50 equals SMA200 (neutral)
    - > 1.0: SMA50 above SMA200 (Golden Cross, bullish long-term trend)
    - < 1.0: SMA50 below SMA200 (Death Cross, bearish long-term trend)
    
    Normalized by clipping to [0.6, 1.4] to cap extreme values:
    - Minimum: 0.6 (SMA50 is 60% of SMA200)
    - Maximum: 1.4 (SMA50 is 140% of SMA200)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'sma50_sma200_ratio' containing clipped SMA50/SMA200 ratios.
    """
    close = _get_close_series(df)
    
    # Calculate 50-day and 200-day SMAs
    sma50 = close.rolling(window=50, min_periods=1).mean()
    sma200 = close.rolling(window=200, min_periods=1).mean()
    
    # Calculate ratio: SMA50 / SMA200
    feat = sma50 / sma200
    
    # Clip to [0.6, 1.4] to normalize extreme values
    feat = feat.clip(0.6, 1.4)
    feat.name = "sma50_sma200_ratio"
    return feat


def feature_sma50_slope(df: DataFrame) -> Series:
    """
    Compute SMA50 slope: sma50.diff(5) / close.
    
    This measures the 5-day change in the 50-day moving average, normalized by the current price.
    It indicates the rate of change (slope) of the medium-term trend.
    
    Values:
    - 0.0: SMA50 is flat (no change over 5 days)
    - > 0.0: SMA50 is rising (bullish momentum)
    - < 0.0: SMA50 is falling (bearish momentum)
    
    Normalized by clipping to [-0.1, 0.1] to cap extreme values:
    - Minimum: -0.1 (SMA50 falling by 10% of price over 5 days)
    - Maximum: +0.1 (SMA50 rising by 10% of price over 5 days)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'sma50_slope' containing clipped SMA50 slope values.
    """
    close = _get_close_series(df)
    
    # Calculate 50-day SMA
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    # Calculate 5-day change in SMA50, normalized by current price
    feat = sma50.diff(5) / close
    
    # Clip to [-0.1, 0.1] to normalize extreme values
    feat = feat.clip(-0.1, 0.1)
    feat.name = "sma50_slope"
    return feat


def feature_sma200_slope(df: DataFrame) -> Series:
    """
    Compute SMA200 slope: sma200.diff(10) / close.
    
    This measures the 10-day change in the 200-day moving average, normalized by the current price.
    It indicates the rate of change (slope) of the long-term trend.
    
    Values:
    - 0.0: SMA200 is flat (no change over 10 days)
    - > 0.0: SMA200 is rising (bullish long-term momentum)
    - < 0.0: SMA200 is falling (bearish long-term momentum)
    
    Normalized by clipping to [-0.1, 0.1] to cap extreme values:
    - Minimum: -0.1 (SMA200 falling by 10% of price over 10 days)
    - Maximum: +0.1 (SMA200 rising by 10% of price over 10 days)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'sma200_slope' containing clipped SMA200 slope values.
    """
    close = _get_close_series(df)
    
    # Calculate 200-day SMA
    sma200 = close.rolling(window=200, min_periods=1).mean()
    
    # Calculate 10-day change in SMA200, normalized by current price
    feat = sma200.diff(10) / close
    
    # Clip to [-0.1, 0.1] to normalize extreme values
    feat = feat.clip(-0.1, 0.1)
    feat.name = "sma200_slope"
    return feat


def feature_volatility_5d(df: DataFrame) -> Series:
    """
    Compute 5-day volatility: close.pct_change().rolling(5).std().
    
    This measures the standard deviation of daily returns over a 5-day rolling window.
    Higher values indicate more volatile price movements.
    
    Normalized by clipping to [0.0, 0.15] to cap extreme values:
    - Minimum: 0.0 (no volatility)
    - Maximum: 0.15 (15% daily volatility)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'volatility_5d' containing clipped 5-day volatility values.
    """
    close = _get_close_series(df)
    
    # Calculate 5-day rolling standard deviation of daily returns
    vol_5 = close.pct_change().rolling(window=5, min_periods=1).std()
    
    # Clip to [0, 0.15] to normalize extreme values
    vol_5 = vol_5.clip(0.0, 0.15)
    vol_5.name = "volatility_5d"
    return vol_5


def feature_volatility_21d(df: DataFrame) -> Series:
    """
    Compute 21-day volatility: close.pct_change().rolling(21).std().
    
    This measures the standard deviation of daily returns over a 21-day rolling window.
    Higher values indicate more volatile price movements over the medium term.
    
    Normalized by clipping to [0.0, 0.15] to cap extreme values:
    - Minimum: 0.0 (no volatility)
    - Maximum: 0.15 (15% daily volatility)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'volatility_21d' containing clipped 21-day volatility values.
    """
    close = _get_close_series(df)
    
    # Calculate 21-day rolling standard deviation of daily returns
    vol_21 = close.pct_change().rolling(window=21, min_periods=1).std()
    
    # Clip to [0, 0.15] to normalize extreme values
    vol_21 = vol_21.clip(0.0, 0.15)
    vol_21.name = "volatility_21d"
    return vol_21


def feature_volatility_ratio(df: DataFrame) -> Series:
    """
    Compute Volatility Ratio: short-term volatility (5-day) vs long-term volatility (21-day).
    
    The ratio of short-term volatility to long-term volatility identifies volatility
    expansion/compression regimes more cleanly than ATR alone.
    
    This is extremely important for predicting swing duration and follow-through.
    
    Calculation:
    1. vol5 = volatility_5d (already computed)
    2. vol21 = volatility_21d (already computed)
    3. volatility_ratio = vol5 / vol21
    4. Normalize by clipping to [0, 2]
    
    Normalized by clipping to [0, 2]:
    - > 1: Volatility expanding (short-term vol > long-term vol)
    - < 1: Volatility contracting (short-term vol < long-term vol)
    - ≈ 1: Stable regime (short-term vol ≈ long-term vol)
    - Range: [0, 2]
    - Identifies volatility expansion/compression regimes
    
    Why it adds value:
    - Captures regime shifts
    - Improves breakout & pullback predictions
    - Helps differentiate choppy vs trending markets
    - More cleanly identifies volatility expansion/compression than ATR alone
    - Extremely important for predicting swing duration and follow-through
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'volatility_ratio' containing normalized volatility ratio values.
    """
    # Get existing volatility features
    vol5 = feature_volatility_5d(df)
    vol21 = feature_volatility_21d(df)
    
    # Step 1: Calculate volatility ratio
    # volatility_ratio = vol5 / vol21
    # Handle division by zero (when vol21 is 0)
    volatility_ratio = vol5 / (vol21 + 1e-10)
    
    # Step 2: Normalize by clipping to [0, 2]
    volatility_ratio = volatility_ratio.clip(0.0, 2.0)
    
    volatility_ratio.name = "volatility_ratio"
    return volatility_ratio


def feature_atr14_normalized(df: DataFrame) -> Series:
    """
    Compute normalized ATR14: ATR(14) / close.
    
    This measures the Average True Range over 14 days, normalized by the current price.
    ATR measures volatility based on the true range (high-low, high-prev_close, low-prev_close).
    
    True Range calculation:
    - TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    - ATR14 = rolling mean of TR over 14 days
    
    Normalized by clipping to [0.0, 0.2] to cap extreme values:
    - Minimum: 0.0 (no volatility)
    - Maximum: 0.2 (ATR is 20% of price)
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with 'high', 'low', and 'close' columns.
    
    Returns:
        Series named 'atr14_normalized' containing clipped normalized ATR14 values.
    """
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
    feat = atr14 / close
    
    # Clip to [0, 0.2] to normalize extreme values
    feat = feat.clip(0.0, 0.2)
    feat.name = "atr14_normalized"
    return feat


def feature_log_volume(df: DataFrame) -> Series:
    """
    Compute log volume: np.log1p(volume).
    
    This transforms volume using the natural logarithm of (1 + volume).
    The log1p function is used to handle zero volumes gracefully and compress
    the wide range of volume values into a more manageable scale.
    
    Already normalized by the log transformation, which naturally compresses
    large values and handles the wide range of volume data.
    
    Args:
        df: Input DataFrame with a 'volume' column.
    
    Returns:
        Series named 'log_volume' containing log-transformed volume values.
    """
    volume = _get_volume_series(df)
    
    # Apply log1p transformation (log(1 + volume))
    feat = np.log1p(volume)
    feat.name = "log_volume"
    return feat


def feature_log_avg_volume_20d(df: DataFrame) -> Series:
    """
    Compute log average volume (20-day): np.log1p(volume.rolling(20).mean()).
    
    This transforms the 20-day rolling average of volume using the natural logarithm.
    It provides a smoothed, normalized view of volume trends over the medium term.
    
    Already normalized by the log transformation, which naturally compresses
    large values and handles the wide range of volume data.
    
    Args:
        df: Input DataFrame with a 'volume' column.
    
    Returns:
        Series named 'log_avg_volume_20d' containing log-transformed 20-day average volume.
    """
    volume = _get_volume_series(df)
    
    # Calculate 20-day rolling average volume
    vol_avg20 = volume.rolling(window=20, min_periods=1).mean()
    
    # Apply log1p transformation
    feat = np.log1p(vol_avg20)
    feat.name = "log_avg_volume_20d"
    return feat


def feature_relative_volume(df: DataFrame) -> Series:
    """
    Compute relative volume: np.log1p((volume / vol_avg20).clip(0, 10)).
    
    This measures current volume relative to the 20-day average volume.
    Values > 1.0 indicate above-average volume, values < 1.0 indicate below-average volume.
    
    Normalized by:
    1. Clipping the ratio to [0, 10] to cap extreme values
    2. Applying log1p transformation to compress the scale
    
    This prevents outliers from dominating the feature and makes it more
    suitable for ML models.
    
    Args:
        df: Input DataFrame with a 'volume' column.
    
    Returns:
        Series named 'relative_volume' containing log-transformed relative volume values.
    """
    volume = _get_volume_series(df)
    
    # Calculate 20-day rolling average volume
    vol_avg20 = volume.rolling(window=20, min_periods=1).mean()
    
    # Calculate relative volume: current volume / average volume
    rvol = volume / vol_avg20
    
    # Clip to [0, 10] to normalize extreme values
    rvol = rvol.clip(0, 10)
    
    # Apply log1p transformation to compress scale
    feat = np.log1p(rvol)
    feat.name = "relative_volume"
    return feat


def feature_rsi14(df: DataFrame) -> Series:
    """
    Compute RSI14 (Relative Strength Index) with centered normalization.
    
    RSI calculation:
    1. Calculate price change: delta = close.diff()
    2. Separate gains and losses:
       - gain = delta.clip(lower=0)  (positive changes)
       - loss = -delta.clip(upper=0)  (negative changes, made positive)
    3. Calculate 14-day averages:
       - avg_gain = gain.rolling(14).mean()
       - avg_loss = loss.rolling(14).mean()
    4. Calculate relative strength: rs = avg_gain / avg_loss
    5. Calculate RSI: rsi = 100 - (100 / (1 + rs))
    
    Normalized (centered) by: (rsi - 50) / 50
    This transforms RSI from [0, 100] range to [-1, +1] range:
    - -1.0: RSI = 0 (extremely oversold)
    - 0.0: RSI = 50 (neutral)
    - +1.0: RSI = 100 (extremely overbought)
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'rsi14' containing centered RSI14 values in [-1, +1] range.
    """
    close = _get_close_series(df)
    
    # Calculate price change
    delta = close.diff()
    
    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate 14-day rolling averages
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI (0-100 range)
    rsi = 100 - (100 / (1 + rs))
    
    # Normalize (center) to [-1, +1] range
    feat = (rsi - 50) / 50
    
    feat.name = "rsi14"
    return feat


def feature_candle_body_pct(df: DataFrame) -> Series:
    """
    Compute candle body percentage: body / range.
    
    This measures the size of the candle body relative to the total candle range.
    Calculated as: abs(close - open) / (high - low).
    
    Values are already in [0, 1] range:
    - 0.0: No body (doji - open equals close)
    - 1.0: Full body (no wicks - body equals range)
    
    Args:
        df: Input DataFrame with 'open', 'high', 'low', and 'close' columns.
    
    Returns:
        Series named 'candle_body_pct' containing body percentage values in [0, 1] range.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate body (absolute difference between open and close)
    body = (close - openp).abs()
    
    # Calculate range (high - low), replace 0 with NaN to avoid division by zero
    range_ = (high - low).replace(0, np.nan)
    
    # Calculate body percentage
    body_pct = body / range_
    
    body_pct.name = "candle_body_pct"
    return body_pct


def feature_candle_upper_wick_pct(df: DataFrame) -> Series:
    """
    Compute upper wick percentage: upper_wick / range.
    
    This measures the size of the upper wick relative to the total candle range.
    Calculated as: (high - max(close, open)) / (high - low).
    
    Values are already in [0, 1] range:
    - 0.0: No upper wick (high equals max(close, open))
    - 1.0: Full upper wick (entire range is upper wick)
    
    Args:
        df: Input DataFrame with 'open', 'high', 'low', and 'close' columns.
    
    Returns:
        Series named 'candle_upper_wick_pct' containing upper wick percentage values in [0, 1] range.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate upper wick: high - max(close, open)
    # Using close.where(close >= openp, openp) gives max(close, open)
    upper = high - close.where(close >= openp, openp)
    
    # Calculate range (high - low), replace 0 with NaN to avoid division by zero
    range_ = (high - low).replace(0, np.nan)
    
    # Calculate upper wick percentage
    upper_pct = upper / range_
    
    upper_pct.name = "candle_upper_wick_pct"
    return upper_pct


def feature_candle_lower_wick_pct(df: DataFrame) -> Series:
    """
    Compute lower wick percentage: lower_wick / range.
    
    This measures the size of the lower wick relative to the total candle range.
    Calculated as: (min(close, open) - low) / (high - low).
    
    Values are already in [0, 1] range:
    - 0.0: No lower wick (low equals min(close, open))
    - 1.0: Full lower wick (entire range is lower wick)
    
    Args:
        df: Input DataFrame with 'open', 'high', 'low', and 'close' columns.
    
    Returns:
        Series named 'candle_lower_wick_pct' containing lower wick percentage values in [0, 1] range.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Calculate lower wick: min(close, open) - low
    # Using close.where(close <= openp, openp) gives min(close, open)
    lower = close.where(close <= openp, openp) - low
    
    # Calculate range (high - low), replace 0 with NaN to avoid division by zero
    range_ = (high - low).replace(0, np.nan)
    
    # Calculate lower wick percentage
    lower_pct = lower / range_
    
    lower_pct.name = "candle_lower_wick_pct"
    return lower_pct


def feature_higher_high_10d(df: DataFrame) -> Series:
    """
    Compute higher high (10-day) binary flag.
    
    This indicates if the current close is higher than the maximum close
    of the previous 10 days. A higher high suggests bullish momentum.
    
    Calculated as: (close > close.shift(1).rolling(10).max()).astype(int)
    
    Values are binary (0/1):
    - 0: Current close is NOT higher than previous 10-day max
    - 1: Current close IS higher than previous 10-day max (higher high)
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'higher_high_10d' containing binary values (0 or 1).
    """
    close = _get_close_series(df)
    
    # Calculate max of previous 10 days (excluding current day)
    prev_10d_max = close.shift(1).rolling(window=10, min_periods=1).max()
    
    # Check if current close is higher than previous 10-day max
    hh = (close > prev_10d_max).astype(int)
    
    hh.name = "higher_high_10d"
    return hh


def feature_higher_low_10d(df: DataFrame) -> Series:
    """
    Compute higher low (10-day) binary flag.
    
    This indicates if the current close is higher than the minimum close
    of the previous 10 days. A higher low suggests bullish momentum and
    potential trend continuation.
    
    Calculated as: (close > close.shift(1).rolling(10).min()).astype(int)
    
    Values are binary (0/1):
    - 0: Current close is NOT higher than previous 10-day min
    - 1: Current close IS higher than previous 10-day min (higher low)
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'higher_low_10d' containing binary values (0 or 1).
    """
    close = _get_close_series(df)
    
    # Calculate min of previous 10 days (excluding current day)
    prev_10d_min = close.shift(1).rolling(window=10, min_periods=1).min()
    
    # Check if current close is higher than previous 10-day min
    hl = (close > prev_10d_min).astype(int)
    
    hl.name = "higher_low_10d"
    return hl


def feature_swing_low_10d(df: DataFrame) -> Series:
    """
    Compute recent swing low (10-day): the lowest low price over the previous 10 days.
    
    This feature identifies the most recent structural support level (swing low)
    which can be used for stop-loss placement. The swing low represents the
    lowest price point in the recent price action, indicating a support level.
    
    Calculated as: low.shift(1).rolling(10, min_periods=1).min()
    - Excludes current day to avoid lookahead bias
    - Uses previous 10 days of low prices
    
    This uses the 'low' price (not close) to capture the actual swing low point.
    
    Values:
    - Returns the actual swing low price (not normalized)
    - Used in conjunction with entry price to calculate stop distance
    - Lower values indicate stronger support levels
    
    Args:
        df: Input DataFrame with 'low' price column.
    
    Returns:
        Series named 'swing_low_10d' containing the swing low price values.
    """
    low = _get_low_series(df)
    
    # Calculate the minimum low over the previous 10 days (excluding current day)
    # shift(1) excludes current day to avoid lookahead bias
    swing_low = low.shift(1).rolling(window=10, min_periods=1).min()
    
    swing_low.name = "swing_low_10d"
    return swing_low


def feature_trend_residual(df: DataFrame) -> Series:
    """
    Compute trend residual (noise vs trend) using linear regression.
    
    This feature measures how much the current price deviates from a linear
    trend fitted over the previous 50 days. A positive residual indicates
    the price is above the trend line, while negative indicates below.
    
    Calculated by:
    1. Fitting linear regression to last 50 close values
    2. Calculating residual: (actual - fitted) / actual
    3. Taking the last residual value
    4. Clipping to [-0.2, 0.2]
    
    Values are clipped to [-0.2, 0.2]:
    - Negative: Price below trend (potential oversold)
    - Positive: Price above trend (potential overbought)
    - Near 0: Price follows trend closely
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'trend_residual' containing normalized residual values.
    """
    close = _get_close_series(df)
    
    # Apply rolling window regression (50 days)
    # Using rolling.apply with the helper function
    resid = close.rolling(window=50, min_periods=50).apply(
        _trend_residual_window, raw=True
    )
    
    # Clip to [-0.2, 0.2]
    resid = resid.clip(-0.2, 0.2)
    
    resid.name = "trend_residual"
    return resid


# Ownership features removed - not consistent and cannot be trusted


def feature_macd_histogram_normalized(df: DataFrame) -> Series:
    """
    Compute MACD Histogram (normalized by price): (macd_line - signal_line) / close.
    
    MACD Histogram measures momentum acceleration vs. deceleration. The histogram
    is the most predictive part of MACD — it shows the strength of the trend and
    detects early momentum shifts.
    
    Calculation:
    1. EMA(close, 12) - fast exponential moving average
    2. EMA(close, 26) - slow exponential moving average
    3. MACD line = EMA12 - EMA26
    4. Signal line = EMA(MACD line, 9)
    5. Histogram = MACD line - Signal line
    6. Normalized = Histogram / close
    
    Normalization by price makes the feature scale-independent, allowing ML models
    to compare values across different price ranges. This keeps everything in a
    similar range regardless of stock price.
    
    Why it's valuable:
    - Captures trend momentum
    - Detects divergence
    - Identifies turning points
    - Shows acceleration vs deceleration
    - Much more expressive than RSI alone
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'macd_histogram_normalized' containing normalized MACD histogram values.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate EMAs
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    
    # Step 2: Calculate MACD line
    macd_line = ema12 - ema26
    
    # Step 3: Calculate signal line (EMA of MACD line)
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    
    # Step 4: Calculate histogram
    macd_hist = macd_line - signal_line
    
    # Step 5: Normalize by price
    macd_hist_norm = macd_hist / close
    
    macd_hist_norm.name = "macd_histogram_normalized"
    return macd_hist_norm


def feature_ppo_histogram(df: DataFrame) -> Series:
    """
    Compute PPO (Percentage Price Oscillator) Histogram: percentage-based momentum acceleration/deceleration.
    
    PPO measures momentum in percent, making it:
    - Cross-ticker comparable
    - Scale-invariant
    - More stable across expensive vs cheap stocks
    
    PPO histogram = acceleration/deceleration of percentage momentum.
    
    Calculation:
    1. Calculate two EMAs:
       - ema12 = EMA(close, 12)
       - ema26 = EMA(close, 26)
    2. Calculate PPO line:
       - ppo = (ema12 - ema26) / ema26
    3. Calculate PPO signal:
       - ppo_signal = EMA(ppo, 9)
    4. Calculate PPO histogram:
       - ppo_hist = ppo - ppo_signal
    5. Normalize by clipping to [-0.2, 0.2]
    
    Normalized by clipping to [-0.2, 0.2]:
    - Positive: PPO accelerating above signal (bullish momentum acceleration)
    - Negative: PPO decelerating below signal (bearish momentum deceleration)
    - Near 0: PPO and signal converging (momentum neutral)
    - Range: [-0.2, 0.2] (typically ranges around [-0.1, 0.1])
    - Scale-invariant and cross-ticker comparable
    
    Why it adds value:
    - Less redundant with MACD than you'd think (percentage vs absolute)
    - Captures percent-based momentum cleanly
    - Often improves ML performance with multi-ticker training
    - More stable across expensive vs cheap stocks
    - Cross-ticker comparable (unlike MACD which is in price units)
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'ppo_histogram' containing normalized PPO histogram values.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate two EMAs
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    
    # Step 2: Calculate PPO line
    # PPO = (ema12 - ema26) / ema26
    ppo = (ema12 - ema26) / ema26
    
    # Step 3: Calculate PPO signal
    # ppo_signal = EMA(ppo, 9)
    ppo_signal = ppo.ewm(span=9, adjust=False).mean()
    
    # Step 4: Calculate PPO histogram
    # ppo_hist = ppo - ppo_signal
    ppo_hist = ppo - ppo_signal
    
    # Step 5: Normalize by clipping to [-0.2, 0.2]
    # It typically ranges around [-0.1, 0.1], so clip to [-0.2, 0.2] for safety
    ppo_hist_norm = ppo_hist.clip(-0.2, 0.2)
    
    ppo_hist_norm.name = "ppo_histogram"
    return ppo_hist_norm


def feature_dpo(df: DataFrame, period: int = 20) -> Series:
    """
    Compute DPO (Detrended Price Oscillator): cyclical indicator that removes long-term trend.
    
    DPO removes long-term trend and highlights short-term price cycles, helping ML detect:
    - Cycle peaks
    - Cycle troughs
    - Trend pullbacks
    - Mean-reversion zones
    
    Perfect for 10-30 day windows.
    
    Calculation:
    1. Calculate centered SMA:
       - period = 20
       - sma = close.rolling(period).mean()
       - shifted_sma = sma.shift(period//2 + 1)
    2. Calculate DPO:
       - dpo = close - shifted_sma
    3. Normalize by closing price:
       - dpo_norm = dpo / close
    
    Normalized by dividing by closing price (dpo / close):
    - Positive: Price above detrended average (cycle peak, overextended)
    - Negative: Price below detrended average (cycle trough, compressed)
    - Near 0: Price at detrended average (neutral)
    - Range: unbounded, but typically small values
    - Highlights short-term price cycles
    
    Why it adds value:
    - Gives the model cycle structure, which no other feature gives
    - Helps detect "overextended" or "compressed" prices
    - Complements CCI & Williams %R
    - Removes long-term trend to focus on cyclical patterns
    - Perfect for identifying mean-reversion zones
    
    Args:
        df: Input DataFrame with 'close' column.
        period: Period for SMA calculation (default 20).
    
    Returns:
        Series named 'dpo' containing normalized DPO values.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate centered SMA
    # period = 20
    sma = close.rolling(window=period, min_periods=period).mean()
    
    # Shift the SMA by period//2 + 1 to center it
    # For period = 20: shift = 20//2 + 1 = 10 + 1 = 11
    shift_amount = period // 2 + 1
    shifted_sma = sma.shift(shift_amount)
    
    # Step 2: Calculate DPO
    # dpo = close - shifted_sma
    dpo = close - shifted_sma
    
    # Step 3: Normalize by closing price
    # dpo_norm = dpo / close
    dpo_norm = dpo / close
    
    # Clip to reasonable range to handle extreme values
    dpo_norm = dpo_norm.clip(-0.2, 0.2)
    
    dpo_norm.name = "dpo"
    return dpo_norm


def feature_roc10(df: DataFrame) -> Series:
    """
    Compute ROC (Rate of Change) 10-period: short-term momentum velocity indicator.
    
    ROC measures percentage velocity of price movement.
    ROC10 captures short-term momentum.
    
    Different from log returns because ROC captures velocity, not simple percent change.
    
    Calculation:
    1. roc10 = (close - close.shift(10)) / close.shift(10)
    2. Normalize by clipping to [-0.5, 0.5]
    
    Normalized by clipping to [-0.5, 0.5]:
    - Positive: Price rising over 10 periods (bullish momentum)
    - Negative: Price falling over 10 periods (bearish momentum)
    - Near 0: Price relatively stable (neutral momentum)
    - Range: [-0.5, 0.5] (±50% change over 10 periods)
    - Standardized and directional momentum indicator
    
    Why it adds value:
    - Highly predictive in breakouts and pullbacks
    - A more expressive form of momentum than basic log returns
    - ROC10 + ROC20 = excellent short/medium-term momentum pair
    - Captures velocity, not just simple percent change
    - Standardized momentum indicator (unlike returns)
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'roc10' containing normalized ROC10 values.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate ROC10
    # roc10 = (close - close.shift(10)) / close.shift(10)
    roc10 = (close - close.shift(10)) / close.shift(10)
    
    # Step 2: Normalize by clipping to [-0.5, 0.5]
    roc10 = roc10.clip(-0.5, 0.5)
    
    roc10.name = "roc10"
    return roc10


def feature_roc20(df: DataFrame) -> Series:
    """
    Compute ROC (Rate of Change) 20-period: medium-term momentum velocity indicator.
    
    ROC measures percentage velocity of price movement.
    ROC20 captures medium-term momentum.
    
    Different from log returns because ROC captures velocity, not simple percent change.
    
    Calculation:
    1. roc20 = (close - close.shift(20)) / close.shift(20)
    2. Normalize by clipping to [-0.7, 0.7]
    
    Normalized by clipping to [-0.7, 0.7]:
    - Positive: Price rising over 20 periods (bullish momentum)
    - Negative: Price falling over 20 periods (bearish momentum)
    - Near 0: Price relatively stable (neutral momentum)
    - Range: [-0.7, 0.7] (±70% change over 20 periods)
    - Standardized and directional momentum indicator
    
    Why it adds value:
    - Highly predictive in breakouts and pullbacks
    - A more expressive form of momentum than basic log returns
    - ROC10 + ROC20 = excellent short/medium-term momentum pair
    - Captures velocity, not just simple percent change
    - Standardized momentum indicator (unlike returns)
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'roc20' containing normalized ROC20 values.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate ROC20
    # roc20 = (close - close.shift(20)) / close.shift(20)
    roc20 = (close - close.shift(20)) / close.shift(20)
    
    # Step 2: Normalize by clipping to [-0.7, 0.7]
    roc20 = roc20.clip(-0.7, 0.7)
    
    roc20.name = "roc20"
    return roc20


def feature_stochastic_k14(df: DataFrame) -> Series:
    """
    Compute Stochastic Oscillator %K (14-period): (close - low_14) / (high_14 - low_14).
    
    Stochastic %K measures where price sits within the recent trading range.
    Unlike RSI, it directly captures overbought/oversold relative to range.
    
    Calculation:
    1. low_14 = lowest low over last 14 bars
    2. high_14 = highest high over last 14 bars
    3. %K = (close - low_14) / (high_14 - low_14)
    
    This gives a 0 → 1 value:
    - 0.0: Close at the lowest low (extremely oversold)
    - 0.5: Close in the middle of the range (neutral)
    - 1.0: Close at the highest high (extremely overbought)
    
    Already normalized (0-1 range), so no additional normalization needed.
    
    Why it's valuable:
    - Better than RSI in many trend scenarios
    - Normalized across stocks
    - Captures range compression & exhaustion
    - Helps detect early reversals & continuation setups
    
    Args:
        df: Input DataFrame with 'close', 'high', and 'low' columns.
    
    Returns:
        Series named 'stochastic_k14' containing Stochastic %K values in [0, 1] range.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Step 1: Calculate rolling high/low over 14 periods
    low_14 = low.rolling(window=14, min_periods=1).min()
    high_14 = high.rolling(window=14, min_periods=1).max()
    
    # Step 2: Calculate Stochastic %K
    # Handle division by zero (when high_14 == low_14, meaning no range)
    range_14 = high_14 - low_14
    range_14 = range_14.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
    
    stoch_k = (close - low_14) / range_14
    
    # Clip to [0, 1] to handle any edge cases
    stoch_k = stoch_k.clip(0.0, 1.0)
    
    stoch_k.name = "stochastic_k14"
    return stoch_k


def feature_bollinger_band_width(df: DataFrame) -> Series:
    """
    Compute Bollinger Band Width (log normalized): log1p((upper_band - lower_band) / mid_band).
    
    Bollinger Band Width measures how "tight" or "expanded" price is relative to its
    recent volatility. BB Width is one of the best predictors of breakouts and volatility
    expansions.
    
    Calculation:
    1. Middle band = SMA(close, 20)
    2. Standard deviation = rolling_std(close, 20)
    3. Upper band = mid + 2 * std
    4. Lower band = mid - 2 * std
    5. BB Width = (upper - lower) / mid
    6. Normalized with log1p: bbw_log = log1p(bbw)
    
    Log normalization (log1p) works well because BB width can be very small or large,
    and log1p compresses the range while handling small values gracefully.
    
    Why it's valuable:
    - Captures volatility squeezes
    - Predicts breakout probability
    - Identifies trend exhaustion
    - Detects range tightening
    - Signals market regime shifts
    - BB squeezes often precede strong 10-30 day swings (exactly what swing trading targets)
    
    Args:
        df: Input DataFrame with a 'close' price column.
    
    Returns:
        Series named 'bollinger_band_width' containing log-normalized Bollinger Band Width values.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate middle band (SMA20)
    mid = close.rolling(window=20, min_periods=1).mean()
    
    # Step 2: Calculate standard deviation (20-period)
    std = close.rolling(window=20, min_periods=1).std()
    
    # Step 3: Calculate upper and lower bands
    upper = mid + 2 * std
    lower = mid - 2 * std
    
    # Step 4: Calculate Bollinger Band Width
    # Handle division by zero (when mid is 0, which shouldn't happen but safety first)
    mid_safe = mid.replace(0, np.nan)
    bbw = (upper - lower) / mid_safe
    
    # Step 5: Log normalization
    bbw_log = np.log1p(bbw)
    
    bbw_log.name = "bollinger_band_width"
    return bbw_log


def feature_adx14(df: DataFrame) -> Series:
    """
    Compute ADX (Average Directional Index) 14-period, normalized to [0, 1] range.
    
    ADX measures trend strength, independent of direction. It tells whether the stock is:
    - Trending strongly
    - Ranging
    - Losing momentum
    - Entering trend continuation
    
    This is one of the most predictive free indicators.
    
    Calculation (14-day ADX):
    1. True Range & Directional Movement:
       - +DM = today's high - yesterday high (if positive and > -DM)
       - -DM = yesterday low - today's low (if positive and > +DM)
       - TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    2. Smooth 14-day averages of +DM, -DM, and TR
    3. DI lines:
       - +DI = 100 * (+DM14 / TR14)
       - -DI = 100 * (-DM14 / TR14)
    4. DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    5. ADX = EMA(DX, 14)
    6. Normalize: adx_norm = adx / 100
    
    Normalized to [0, 1] range:
    - 0.0: No trend (ranging market)
    - 0.25: Weak trend
    - 0.50: Moderate trend
    - 0.75: Strong trend
    - 1.0: Very strong trend
    
    Why it adds value:
    - Model currently knows trend direction (via slopes and HH/HL)
    - But it does NOT know how strong the trend is
    - ADX fills that gap perfectly
    
    Args:
        df: Input DataFrame with 'high', 'low', and 'close' columns.
    
    Returns:
        Series named 'adx14' containing normalized ADX values in [0, 1] range.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Step 1: Calculate True Range and Directional Movement
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # True Range: max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high - prev_high
    minus_dm = prev_low - low
    
    # +DM: today's high - yesterday high (if positive and > -DM)
    # -DM: yesterday low - today's low (if positive and > +DM)
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
    
    # Step 2: Smooth 14-day averages (using Wilder's smoothing method)
    # Wilder's smoothing: first value is sum, then use: smoothed = prev_smoothed * (n-1)/n + current * 1/n
    # For simplicity, we'll use rolling mean with min_periods=1 for initial values
    # Then apply Wilder's smoothing formula
    def wilder_smooth(series, period=14):
        """Apply Wilder's smoothing method."""
        smoothed = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i == 0:
                smoothed.iloc[i] = series.iloc[i]
            elif i < period:
                # For first period values, use simple average
                smoothed.iloc[i] = series.iloc[:i+1].mean()
            else:
                # Wilder's smoothing: prev * (n-1)/n + current * 1/n
                smoothed.iloc[i] = smoothed.iloc[i-1] * (period - 1) / period + series.iloc[i] / period
        return smoothed
    
    plus_dm14 = wilder_smooth(plus_dm, period=14)
    minus_dm14 = wilder_smooth(minus_dm, period=14)
    tr14 = wilder_smooth(tr, period=14)
    
    # Step 3: Calculate DI lines
    # Handle division by zero
    tr14_safe = tr14.replace(0, np.nan)
    plus_di = 100 * (plus_dm14 / tr14_safe)
    minus_di = 100 * (minus_dm14 / tr14_safe)
    
    # Step 4: Calculate DX
    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum_safe
    
    # Step 5: Calculate ADX (smoothed DX with EMA)
    adx = dx.ewm(span=14, adjust=False).mean()
    
    # Step 6: Normalize to [0, 1] range
    adx_norm = adx / 100
    
    # Clip to [0, 1] to handle any edge cases
    adx_norm = adx_norm.clip(0.0, 1.0)
    
    adx_norm.name = "adx14"
    return adx_norm


def feature_chaikin_money_flow(df: DataFrame) -> Series:
    """
    Compute Chaikin Money Flow (CMF) 20-period: sum(mfv over 20d) / sum(volume over 20d).
    
    CMF detects accumulation vs distribution, combining price movement WITH volume.
    - Positive CMF → buying pressure / accumulation
    - Negative CMF → selling pressure / distribution
    
    Volume-price flow is one of the strongest predictors of future swings.
    
    Calculation:
    1. Money Flow Multiplier:
       mfm = ((close - low) - (high - close)) / (high - low)
       This simplifies to: mfm = (2*close - high - low) / (high - low)
    2. Money Flow Volume:
       mfv = mfm * volume
    3. Chaikin Money Flow (20-day):
       cmf = sum(mfv over 20d) / sum(volume over 20d)
    4. Normalize: clip to [-1, 1]
    
    Already normalized between -1 and +1:
    - -1.0: Strong selling pressure / distribution
    - 0.0: Neutral (balanced buying/selling)
    - +1.0: Strong buying pressure / accumulation
    
    Why it adds value:
    - CMF exposes quiet accumulation before breakouts
    - Combines price action with volume (stronger signal than price alone)
    - Detects institutional accumulation/distribution
    - One of the strongest predictors of future swings
    
    Args:
        df: Input DataFrame with 'close', 'high', 'low', and 'volume' columns.
    
    Returns:
        Series named 'chaikin_money_flow' containing CMF values in [-1, 1] range.
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    # Step 1: Calculate Money Flow Multiplier
    # mfm = ((close - low) - (high - close)) / (high - low)
    # Simplifies to: mfm = (2*close - high - low) / (high - low)
    range_hl = high - low
    range_hl = range_hl.replace(0, np.nan)  # Handle division by zero
    
    mfm = (2 * close - high - low) / range_hl
    
    # Step 2: Calculate Money Flow Volume
    mfv = mfm * volume
    
    # Step 3: Calculate Chaikin Money Flow (20-day)
    # cmf = sum(mfv over 20d) / sum(volume over 20d)
    mfv_sum_20d = mfv.rolling(window=20, min_periods=1).sum()
    volume_sum_20d = volume.rolling(window=20, min_periods=1).sum()
    
    # Handle division by zero
    volume_sum_20d_safe = volume_sum_20d.replace(0, np.nan)
    cmf = mfv_sum_20d / volume_sum_20d_safe
    
    # Step 4: Clip to [-1, 1] range
    cmf = cmf.clip(-1.0, 1.0)
    
    cmf.name = "chaikin_money_flow"
    return cmf


def feature_donchian_position(df: DataFrame) -> Series:
    """
    Compute Donchian Channel position: (close - donchian_low_20) / (donchian_high_20 - donchian_low_20).
    
    Donchian channels measure breakout levels. Markets trend when breaking out of established ranges.
    This feature measures the normalized position of price within the 20-period Donchian channel.
    
    Calculation:
    1. donchian_high_20 = rolling_max(high, 20)
    2. donchian_low_20 = rolling_min(low, 20)
    3. donchian_position = (close - donchian_low_20) / (donchian_high_20 - donchian_low_20)
    
    Already normalized to [0, 1] range:
    - 0.0: Close at the lowest low (at lower channel)
    - 0.5: Close in the middle of the channel
    - 1.0: Close at the highest high (at upper channel)
    - Values > 1.0: Breakout above upper channel (clipped to 1.0)
    - Values < 0.0: Breakdown below lower channel (clipped to 0.0)
    
    Why it adds value:
    - Model captures trend shape, but not breakout structure
    - Donchian provides a clean, ML-friendly breakout signal
    - Measures position within established trading range
    - Identifies when price is near breakout levels
    
    Args:
        df: Input DataFrame with 'close', 'high', and 'low' columns.
    
    Returns:
        Series named 'donchian_position' containing normalized position within Donchian channel [0, 1].
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    
    # Step 1: Calculate Donchian channels (20-period)
    donchian_high_20 = high.rolling(window=20, min_periods=1).max()
    donchian_low_20 = low.rolling(window=20, min_periods=1).min()
    
    # Step 2: Calculate position within channel
    # Handle division by zero (when high == low, meaning no range)
    channel_range = donchian_high_20 - donchian_low_20
    channel_range = channel_range.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
    
    donchian_position = (close - donchian_low_20) / channel_range
    
    # Clip to [0, 1] to handle breakouts/breakdowns
    donchian_position = donchian_position.clip(0.0, 1.0)
    
    donchian_position.name = "donchian_position"
    return donchian_position


def feature_donchian_breakout(df: DataFrame) -> Series:
    """
    Compute Donchian Channel breakout: binary flag indicating if close > prior 20-day high close.
    
    Donchian channels measure breakout levels. Markets trend when breaking out of established ranges.
    This feature identifies when price breaks above the prior 20-day highest close (non-lookahead).
    
    Calculation (non-lookahead):
    1. prior_20d_high_close = close.rolling(20, min_periods=20).max().shift(1)
    2. donchian_breakout = (close > prior_20d_high_close).astype(int)
    
    This uses the prior 20-day highest CLOSE (not high), shifted by 1 bar to avoid lookahead bias.
    The shift ensures we only use information available before the current bar.
    
    Binary flag (0 or 1):
    - 0: Close is NOT above prior 20-day high close (no breakout)
    - 1: Close IS above prior 20-day high close (breakout detected)
    
    Why it adds value:
    - Model captures trend shape, but not breakout structure
    - Donchian provides a clean, ML-friendly breakout signal
    - Breakouts often precede strong trending moves
    - Binary signal is easy for ML models to interpret
    - Non-lookahead implementation ensures no data leakage
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'donchian_breakout' containing binary values (0 or 1).
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate prior 20-day highest close, shifted by 1 bar (non-lookahead)
    # This uses only information available before the current bar
    prior_20d_high_close = close.rolling(window=20, min_periods=20).max().shift(1)
    
    # Step 2: Check if current close is above prior 20-day high close
    donchian_breakout = (close > prior_20d_high_close).astype(int)
    
    # Set NaN values (from shift or insufficient data) to 0 (no breakout)
    donchian_breakout = donchian_breakout.fillna(0).astype(int)
    
    donchian_breakout.name = "donchian_breakout"
    return donchian_breakout


def feature_ttm_squeeze_on(df: DataFrame) -> Series:
    """
    Compute TTM Squeeze condition: binary flag indicating volatility contraction (squeeze).
    
    TTM Squeeze is the industry's favorite breakout/tight-range indicator. It detects
    volatility contraction ("squeeze") by comparing Bollinger Bands to Keltner Channels.
    
    Calculation:
    1. Bollinger Bands:
       - mid = SMA(close, 20)
       - std = rolling_std(close, 20)
       - upper_bb = mid + 2*std
       - lower_bb = mid - 2*std
    2. Keltner Channels:
       - atr = ATR(20)
       - upper_kc = mid + 1.5*atr
       - lower_kc = mid - 1.5*atr
    3. Squeeze condition:
       - squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    
    Binary flag (0 or 1):
    - 0: No squeeze (Bollinger Bands wider than Keltner Channels)
    - 1: Squeeze active (Bollinger Bands inside Keltner Channels - volatility contraction)
    
    Why it adds value:
    - This is the #1 breakout indicator used by quant retail traders
    - Catches explosive moves after volatility compression
    - Identifies tight trading ranges that often precede breakouts
    - Combines volatility analysis with momentum direction
    
    Args:
        df: Input DataFrame with 'close', 'high', and 'low' columns.
    
    Returns:
        Series named 'ttm_squeeze_on' containing binary values (0 or 1).
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
    squeeze_on = ((lower_bb > lower_kc) & (upper_bb < upper_kc)).astype(int)
    
    squeeze_on.name = "ttm_squeeze_on"
    return squeeze_on


def feature_ttm_squeeze_momentum(df: DataFrame) -> Series:
    """
    Compute TTM Squeeze momentum (normalized): (close - SMA20) / close.
    
    TTM Squeeze momentum provides direction during squeeze conditions. It's a simple
    momentum proxy that measures how far price is from the 20-day moving average,
    normalized by price.
    
    Calculation:
    1. mid = SMA(close, 20)
    2. squeeze_momentum = close - mid
    3. squeeze_momentum_norm = squeeze_momentum / close
    
    Normalized by price:
    - Positive: Price above SMA20 (bullish momentum)
    - Negative: Price below SMA20 (bearish momentum)
    - Near 0: Price at SMA20 (neutral)
    
    Why it adds value:
    - Provides momentum direction during squeeze conditions
    - Helps identify which direction the breakout is likely to occur
    - Normalized by price makes it comparable across different price ranges
    - Works in combination with squeeze_on to identify high-probability setups
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'ttm_squeeze_momentum' containing normalized momentum values.
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


def feature_obv_momentum(df: DataFrame) -> Series:
    """
    Compute OBV Momentum (OBV Rate of Change): 10-day percentage change of On-Balance Volume.
    
    On-Balance Volume shows cumulative volume moving with price. Its rate of change (ROC)
    captures whether volume is accelerating into the trend. Big funds often move volume
    before price shifts — OBV ROC catches that.
    
    Calculation:
    1. Build OBV (On-Balance Volume):
       - Close up → add volume
       - Close down → subtract volume
       - Equal → no change
    2. Calculate 10-day percentage change of OBV:
       - obv_roc = OBV.pct_change(10)
    3. Normalize by clipping to [-0.5, 0.5]
    
    Normalized by clipping to [-0.5, 0.5] to cap extreme values:
    - Positive: Volume accelerating upward (bullish)
    - Negative: Volume accelerating downward (bearish)
    - Near 0: Volume momentum neutral
    - Range: [-0.5, 0.5] (±50% change)
    
    Why it adds value:
    - Gives volume acceleration, not just volume level
    - Works extremely well with breakouts and volatility squeezes
    - One of the highest-impact free indicators you can add
    - Catches institutional volume movements before price shifts
    
    Args:
        df: Input DataFrame with 'close' and 'volume' columns.
    
    Returns:
        Series named 'obv_momentum' containing clipped OBV rate of change values.
    """
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    # Step 1: Calculate OBV (On-Balance Volume)
    # OBV rules:
    # - If close > prev_close: add volume
    # - If close < prev_close: subtract volume
    # - If close == prev_close: no change (add 0)
    
    # Calculate price change direction
    price_change = close.diff()
    
    # Create signed volume: positive for up days, negative for down days, zero for equal
    signed_volume = volume.copy()
    signed_volume = signed_volume.where(price_change > 0, -signed_volume)  # Down days: negative
    signed_volume = signed_volume.where(price_change != 0, 0)  # Equal days: zero
    
    # Calculate OBV as cumulative sum
    obv = signed_volume.cumsum()
    
    # Step 2: Calculate 10-day percentage change of OBV
    obv_roc = obv.pct_change(10)
    
    # Step 3: Normalize by clipping to [-0.5, 0.5]
    obv_roc = obv_roc.clip(-0.5, 0.5)
    
    obv_roc.name = "obv_momentum"
    return obv_roc


def feature_aroon_up(df: DataFrame) -> Series:
    """
    Compute Aroon Up (25-period): normalized measure of days since highest high.
    
    Aroon measures how many days since the last highest high (Aroon Up) or lowest low (Aroon Down).
    It tells the model whether the uptrend is fresh, maturing, or exhausted.
    
    Calculation (25-period window):
    1. Find days since highest high in rolling 25-period window
    2. Aroon Up = 100 * (25 - days_since_highest_high) / 25
    3. Normalize: aroon_up_norm = aroon_up / 100
    
    Normalized to [0, 1] range:
    - 1.0: Highest high was today (fresh uptrend)
    - 0.8: Highest high was 5 days ago
    - 0.0: Highest high was 25+ days ago (exhausted uptrend)
    - Range: [0.0, 1.0]
    
    Why it adds value:
    - Model currently knows if trend exists and how strong it is (ADX)
    - But Aroon tells it how long the trend has been going on
    - Trend age is often where swings succeed or fail
    - Identifies if uptrend is fresh, maturing, or exhausted
    
    Args:
        df: Input DataFrame with 'high' column.
    
    Returns:
        Series named 'aroon_up' containing normalized Aroon Up values in [0, 1] range.
    """
    high = _get_high_series(df)
    
    # Step 1: Find days since highest high in each rolling 25-period window
    # In rolling().apply(), window_values[0] is oldest, window_values[-1] is newest
    def days_since_max(window_values):
        """Calculate days since highest high in the window (0 = today, 24 = 24 days ago)."""
        if len(window_values) == 0 or np.isnan(window_values).all():
            return np.nan
        
        # Find the index of the maximum value, preferring most recent if tie
        # Reverse array to search from newest to oldest, then convert back to original index
        reversed_idx = np.argmax(window_values[::-1])
        max_idx_original = len(window_values) - 1 - reversed_idx
        
        # Days since = distance from newest value (index len-1) to max_idx
        days_since = len(window_values) - 1 - max_idx_original
        return days_since
    
    # Calculate days since highest high for each rolling window
    days_since_high = high.rolling(window=25, min_periods=25).apply(
        days_since_max, raw=True
    )
    
    # Step 2: Calculate Aroon Up
    # Aroon Up = 100 * (25 - days_since_highest_high) / 25
    aroon_up = 100 * (25 - days_since_high) / 25
    
    # Step 3: Normalize by dividing by 100
    aroon_up_norm = aroon_up / 100
    
    # Clip to [0, 1] to handle any edge cases
    aroon_up_norm = aroon_up_norm.clip(0.0, 1.0)
    
    aroon_up_norm.name = "aroon_up"
    return aroon_up_norm


def feature_aroon_down(df: DataFrame) -> Series:
    """
    Compute Aroon Down (25-period): normalized measure of days since lowest low.
    
    Aroon measures how many days since the last highest high (Aroon Up) or lowest low (Aroon Down).
    It tells the model whether the downtrend is starting or ending.
    
    Calculation (25-period window):
    1. Find days since lowest low in rolling 25-period window
    2. Aroon Down = 100 * (25 - days_since_lowest_low) / 25
    3. Normalize: aroon_down_norm = aroon_down / 100
    
    Normalized to [0, 1] range:
    - 1.0: Lowest low was today (fresh downtrend)
    - 0.8: Lowest low was 5 days ago
    - 0.0: Lowest low was 25+ days ago (exhausted downtrend)
    - Range: [0.0, 1.0]
    
    Why it adds value:
    - Model currently knows if trend exists and how strong it is (ADX)
    - But Aroon tells it how long the trend has been going on
    - Trend age is often where swings succeed or fail
    - Identifies if downtrend is starting or ending
    
    Args:
        df: Input DataFrame with 'low' column.
    
    Returns:
        Series named 'aroon_down' containing normalized Aroon Down values in [0, 1] range.
    """
    low = _get_low_series(df)
    
    # Step 1: Find days since lowest low in each rolling 25-period window
    # In rolling().apply(), window_values[0] is oldest, window_values[-1] is newest
    def days_since_min(window_values):
        """Calculate days since lowest low in the window (0 = today, 24 = 24 days ago)."""
        if len(window_values) == 0 or np.isnan(window_values).all():
            return np.nan
        
        # Find the index of the minimum value, preferring most recent if tie
        # Reverse array to search from newest to oldest, then convert back to original index
        reversed_idx = np.argmin(window_values[::-1])
        min_idx_original = len(window_values) - 1 - reversed_idx
        
        # Days since = distance from newest value (index len-1) to min_idx
        days_since = len(window_values) - 1 - min_idx_original
        return days_since
    
    # Calculate days since lowest low for each rolling window
    days_since_low = low.rolling(window=25, min_periods=25).apply(
        days_since_min, raw=True
    )
    
    # Step 2: Calculate Aroon Down
    # Aroon Down = 100 * (25 - days_since_lowest_low) / 25
    aroon_down = 100 * (25 - days_since_low) / 25
    
    # Step 3: Normalize by dividing by 100
    aroon_down_norm = aroon_down / 100
    
    # Clip to [0, 1] to handle any edge cases
    aroon_down_norm = aroon_down_norm.clip(0.0, 1.0)
    
    aroon_down_norm.name = "aroon_down"
    return aroon_down_norm


def feature_aroon_oscillator(df: DataFrame) -> Series:
    """
    Compute Aroon Oscillator: trend dominance indicator combining Aroon Up and Aroon Down.
    
    Aroon Oscillator tells the model which side is in control:
    - Positive → uptrend dominance
    - Negative → downtrend dominance
    - Near zero → trend transition
    
    It's the "net trend pressure" measure Aroon is famous for.
    
    Calculation:
    1. Get normalized Aroon Up and Aroon Down (0-1 range)
    2. Convert back to 0-100 range:
       - aroon_up_raw = aroon_up * 100
       - aroon_down_raw = aroon_down * 100
    3. Calculate oscillator: aroon_osc = aroon_up_raw - aroon_down_raw
    4. Normalize from [-100, 100] to [0, 1]:
       - aroon_osc_norm = (aroon_osc + 100) / 200
    
    Normalized to [0, 1] range:
    - 0.0: Strong downtrend dominance (aroon_osc = -100)
    - 0.5: Neutral/transition (aroon_osc = 0)
    - 1.0: Strong uptrend dominance (aroon_osc = +100)
    - Range: [0.0, 1.0]
    - Provides clean continuous signal for ML
    
    Why it adds value:
    - Captures trend dominance better than Up and Down alone
    - Provides a clean continuous signal for ML
    - Helps identify early trend reversals
    - Combines both Aroon lines into single "net trend pressure" measure
    
    Args:
        df: Input DataFrame with 'high' and 'low' columns.
    
    Returns:
        Series named 'aroon_oscillator' containing normalized Aroon Oscillator values in [0, 1] range.
    """
    # Get normalized Aroon Up and Aroon Down (0-1 range)
    aroon_up = feature_aroon_up(df)
    aroon_down = feature_aroon_down(df)
    
    # Step 1: Convert back to 0-100 range
    aroon_up_raw = aroon_up * 100
    aroon_down_raw = aroon_down * 100
    
    # Step 2: Calculate oscillator: aroon_osc = aroon_up_raw - aroon_down_raw
    # Range: [-100, 100]
    aroon_osc = aroon_up_raw - aroon_down_raw
    
    # Step 3: Normalize from [-100, 100] to [0, 1]
    # aroon_osc_norm = (aroon_osc + 100) / 200
    aroon_osc_norm = (aroon_osc + 100) / 200
    
    # Clip to [0, 1] to handle any edge cases
    aroon_osc_norm = aroon_osc_norm.clip(0.0, 1.0)
    
    aroon_osc_norm.name = "aroon_oscillator"
    return aroon_osc_norm


def feature_cci20(df: DataFrame) -> Series:
    """
    Compute CCI (Commodity Channel Index, 20-period): standardized distance from trend oscillator.
    
    CCI measures how far price deviates from its typical mean relative to volatility.
    It's a hybrid between RSI, momentum, and volatility.
    
    Calculation (20-period):
    1. Typical Price = (high + low + close) / 3
    2. SMA of Typical Price (20-period)
    3. Mean Deviation (20-period): mean absolute deviation from SMA
    4. CCI = (TP - SMA) / (0.015 * mean_deviation)
    5. Normalize: cci_norm = tanh(cci / 100)
    
    Normalized using tanh compression to [0, 1] range:
    - High CCI (>100): momentum burst, overbought
    - Low CCI (<-100): selling pressure, oversold
    - Near 0: price near typical mean
    - Range: approximately [-1, 1] after tanh, but typically [-0.76, 0.76] for CCI in [-100, 100]
    
    Why it adds value:
    - Model lacks a standardized "distance from trend" oscillator
    - CCI captures trend exhaustion & reversion points
    - Great for swing trading windows
    - Adds information RSI & Stochastic do NOT cover
    - Hybrid indicator combining momentum, volatility, and mean reversion
    
    Args:
        df: Input DataFrame with 'high', 'low', and 'close' columns.
    
    Returns:
        Series named 'cci20' containing normalized CCI values.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Step 1: Calculate Typical Price
    typical_price = (high + low + close) / 3
    
    # Step 2: Calculate SMA of Typical Price (20-period)
    sma_tp = typical_price.rolling(window=20, min_periods=20).mean()
    
    # Step 3: Calculate Mean Deviation (20-period)
    # Mean deviation = mean absolute deviation from SMA
    mean_deviation = typical_price.rolling(window=20, min_periods=20).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    
    # Step 4: Calculate CCI
    # CCI = (TP - SMA) / (0.015 * mean_deviation)
    # Avoid division by zero
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)
    
    # Step 5: Normalize using tanh compression
    # cci_norm = tanh(cci / 100)
    cci_norm = np.tanh(cci / 100)
    
    cci_norm.name = "cci20"
    return cci_norm


def feature_williams_r14(df: DataFrame) -> Series:
    """
    Compute Williams %R (14-period): range momentum/reversion oscillator.
    
    Williams %R measures how close price is to the recent lowest lows.
    Where Stochastic goes 0 → 1, %R goes -100 → 0.
    
    Calculation (14-period):
    1. highest_high = high.rolling(14).max()
    2. lowest_low = low.rolling(14).min()
    3. williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    4. Normalize: williams_r_norm = -(williams_r / 100)
    
    Normalized to [0, 1] range:
    - 0.0: Close at highest high (extremely overbought)
    - 0.5: Close in middle of range (neutral)
    - 1.0: Close at lowest low (extremely oversold)
    - Range: [0.0, 1.0]
    - Very sensitive to reversal points
    
    Why it adds value:
    - Very sensitive to reversal points
    - Strong complement to RSI and Stochastic
    - Helps catch swing entries inside trends
    - Detects pullbacks within trends, oversold bounces, momentum shifts in range markets
    - Provides "pressure" version of range position (complement to Stochastic %K)
    
    Args:
        df: Input DataFrame with 'high', 'low', and 'close' columns.
    
    Returns:
        Series named 'williams_r14' containing normalized Williams %R values in [0, 1] range.
    """
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    
    # Step 1: Calculate highest high and lowest low over 14-period window
    highest_high = high.rolling(window=14, min_periods=14).max()
    lowest_low = low.rolling(window=14, min_periods=14).min()
    
    # Step 2: Calculate Williams %R
    # %R = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    price_range = highest_high - lowest_low
    williams_r = ((highest_high - close) / (price_range + 1e-10)) * -100
    
    # Step 3: Normalize from [-100, 0] to [0, 1]
    # williams_r_norm = -(williams_r / 100)
    williams_r_norm = -(williams_r / 100)
    
    # Clip to [0, 1] to handle any edge cases
    williams_r_norm = williams_r_norm.clip(0.0, 1.0)
    
    williams_r_norm.name = "williams_r14"
    return williams_r_norm


def feature_kama_slope(df: DataFrame, period: int = 10, fast: int = 2, slow: int = 30) -> Series:
    """
    Compute KAMA (Kaufman Adaptive Moving Average) Slope: adaptive trend strength indicator.
    
    KAMA adapts to price smoothness:
    - Flat & slow when market is noisy
    - Fast & responsive when trend is efficient
    
    The slope measures the day-to-day change in KAMA, normalized by price.
    
    Calculation:
    1. Efficiency Ratio (ER) = net price movement / total movement
       - Change = abs(close - close[period periods ago])
       - Volatility = sum(abs(close.diff()) for period periods)
       - ER = Change / Volatility
    2. Smoothing Constant (SC) = [ER * (fast_SC - slow_SC) + slow_SC]^2
       - fast_SC = 2 / (fast + 1)
       - slow_SC = 2 / (slow + 1)
    3. KAMA = KAMA[prev] + SC * (close - KAMA[prev])
       - Initial KAMA = close (or SMA for first period)
    4. KAMA Slope = kama.diff() / close
    
    Normalized by dividing by price (scale-invariant):
    - 0.0: KAMA is flat (no change)
    - > 0.0: KAMA is rising (adaptive bullish momentum)
    - < 0.0: KAMA is falling (adaptive bearish momentum)
    - Range: unbounded, but typically small values
    
    Why it adds value:
    - SMA slopes measure linear trend
    - KAMA slope measures adaptive trend strength
    - Works better in choppy tickers
    - Adapts to market efficiency (fast in trends, slow in noise)
    - More responsive than SMA in efficient markets, less whipsaw in choppy markets
    
    Args:
        df: Input DataFrame with 'close' column.
        period: Period for efficiency ratio calculation (default 10).
        fast: Fast smoothing period (default 2).
        slow: Slow smoothing period (default 30).
    
    Returns:
        Series named 'kama_slope' containing normalized KAMA slope values.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate Efficiency Ratio (ER)
    # Change = absolute price change over period
    change = abs(close - close.shift(period))
    
    # Volatility = sum of absolute daily changes over period
    volatility = close.diff().abs().rolling(window=period, min_periods=period).sum()
    
    # ER = Change / Volatility (handle division by zero)
    er = change / (volatility + 1e-10)
    er = er.clip(0.0, 1.0)  # ER should be in [0, 1]
    
    # Step 2: Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    # SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Step 3: Calculate KAMA
    # Initialize KAMA with SMA for first period, then use recursive formula
    kama = close.rolling(window=period, min_periods=period).mean()
    
    # Calculate KAMA iteratively using recursive formula
    # KAMA[t] = KAMA[t-1] + SC[t] * (close[t] - KAMA[t-1])
    # This is more efficient than full vectorization due to recursive nature
    kama_values = kama.values
    sc_values = sc.values
    close_values = close.values
    
    for i in range(period, len(close)):
        if pd.isna(kama_values[i-1]) or pd.isna(sc_values[i]) or pd.isna(close_values[i]):
            continue
        kama_values[i] = kama_values[i-1] + sc_values[i] * (close_values[i] - kama_values[i-1])
    
    kama = pd.Series(kama_values, index=close.index)
    
    # Step 4: Calculate KAMA Slope (day-to-day change normalized by price)
    kama_slope = kama.diff() / close
    
    # Clip to reasonable range to handle extreme values
    kama_slope = kama_slope.clip(-0.1, 0.1)
    
    kama_slope.name = "kama_slope"
    return kama_slope


def feature_beta_spy_252d(df: DataFrame, spy_data: DataFrame = None) -> Series:
    """
    Compute rolling beta vs SPY over 252 trading days.
    
    Beta measures the stock's sensitivity to market movements (SPY).
    Calculation:
    1. Calculate log returns for stock and SPY:
       - stock_ret = np.log(close / close.shift(1))
       - spy_ret = np.log(spy_close / spy_close.shift(1))
    2. Calculate rolling covariance and variance over 252 days:
       - cov = stock_ret.rolling(252).cov(spy_ret)
       - var = spy_ret.rolling(252).var()
    3. Calculate beta: beta = cov / var
    
    Normalized by: ((beta + 1) / 4).clip(0, 1)
    This transforms beta to [0, 1] range:
    - beta = -1 → normalized = 0
    - beta = 0 → normalized = 0.25
    - beta = 1 → normalized = 0.5
    - beta = 3 → normalized = 1
    
    Args:
        df: Input DataFrame with a 'close' price column.
        spy_data: SPY DataFrame (optional, will be loaded if not provided)
    
    Returns:
        Series named 'beta_spy_252d' containing normalized beta values in [0, 1] range.
    """
    close = _get_close_series(df)
    
    # Load SPY data if not provided
    if spy_data is None:
        spy_data = _load_spy_data()
        if spy_data is None:
            # Return NaN if SPY data not available
            result = pd.Series(np.nan, index=df.index)
            result.name = "beta_spy_252d"
            return result
    
    # Get SPY close price - try different column name variations
    if 'Close' in spy_data.columns:
        spy_close = spy_data['Close']
    elif 'close' in spy_data.columns:
        spy_close = spy_data['close']
    elif 'Adj Close' in spy_data.columns:
        spy_close = spy_data['Adj Close']
    else:
        # Try to get first numeric column
        numeric_cols = spy_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            spy_close = spy_data[numeric_cols[0]]
        else:
            result = pd.Series(np.nan, index=df.index)
            result.name = "beta_spy_252d"
            return result
    
    if spy_close is None or len(spy_close) == 0:
        result = pd.Series(np.nan, index=df.index)
        result.name = "beta_spy_252d"
        return result
    
    # Ensure SPY index is DatetimeIndex and sorted
    if not isinstance(spy_close.index, pd.DatetimeIndex):
        spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = spy_close.sort_index()
    
    # Get stock date index
    if isinstance(df.index, pd.DatetimeIndex):
        stock_dates = df.index
    elif df.index.name == 'date' or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
        stock_dates = pd.to_datetime(df.index)
    elif 'date' in df.columns:
        stock_dates = pd.to_datetime(df['date'])
    else:
        result = pd.Series(np.nan, index=df.index)
        result.name = "beta_spy_252d"
        return result
    
    # Normalize dates to date-only for matching
    stock_dates_normalized = pd.to_datetime(stock_dates).normalize()
    spy_dates_normalized = pd.to_datetime(spy_close.index).normalize()
    
    # Create temporary DataFrames for alignment
    stock_temp = pd.DataFrame({
        'date': stock_dates_normalized,
        'idx': range(len(stock_dates_normalized)),
        'close': close.values
    })
    spy_temp = pd.DataFrame({
        'date': spy_dates_normalized,
        'close': spy_close.values
    }).sort_values('date')
    
    # Merge on date (left join to preserve stock dates)
    merged = stock_temp.merge(spy_temp, on='date', how='left', suffixes=('_stock', '_spy'), sort=False)
    merged = merged.sort_values('idx').reset_index(drop=True)
    
    # Forward fill missing SPY values
    merged['close_spy'] = merged['close_spy'].ffill().infer_objects(copy=False)
    
    # Calculate log returns
    stock_ret = np.log(merged['close_stock'] / merged['close_stock'].shift(1))
    spy_ret = np.log(merged['close_spy'] / merged['close_spy'].shift(1))
    
    # Create aligned Series with same index for rolling operations
    stock_ret_series = pd.Series(stock_ret.values, index=df.index)
    spy_ret_series = pd.Series(spy_ret.values, index=df.index)
    
    # Calculate rolling covariance and variance over 252 days
    # Use pandas rolling covariance (requires both series in a DataFrame)
    returns_df = pd.DataFrame({
        'stock_ret': stock_ret_series,
        'spy_ret': spy_ret_series
    })
    
    # Calculate rolling covariance: rolling(252).cov() between the two series
    # Note: rolling().cov() returns a DataFrame, we need the off-diagonal element
    # For efficiency, calculate manually using rolling mean and variance
    stock_mean = returns_df['stock_ret'].rolling(window=252, min_periods=1).mean()
    spy_mean = returns_df['spy_ret'].rolling(window=252, min_periods=1).mean()
    
    # Calculate rolling covariance: E[(X - E[X])(Y - E[Y])]
    # Using the formula: E[XY] - E[X]E[Y]
    stock_spy_product = (returns_df['stock_ret'] * returns_df['spy_ret']).rolling(window=252, min_periods=1).mean()
    cov = stock_spy_product - (stock_mean * spy_mean)
    
    # Calculate rolling variance of SPY returns
    var = returns_df['spy_ret'].rolling(window=252, min_periods=1).var()
    
    # Calculate beta: cov / var
    # Avoid division by zero
    var = var.replace(0, np.nan)
    beta = cov / var
    
    # Normalize: ((beta + 1) / 4).clip(0, 1)
    # This maps beta from [-1, 3] to [0, 1]
    feat = ((beta + 1) / 4).clip(0, 1)
    feat.name = "beta_spy_252d"
    return feat


def feature_fractal_dimension_index(df: DataFrame) -> Series:
    """
    Compute Fractal Dimension Index (FDI) - measures how "rough" the price path is.
    
    Fractal Dimension measures the complexity/roughness of the price path:
    - FDI ≈ 1.0-1.3 → smooth, trending (trend-friendly environment)
    - FDI ≈ 1.5 → borderline
    - FDI ≈ 1.6-1.8 → choppy, mean-reverting, noisy (whipsaw environment)
    
    This tells the model whether it's in a trend-friendly vs whipsaw environment,
    helping it downweight momentum signals in very noisy regimes.
    
    Calculation (sliding window approach, 100-bar window):
    1. For each window of N=100 prices:
       - Compute net displacement: L_net = |P_N-1 - P_0|
       - Compute path length: L_path = sum(|P_i - P_i-1|) for i=1 to N-1
       - Roughness ratio: R = L_path / (L_net + ε)
       - Fractal dimension: FDI = 1 + log(R+1) / log(N)
    
    Normalization:
    - FDI for financial time series typically lives in [1.0, 1.8]
    - Normalize to [0, 1]: FDI_norm = clip((FDI - 1.0) / (1.8 - 1.0), 0, 1)
    
    Why it adds value:
    - Directly encodes "is this tradable with trend-following or not"
    - Helps model downweight momentum signals in very noisy regimes
    - Pairs beautifully with ADX, Aroon, Donchian, TTM squeeze
    - Very few retail systems use it – it's a genuine edge-type feature
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'fractal_dimension_index' containing normalized FDI values in [0, 1].
    """
    close = _get_close_series(df)
    
    # Window size for fractal dimension calculation
    window = 100
    
    # Initialize result series
    fdi = pd.Series(index=close.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Calculate FDI for each rolling window
    for i in range(len(close)):
        if i < window - 1:
            # Not enough data for full window
            fdi.iloc[i] = np.nan
            continue
        
        # Get window of prices
        window_prices = close.iloc[i - window + 1:i + 1].values
        
        # Skip if any NaN values in window
        if np.any(np.isnan(window_prices)):
            fdi.iloc[i] = np.nan
            continue
        
        # Step 1: Compute net displacement
        L_net = abs(window_prices[-1] - window_prices[0])
        
        # Step 2: Compute path length (sum of step distances)
        L_path = np.sum(np.abs(np.diff(window_prices)))
        
        # Step 3: Roughness ratio
        R = L_path / (L_net + epsilon)
        
        # Step 4: Fractal dimension proxy
        # FDI = 1 + log(R+1) / log(N)
        N = len(window_prices)
        FDI_value = 1.0 + np.log(R + 1.0) / np.log(N)
        
        fdi.iloc[i] = FDI_value
    
    # Normalize: Map from [1.0, 1.8] to [0, 1]
    # FDI_norm = clip((FDI - 1.0) / (1.8 - 1.0), 0, 1)
    fdi_normalized = ((fdi - 1.0) / (1.8 - 1.0)).clip(0.0, 1.0)
    
    fdi_normalized.name = "fractal_dimension_index"
    return fdi_normalized


def feature_hurst_exponent(df: DataFrame) -> Series:
    """
    Compute Hurst Exponent (H) using R/S (Rescaled Range) method.
    
    Hurst quantifies whether returns persist, mean-revert, or act like noise:
    - H > 0.5 → persistent/trending (moves tend to continue)
    - H < 0.5 → mean-reverting (moves tend to snap back)
    - H ≈ 0.5 → near-random walk
    
    This tells the model: should I expect continuation or snap-back after a move.
    
    Calculation (R/S method, simplified rolling window approach):
    1. Compute log returns: r_t = ln(P_t / P_{t-1})
    2. For each rolling window of N=100 returns:
       - Mean of returns: μ
       - Cumulative deviation series: X_k = sum(r_i - μ) for i=1 to k
       - Range: R = max(X_k) - min(X_k)
       - Standard deviation: S = std(r_i)
       - Rescaled range: R/S
       - Hurst estimate: H ≈ log(R/S) / log(N)
    
    Normalization:
    - H naturally lives in [0, 1]
    - Clip to [0, 1] for ML use
    
    Why it adds value:
    - Tells the model if momentum features should be trusted
    - Great for swing trading where persistence matters
    - Helps separate "fake breakouts" (H < 0.5, mean-reverting) from real trends
    - Works great with existing ROC, RSI, Stoch, MACD/PPO, Donchian features
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'hurst_exponent' containing Hurst exponent values in [0, 1].
    """
    close = _get_close_series(df)
    
    # Window size for Hurst calculation
    window = 100
    
    # Step 1: Compute log returns
    log_returns = np.log(close / close.shift(1))
    
    # Initialize result series
    hurst = pd.Series(index=close.index, dtype=float)
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Calculate Hurst for each rolling window
    for i in range(len(close)):
        if i < window:
            # Not enough data for full window
            hurst.iloc[i] = np.nan
            continue
        
        # Get window of returns (exclude first NaN return)
        window_returns = log_returns.iloc[i - window + 1:i + 1].values
        
        # Skip if any NaN values in window
        if np.any(np.isnan(window_returns)):
            hurst.iloc[i] = np.nan
            continue
        
        # Step 2: Mean of returns
        mu = np.mean(window_returns)
        
        # Step 3: Cumulative deviation series
        # X_k = sum(r_i - μ) for i=1 to k
        deviations = window_returns - mu
        cumulative_deviations = np.cumsum(deviations)
        
        # Step 4: Range = max(X_k) - min(X_k)
        R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
        
        # Step 5: Standard deviation of returns
        S = np.std(window_returns, ddof=1)  # Use sample std (ddof=1)
        
        # Step 6: Rescaled range
        if S < epsilon:
            # If std is too small, set H to 0.5 (random walk)
            hurst.iloc[i] = 0.5
            continue
        
        RS = R / S
        
        # Step 7: Hurst estimate using simplified single-window approximation
        # H ≈ log(R/S) / log(N)
        # This is a simplified estimator; full R/S method uses multiple window sizes
        if RS < epsilon:
            hurst.iloc[i] = 0.5  # Default to random walk if R/S too small
        else:
            H_estimate = np.log(RS) / np.log(window)
            hurst.iloc[i] = H_estimate
    
    # Normalize: Clip to [0, 1]
    # H naturally lives in [0, 1], but can occasionally go outside due to approximation
    hurst_normalized = hurst.clip(0.0, 1.0)
    
    hurst_normalized.name = "hurst_exponent"
    return hurst_normalized


def feature_price_curvature(df: DataFrame) -> Series:
    """
    Compute Price Curvature (second derivative of trend) using SMA20 as reference.
    
    Curvature measures the acceleration of a trend:
    - Positive curvature → trend is bending up (acceleration)
    - Negative curvature → trend is bending down (deceleration/topping)
    - Near 0 → linear-ish trend, not bending much
    
    This helps catch early reversals and blow-off moves.
    
    Calculation:
    1. Use SMA20 as smooth reference line: T_t = SMA20(close)_t
    2. First derivative (slope): S_t = T_t - T_{t-1}
    3. Second derivative (curvature): C_t = S_t - S_{t-1}
    4. Normalize: C_norm = clip(C_t / (close_t + ε), -0.05, 0.05)
    
    Normalization:
    - Division by price makes it scale-invariant (comparable across tickers)
    - Clipping to [-0.05, 0.05] limits insane spikes from gappy days
    
    Why it adds value:
    - Distinguishes steady trends from accelerating/rolling-over ones
    - Helps the model time entries inside an already-known trend
    - Complements trend_residual, which measures deviation from a line, not curvature
    - Very relevant for swing trading horizons
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'price_curvature' containing normalized curvature values in [-0.05, 0.05].
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate SMA20 as smooth reference line
    sma20 = close.rolling(window=20, min_periods=1).mean()
    
    # Step 2: First derivative (slope) - 1-day change in SMA20
    slope = sma20.diff(1)
    
    # Step 3: Second derivative (curvature) - 1-day change in slope
    curvature = slope.diff(1)
    
    # Step 4: Normalize by price to make it scale-invariant
    epsilon = 1e-10
    curvature_normalized = curvature / (close + epsilon)
    
    # Clip to [-0.05, 0.05] to limit extreme spikes
    curvature_normalized = curvature_normalized.clip(-0.05, 0.05)
    
    curvature_normalized.name = "price_curvature"
    return curvature_normalized


def feature_volatility_of_volatility(df: DataFrame) -> Series:
    """
    Compute Volatility-of-Volatility (VoV) - measures how unstable volatility itself is.
    
    VoV measures the variability of volatility, providing "meta volatility" information:
    - Low VoV → calm, stable regime (signals behave more cleanly)
    - High VoV → chaotic regime, risk of whipsaws/gaps/wild moves
    
    This tells the model whether volatility indicators are reliable or chaotic.
    
    Calculation:
    1. Compute 21-day volatility: σ_21,t = std(r_{t-20}, ..., r_t)
    2. Compute rolling std of volatility over 21 bars: VoV_t = std(σ_21,{t-20}, ..., σ_21,t)
    3. Normalize by dividing by long-term average volatility: VoV_rel = VoV_t / mean(σ_21)
    4. Clip to [0, 3]
    
    Normalization:
    - Division by long-term average volatility makes it relative and comparable
    - Clipping to [0, 3] limits extreme values
    
    Why it adds value:
    - Tells the model whether volatility indicators are reliable or chaotic
    - Helps risk-aware decision making (e.g., avoid super unstable regimes)
    - Strong context feature when paired with ATR, BB width, TTM squeeze, volatility_ratio
    
    Args:
        df: Input DataFrame with 'close' column.
    
    Returns:
        Series named 'volatility_of_volatility' containing normalized VoV values in [0, 3].
    """
    close = _get_close_series(df)
    
    # Step 1: Compute 21-day volatility (same as feature_volatility_21d)
    # Calculate 21-day rolling standard deviation of daily returns
    vol_21 = close.pct_change().rolling(window=21, min_periods=1).std()
    
    # Step 2: Compute rolling std of volatility itself over 21 bars
    # This measures how much volatility is changing
    vov = vol_21.rolling(window=21, min_periods=1).std()
    
    # Step 3: Normalize by dividing by long-term average volatility
    # Use a longer window (e.g., 252 days) for the long-term average
    long_term_avg_vol = vol_21.rolling(window=252, min_periods=1).mean()
    
    # Avoid division by zero
    epsilon = 1e-10
    vov_relative = vov / (long_term_avg_vol + epsilon)
    
    # Step 4: Clip to [0, 3]
    vov_normalized = vov_relative.clip(0.0, 3.0)
    
    vov_normalized.name = "volatility_of_volatility"
    return vov_normalized


def feature_mkt_spy_dist_sma200(df: DataFrame, spy_data: DataFrame = None) -> Series:
    """
    Compute SPY distance from SMA200 (market extension vs long-term trend).
    
    Measures how extended the market is vs its long-term trend baseline.
    This provides market regime context:
    - Higher (positive) = more risk-on / bullish environment (market extended above trend)
    - Near 0 = neutral (market at trend baseline)
    - Lower (negative) = risk-off / bearish regime (market below trend)
    
    Calculation:
    1. Calculate SPY SMA200: `spy_sma200 = mean(SPY_close over 200 trading days)`
    2. Calculate distance: `dist = (SPY_close / spy_sma200) - 1`
    3. Calculate rolling z-score: `z = (dist - mean_rolling(dist)) / std_rolling(dist)`
       - Rolling window: 1260 days (~5 years)
    4. Clip to `[-3, 3]`
    
    Normalization:
    - Z-score normalization makes it comparable across different market regimes
    - Clipping to [-3, 3] limits extreme values
    - Range: [-3, 3] (standard deviations from mean)
    
    Why it adds value:
    - Provides market regime context (risk-on vs risk-off)
    - Helps model understand market environment (extended vs mean-reverting)
    - Complements beta_spy_252d (correlation) with extension/regime information
    - Critical for swing trading where market regime matters
    
    Args:
        df: Input DataFrame with 'close' column (used for date alignment only).
        spy_data: SPY DataFrame (optional, will be loaded if not provided)
    
    Returns:
        Series named 'mkt_spy_dist_sma200' containing normalized SPY distance values in [-3, 3].
    """
    close = _get_close_series(df)
    
    # Load SPY data if not provided
    if spy_data is None:
        spy_data = _load_spy_data()
        if spy_data is None:
            # Return NaN if SPY data not available
            result = pd.Series(np.nan, index=df.index)
            result.name = "mkt_spy_dist_sma200"
            return result
    
    # Get SPY close price - try different column name variations
    if 'Close' in spy_data.columns:
        spy_close = spy_data['Close']
    elif 'close' in spy_data.columns:
        spy_close = spy_data['close']
    elif 'Adj Close' in spy_data.columns:
        spy_close = spy_data['Adj Close']
    else:
        # Try to get first numeric column
        numeric_cols = spy_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            spy_close = spy_data[numeric_cols[0]]
        else:
            result = pd.Series(np.nan, index=df.index)
            result.name = "mkt_spy_dist_sma200"
            return result
    
    if spy_close is None or len(spy_close) == 0:
        result = pd.Series(np.nan, index=df.index)
        result.name = "mkt_spy_dist_sma200"
        return result
    
    # Ensure SPY index is DatetimeIndex and sorted
    if not isinstance(spy_close.index, pd.DatetimeIndex):
        spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = spy_close.sort_index()
    
    # Get stock date index for alignment
    if isinstance(df.index, pd.DatetimeIndex):
        stock_dates = df.index
    elif df.index.name == 'date' or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
        stock_dates = pd.to_datetime(df.index)
    elif 'date' in df.columns:
        stock_dates = pd.to_datetime(df['date'])
    else:
        result = pd.Series(np.nan, index=df.index)
        result.name = "mkt_spy_dist_sma200"
        return result
    
    # Normalize dates to date-only for matching
    stock_dates_normalized = pd.to_datetime(stock_dates).normalize()
    spy_dates_normalized = pd.to_datetime(spy_close.index).normalize()
    
    # Create temporary DataFrames for alignment
    stock_temp = pd.DataFrame({
        'date': stock_dates_normalized,
        'idx': range(len(stock_dates_normalized))
    })
    spy_temp = pd.DataFrame({
        'date': spy_dates_normalized,
        'close': spy_close.values
    }).sort_values('date')
    
    # Merge on date (left join to preserve stock dates)
    merged = stock_temp.merge(spy_temp, on='date', how='left', suffixes=('_stock', '_spy'), sort=False)
    merged = merged.sort_values('idx').reset_index(drop=True)
    
    # Forward fill missing SPY values
    merged['close'] = merged['close'].ffill().infer_objects(copy=False)
    
    # Step 1: Calculate SPY SMA200
    spy_sma200 = merged['close'].rolling(window=200, min_periods=1).mean()
    
    # Step 2: Calculate distance: (SPY_close / spy_sma200) - 1
    # Avoid division by zero
    epsilon = 1e-10
    dist = (merged['close'] / (spy_sma200 + epsilon)) - 1
    
    # Step 3: Calculate rolling z-score over 1260 days (~5 years)
    # Use full window (no min_periods reduction as requested)
    dist_mean = dist.rolling(window=1260, min_periods=1).mean()
    dist_std = dist.rolling(window=1260, min_periods=1).std()
    
    # Calculate z-score: (dist - mean) / std
    # Avoid division by zero
    dist_std = dist_std.replace(0, np.nan)
    z_score = (dist - dist_mean) / (dist_std + epsilon)
    
    # Step 4: Clip to [-3, 3]
    z_score_clipped = z_score.clip(-3.0, 3.0)
    
    # Create aligned Series with same index as input DataFrame
    result = pd.Series(z_score_clipped.values, index=df.index)
    result.name = "mkt_spy_dist_sma200"
    
    return result


def feature_mkt_spy_sma200_slope(df: DataFrame, spy_data: DataFrame = None) -> Series:
    """
    Compute SPY SMA200 slope (direction/persistence of market's long-term trend).
    
    Measures the direction and persistence of the market's long-term trend.
    This provides market regime context:
    - High percentile (positive) = strong uptrend regime
    - Low percentile (negative) = downtrend regime
    - Mid (near 0) = flat/range regime
    
    Calculation:
    1. Calculate SPY SMA200: `spy_sma200 = mean(SPY_close over 200 trading days)`
    2. Calculate 60-day slope: `slope = (spy_sma200 / spy_sma200.shift(60)) - 1`
    3. Calculate rolling percentile rank: `pct = percentile_rank(slope over 1260 days)`
    4. Remap to [-1, +1]: `mapped = 2 * pct - 1`
    
    Normalization:
    - Percentile rank over 1260 days (~5 years) makes it comparable across different market regimes
    - Remapping to [-1, +1] provides clear interpretation:
      - +1 = strong uptrend (top percentile)
      - 0 = neutral/flat trend (median)
      - -1 = strong downtrend (bottom percentile)
    - Range: [-1, +1]
    
    Why it adds value:
    - Provides market trend direction context (uptrend vs downtrend vs flat)
    - Complements mkt_spy_dist_sma200 (position) with direction information
    - Helps model understand market regime (trending vs ranging)
    - Critical for swing trading where trend direction matters
    - Pairs with distance feature: position + direction = complete market context
    
    Args:
        df: Input DataFrame with 'close' column (used for date alignment only).
        spy_data: SPY DataFrame (optional, will be loaded if not provided)
    
    Returns:
        Series named 'mkt_spy_sma200_slope' containing normalized SPY SMA200 slope values in [-1, +1].
    """
    close = _get_close_series(df)
    
    # Load SPY data if not provided
    if spy_data is None:
        spy_data = _load_spy_data()
        if spy_data is None:
            # Return NaN if SPY data not available
            result = pd.Series(np.nan, index=df.index)
            result.name = "mkt_spy_sma200_slope"
            return result
    
    # Get SPY close price - try different column name variations
    if 'Close' in spy_data.columns:
        spy_close = spy_data['Close']
    elif 'close' in spy_data.columns:
        spy_close = spy_data['close']
    elif 'Adj Close' in spy_data.columns:
        spy_close = spy_data['Adj Close']
    else:
        # Try to get first numeric column
        numeric_cols = spy_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            spy_close = spy_data[numeric_cols[0]]
        else:
            result = pd.Series(np.nan, index=df.index)
            result.name = "mkt_spy_sma200_slope"
            return result
    
    if spy_close is None or len(spy_close) == 0:
        result = pd.Series(np.nan, index=df.index)
        result.name = "mkt_spy_sma200_slope"
        return result
    
    # Ensure SPY index is DatetimeIndex and sorted
    if not isinstance(spy_close.index, pd.DatetimeIndex):
        spy_close.index = pd.to_datetime(spy_close.index)
    spy_close = spy_close.sort_index()
    
    # Get stock date index for alignment
    if isinstance(df.index, pd.DatetimeIndex):
        stock_dates = df.index
    elif df.index.name == 'date' or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
        stock_dates = pd.to_datetime(df.index)
    elif 'date' in df.columns:
        stock_dates = pd.to_datetime(df['date'])
    else:
        result = pd.Series(np.nan, index=df.index)
        result.name = "mkt_spy_sma200_slope"
        return result
    
    # Normalize dates to date-only for matching
    stock_dates_normalized = pd.to_datetime(stock_dates).normalize()
    spy_dates_normalized = pd.to_datetime(spy_close.index).normalize()
    
    # Create temporary DataFrames for alignment
    stock_temp = pd.DataFrame({
        'date': stock_dates_normalized,
        'idx': range(len(stock_dates_normalized))
    })
    spy_temp = pd.DataFrame({
        'date': spy_dates_normalized,
        'close': spy_close.values
    }).sort_values('date')
    
    # Merge on date (left join to preserve stock dates)
    merged = stock_temp.merge(spy_temp, on='date', how='left', suffixes=('_stock', '_spy'), sort=False)
    merged = merged.sort_values('idx').reset_index(drop=True)
    
    # Forward fill missing SPY values
    merged['close'] = merged['close'].ffill().infer_objects(copy=False)
    
    # Step 1: Calculate SPY SMA200
    spy_sma200 = merged['close'].rolling(window=200, min_periods=1).mean()
    
    # Step 2: Calculate 60-day slope: (spy_sma200 / spy_sma200.shift(60)) - 1
    # Avoid division by zero
    epsilon = 1e-10
    spy_sma200_shifted = spy_sma200.shift(60)
    slope = (spy_sma200 / (spy_sma200_shifted + epsilon)) - 1
    
    # Step 3: Calculate rolling percentile rank over 1260 days (~5 years)
    # Use full window (no min_periods reduction as requested)
    # Calculate percentile rank: for each value, what percentile is it within the rolling window?
    pct_rank = pd.Series(index=slope.index, dtype=float)
    
    for i in range(len(slope)):
        if pd.isna(slope.iloc[i]):
            pct_rank.iloc[i] = np.nan
            continue
        
        # Get rolling window (up to 1260 days, but use available data if less)
        start_idx = max(0, i - 1260 + 1)
        window_data = slope.iloc[start_idx:i + 1]
        
        # Remove NaN values from window
        window_data_clean = window_data.dropna()
        
        if len(window_data_clean) == 0:
            pct_rank.iloc[i] = np.nan
        elif len(window_data_clean) == 1:
            # Only one value, default to median (0.5)
            pct_rank.iloc[i] = 0.5
        else:
            # Calculate percentile rank: (number of values <= current) / total
            current_val = slope.iloc[i]
            rank = (window_data_clean <= current_val).sum() / len(window_data_clean)
            pct_rank.iloc[i] = rank
    
    # Step 4: Remap to [-1, +1]: mapped = 2 * pct - 1
    mapped = 2 * pct_rank - 1
    
    # Create aligned Series with same index as input DataFrame
    result = pd.Series(mapped.values, index=df.index)
    result.name = "mkt_spy_sma200_slope"
    
    return result


# Ownership features removed - not consistent and cannot be trusted
