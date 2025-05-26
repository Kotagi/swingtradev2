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
import pandas_ta as ta
from pandas import Series, DataFrame


def _get_close_series(df: DataFrame) -> Series:
    """
    Return the 'close' price series, handling both lowercase and uppercase.

    Args:
        df: Input DataFrame.

    Returns:
        Series of closing prices.

    Raises:
        KeyError: If neither 'close' nor 'Close' is present.
    """
    if 'close' in df.columns:
        return df['close']
    if 'Close' in df.columns:
        return df['Close']
    raise KeyError("DataFrame must contain 'close' or 'Close' column")

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

def feature_5d_return(df: DataFrame) -> Series:
    """
    Compute 5-day forward return: (close_{t+5} / close_t) - 1.

    Args:
        df: Input DataFrame with 'close' prices.

    Returns:
        Series named '5d_return' of forward returns.
    """
    close = _get_close_series(df)
    s = close.shift(-5) / close - 1
    s.name = "5d_return"
    return s


def feature_10d_return(df: DataFrame) -> Series:
    """
    Compute 10-day forward return: (close_{t+10} / close_t) - 1.

    Args:
        df: Input DataFrame with 'close' prices.

    Returns:
        Series named '10d_return' of forward returns.
    """
    close = _get_close_series(df)
    s = close.shift(-10) / close - 1
    s.name = "10d_return"
    return s

def feature_close_vs_ma10(df: DataFrame) -> Series:
    """
    Compute the ratio of today's close to its 10-day simple moving average.

    Steps:
      1. Retrieve closing price series (handles 'close'/'Close').
      2. Compute 10-day SMA.
      3. Divide close_t by SMA10 and name the result.

    Args:
        df: Input DataFrame with a 'close' column.

    Returns:
        Series named 'close_vs_ma10' of close/SMA10 ratios.
    """
    close = _get_close_series(df)
    sma10 = close.rolling(window=10, min_periods=1).mean()
    ratio = close / sma10
    ratio.name = "close_vs_ma10"
    return ratio


def feature_atr(df: DataFrame, period: int = 14) -> Series:
    """
    Compute Average True Range (ATR) over a given period via pandas-ta.

    Args:
        df: Input DataFrame with 'high', 'low', 'close' columns.
        period: Lookback length for ATR calculation.

    Returns:
        Series named 'atr_{period}' of ATR values.
    """
    s = ta.atr(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=period
    )
    s.name = f"atr_{period}"
    return s


def feature_bb_width(df: DataFrame, period: int = 20, std_dev: float = 2.0) -> Series:
    """
    Compute Bollinger Band width: (upper_band - lower_band) / middle_band.

    Uses pandas-ta's bbands function.

    Args:
        df: Input DataFrame with 'close' column.
        period: Window length for moving average and bands.
        std_dev: Multiplier for standard deviation in bands.

    Returns:
        Series named 'bb_width_{period}' of band widths.
    """
    bb = ta.bbands(close=df["close"], length=period, std=std_dev)
    # Extract lower, middle, upper bands by column order
    lower = bb.iloc[:, 0]
    mid = bb.iloc[:, 1]
    upper = bb.iloc[:, 2]
    width = (upper - lower) / mid
    width.name = f"bb_width_{period}"
    return width


def feature_ema_cross(df: DataFrame, span_short: int = 12, span_long: int = 26) -> Series:
    """
    Compute EMA difference: EMA(short) - EMA(long).

    Args:
        df: Input DataFrame with 'close' column.
        span_short: Span for short EMA.
        span_long: Span for long EMA.

    Returns:
        Series named 'ema_cross_{span_short}_{span_long}'.
    """
    e1 = ta.ema(df["close"], length=span_short)
    e2 = ta.ema(df["close"], length=span_long)
    diff = e1 - e2
    diff.name = f"ema_cross_{span_short}_{span_long}"
    return diff


def feature_obv(df: DataFrame) -> Series:
    """
    Compute On-Balance Volume (OBV) via pandas-ta.

    Args:
        df: Input DataFrame with 'close' and 'volume' columns.

    Returns:
        Series named 'obv' of cumulative OBV values.
    """
    s = ta.obv(df["close"], df["volume"])
    s.name = "obv"
    return s


def feature_obv_pct(df: DataFrame, length: int = 1) -> Series:
    """
    Compute daily percent change of OBV via Rate-of-Change.

    Args:
        df: Input DataFrame with 'close' and 'volume' columns.
        length: Period for percent change (default 1 day).

    Returns:
        Series named 'obv_pct' of percent changes.
    """
    obv = ta.obv(df["close"], df["volume"])
    s = ta.roc(obv, length=length)
    s.name = "obv_pct"
    return s


def feature_obv_zscore(df: DataFrame, length: int = 20) -> Series:
    """
    Compute z-score of OBV relative to its moving average.

    Args:
        df: Input DataFrame with 'close' and 'volume' columns.
        length: Window for z-score normalization.

    Returns:
        Series named 'obv_z{length}' of z-scored values.
    """
    obv = ta.obv(df["close"], df["volume"])
    s = ta.zscore(obv, length=length)
    s.name = f"obv_z{length}"
    return s


def feature_rsi(df: DataFrame, period: int = 14) -> Series:
    """
    Compute Relative Strength Index (RSI) via pandas-ta.

    Args:
        df: Input DataFrame with 'close' column.
        period: Lookback length for RSI.

    Returns:
        Series named 'rsi_{period}' of RSI values.
    """
    s = ta.rsi(df["close"], length=period)
    s.name = f"rsi_{period}"
    return s


def feature_sma_5(df: DataFrame) -> Series:
    """
    Compute 5-day Simple Moving Average via pandas-ta.

    Args:
        df: Input DataFrame with 'close' column.

    Returns:
        Series named 'sma_5' of SMA values.
    """
    s = ta.sma(df["close"], length=5)
    s.name = "sma_5"
    return s


def feature_ema_5(df: DataFrame) -> Series:
    """
    Compute 5-day Exponential Moving Average via pandas-ta.

    Args:
        df: Input DataFrame with 'close' column.

    Returns:
        Series named 'ema_5' of EMA values.
    """
    s = ta.ema(df["close"], length=5)
    s.name = "ema_5"
    return s


def feature_sma_10(df: DataFrame) -> Series:
    """
    Compute 10-day Simple Moving Average via pandas-ta.

    Args:
        df: Input DataFrame with 'close' column.

    Returns:
        Series named 'sma_10' of SMA values.
    """
    s = ta.sma(df["close"], length=10)
    s.name = "sma_10"
    return s


def feature_ema_10(df: DataFrame) -> Series:
    """
    Compute 10-day Exponential Moving Average via pandas-ta.

    Args:
        df: Input DataFrame with 'close' column.

    Returns:
        Series named 'ema_10' of EMA values.
    """
    s = ta.ema(df["close"], length=10)
    s.name = "ema_10"
    return s


def feature_sma_50(df: DataFrame) -> Series:
    """
    Compute 50-day Simple Moving Average via pandas-ta.

    Args:
        df: Input DataFrame with 'close' column.

    Returns:
        Series named 'sma_50' of SMA values.
    """
    s = ta.sma(df["close"], length=50)
    s.name = "sma_50"
    return s


def feature_ema_50(df: DataFrame) -> Series:
    """
    Compute 50-day Exponential Moving Average via pandas-ta.

    Args:
        df: Input DataFrame with 'close' column.

    Returns:
        Series named 'ema_50' of EMA values.
    """
    s = ta.ema(df["close"], length=50)
    s.name = "ema_50"
    return s


def feature_adx_14(df: DataFrame, period: int = 14) -> Series:
    """
    Compute 14-day Average Directional Index (ADX) via pandas-ta.

    Args:
        df: Input DataFrame with 'high', 'low', 'close' columns.
        period: Lookback length for ADX.

    Returns:
        Series named 'adx_{period}' of ADX values.
    """
    adx_df = ta.adx(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=period
    )
    s = adx_df[f"ADX_{period}"]
    s.name = f"adx_{period}"
    return s
