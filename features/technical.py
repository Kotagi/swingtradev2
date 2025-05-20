# src/features/technical.py

import pandas as pd
import numpy as np
import pandas_ta as ta

def _get_close_series(df: pd.DataFrame) -> pd.Series:
    if 'close' in df.columns:
        s = df['close']
    elif 'Close' in df.columns:
        s = df['Close']
    else:
        raise KeyError("DataFrame must contain 'close' or 'Close'")
    return s

def feature_5d_return(df: pd.DataFrame) -> pd.Series:
    close = _get_close_series(df)
    s = close.shift(-5) / close - 1
    s.name = "5d_return"
    return s

def feature_10d_return(df: pd.DataFrame) -> pd.Series:
    close = _get_close_series(df)
    s = close.shift(-10) / close - 1
    s.name = "10d_return"
    return s

def feature_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df.get('High', df.get('high'))
    low   = df.get('Low',  df.get('low'))
    close = _get_close_series(df)

    prev = close.shift(1)
    tr1  = high - low
    tr2  = (high - prev).abs()
    tr3  = (low  - prev).abs()
    tr   = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=period).mean()
    atr.name = f"atr_{period}"
    return atr

def feature_bb_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    close = _get_close_series(df)
    sma   = close.rolling(window=period, min_periods=period).mean()
    std   = close.rolling(window=period, min_periods=period).std(ddof=0)
    width = (2 * std_dev * std) / sma
    width.name = f"bb_width_{period}"
    return width

def feature_ema_cross(df: pd.DataFrame, span_short: int = 12, span_long: int = 26) -> pd.Series:
    close     = _get_close_series(df)
    ema_short = close.ewm(span=span_short, adjust=False, min_periods=span_short).mean()
    ema_long  = close.ewm(span=span_long,  adjust=False, min_periods=span_long).mean()
    diff      = ema_short - ema_long
    diff.name = f"ema_cross_{span_short}_{span_long}"
    return diff

def feature_obv(df: pd.DataFrame) -> pd.Series:
    close = _get_close_series(df)
    vol   = df.get('volume', df.get('Volume'))
    dir   = np.sign(close.diff().fillna(0))
    obv   = (dir * vol).cumsum()
    obv.name = "obv"
    return obv

def feature_obv_pct(df: pd.DataFrame) -> pd.Series:
    """
    Daily percent change of OBV.
    Replace infinities (from divide-by-zero) with NaN.
    """
    obv = feature_obv(df)
    pct = obv.pct_change()
    pct = pct.replace([np.inf, -np.inf], np.nan)
    pct.name = "obv_pct"
    return pct

def feature_obv_zscore(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    OBV relative to rolling mean.
    Replace infinities (from divide-by-zero) with NaN.
    """
    obv = feature_obv(df)
    ma  = obv.rolling(window=window, min_periods=window).mean()
    rel = (obv - ma) / ma
    rel = rel.replace([np.inf, -np.inf], np.nan)
    rel.name = f"obv_z{window}"
    return rel

def feature_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI) over `period` days via pandas-ta.
    """
    s = ta.rsi(df["close"], length=period)       # returns a Series
    s.name = f"rsi_{period}"
    return s

def feature_sma_5(df: pd.DataFrame) -> pd.Series:
    """
    5-day simple moving average of close.
    """
    close = _get_close_series(df)
    sma5  = close.rolling(window=5, min_periods=5).mean()
    sma5.name = "sma_5"
    return sma5

def feature_ema_5(df: pd.DataFrame) -> pd.Series:
    """
    5-day exponential moving average of close.
    """
    close = _get_close_series(df)
    ema5  = close.ewm(span=5, adjust=False, min_periods=5).mean()
    ema5.name = "ema_5"
    return ema5

def feature_sma_10(df: pd.DataFrame) -> pd.Series:
    """
    10-day simple moving average of close.
    """
    close = _get_close_series(df)
    sma10 = close.rolling(window=10, min_periods=10).mean()
    sma10.name = "sma_10"
    return sma10

def feature_ema_10(df: pd.DataFrame) -> pd.Series:
    """
    10-day exponential moving average of close.
    """
    close = _get_close_series(df)
    ema10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
    ema10.name = "ema_10"
    return ema10

def feature_sma_50(df: pd.DataFrame) -> pd.Series:
    """
    50-day simple moving average of close.
    """
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=50).mean()
    sma50.name = "sma_50"
    return sma50

def feature_ema_50(df: pd.DataFrame) -> pd.Series:
    """
    50-day exponential moving average of close.
    """
    close = _get_close_series(df)
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema50.name = "ema_50"
    return ema50

def feature_adx_14(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute 14-day Average Directional Index (ADX) via pandas-ta.
    """
    # ta.adx returns a DataFrame with columns: ['ADX_14','DMP_14','DMN_14']
    adx_df = ta.adx(high=df["high"], low=df["low"], close=df["close"], length=period)
    s = adx_df[f"ADX_{period}"]
    s.name = f"adx_{period}"
    return s
