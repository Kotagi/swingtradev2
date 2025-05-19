import pandas as pd
import numpy as np

def _get_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Return the 'close' series, handling different casing.
    """
    if 'close' in df.columns:
        return df['close']
    elif 'Close' in df.columns:
        return df['Close']
    else:
        raise KeyError("DataFrame must contain 'close' or 'Close' column")

def feature_5d_return(df: pd.DataFrame) -> pd.Series:
    """
    Compute 5-day return:
    (close_{t+4} / close_t) - 1 for t = 0..len(df)-5
    """
    close = _get_close_series(df)
    n = 5
    out = pd.Series(np.nan, index=df.index)
    valid_count = len(close) - n + 1
    for i in range(valid_count):
        out.iloc[i] = close.iloc[i + n - 1] / close.iloc[i] - 1
    return out

def feature_10d_return(df: pd.DataFrame) -> pd.Series:
    """
    Compute 10-day forward return:
      (close_{t+10} / close_t) - 1
    for every t where close_{t+10} exists.
    """
    close = _get_close_series(df)
    # This will produce NaN for the final 10 rows automatically.
    return close.shift(-10) / close - 1

def feature_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) over the given period.
    ATR is the rolling mean of True Range (TR), where
      TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    """
    # Retrieve high series
    if 'High' in df.columns:
        high = df['High']
    elif 'high' in df.columns:
        high = df['high']
    else:
        raise KeyError("DataFrame must contain 'High' or 'high' column")
    # Retrieve low series
    if 'Low' in df.columns:
        low = df['Low']
    elif 'low' in df.columns:
        low = df['low']
    else:
        raise KeyError("DataFrame must contain 'Low' or 'low' column")
    # Retrieve close series
    close = _get_close_series(df)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # Combine into one DataFrame and take the row-wise max
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Simple rolling mean for ATR
    atr = tr.rolling(window=period).mean()
    return atr

def feature_bb_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Compute Bollinger Band width: (upper_band - lower_band) / middle_band
    where upper/lower are mean ± std_dev * std.
    """
    raise NotImplementedError

def feature_ema_cross(df: pd.DataFrame, span_short: int = 12, span_long: int = 26) -> pd.Series:
    """
    Compute EMA(span_short) - EMA(span_long)
    """
    raise NotImplementedError

def feature_obv(df: pd.DataFrame) -> pd.Series:
    """
    Compute On-Balance Volume (OBV)
    """
    raise NotImplementedError

def feature_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI)
    """
    raise NotImplementedError
