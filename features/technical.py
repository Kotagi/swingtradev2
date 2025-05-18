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
    Compute 10-day return:
    (close_{t+10} / close_t) - 1 for t = 0 only
    """
    close = _get_close_series(df)
    out = pd.Series(np.nan, index=df.index)
    if len(close) > 10:
        out.iloc[0] = close.iloc[10] / close.iloc[0] - 1
    return out

# Stubs for the remaining Phase-3 features
def feature_atr(df, period=14):
    raise NotImplementedError

def feature_bb_width(df, period=20, std_dev=2.0):
    raise NotImplementedError

def feature_ema_cross(df, span_short=12, span_long=26):
    raise NotImplementedError

def feature_obv(df):
    raise NotImplementedError

def feature_rsi(df, period=14):
    raise NotImplementedError
