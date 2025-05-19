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
    Compute true 5-day forward return:
      (close_{t+5} / close_t) - 1
    """
    close = _get_close_series(df)
    # shift(-5) looks exactly 5 rows ahead
    return close.shift(-5) / close - 1

def feature_10d_return(df: pd.DataFrame) -> pd.Series:
    """
    Compute 10-day forward return: (close_{t+10} / close_t) - 1.
    """
    close = _get_close_series(df)
    return close.shift(-10) / close - 1

def feature_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR) over the given period.
    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = rolling mean of TR.
    """
    # High series
    if 'High' in df.columns:
        high = df['High']
    elif 'high' in df.columns:
        high = df['high']
    else:
        raise KeyError("DataFrame must contain 'High' or 'high' column")
    # Low series
    if 'Low' in df.columns:
        low = df['Low']
    elif 'low' in df.columns:
        low = df['low']
    else:
        raise KeyError("DataFrame must contain 'Low' or 'low' column")
    # Close series
    close = _get_close_series(df)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr

def feature_bb_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Compute Bollinger Band width:
      width = (upper - lower) / middle = (2 * std_dev * std) / sma
    """
    close = _get_close_series(df)
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    width = (2 * std_dev * std) / sma
    return width

def feature_ema_cross(df: pd.DataFrame, span_short: int = 12, span_long: int = 26) -> pd.Series:
    """
    Compute EMA(span_short) - EMA(span_long).
    """
    close = _get_close_series(df)
    ema_short = close.ewm(span=span_short, adjust=False).mean()
    ema_long  = close.ewm(span=span_long,  adjust=False).mean()
    return ema_short - ema_long

def feature_obv(df: pd.DataFrame) -> pd.Series:
    """
    Compute On‐Balance Volume (OBV):
    OBV[0] = volume[0]
    For i > 0:
        if close[i] > close[i-1]:  OBV[i] = OBV[i-1] + volume[i]
        elif close[i] < close[i-1]: OBV[i] = OBV[i-1] - volume[i]
        else:                       OBV[i] = OBV[i-1]
    """
    close = _get_close_series(df)

    # fetch volume with case‐insensitivity
    if 'volume' in df.columns:
        vol = df['volume']
    elif 'Volume' in df.columns:
        vol = df['Volume']
    else:
        raise KeyError("DataFrame must contain 'volume' or 'Volume' column")

    # seed OBV with the first day's volume
    obv = [vol.iloc[0]]

    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + vol.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - vol.iloc[i])
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=df.index)

def feature_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index over `period`:
      RSI = 100 - 100/(1 + RS), where
      RS = avg_gain / avg_loss (rolling means)
    """
    close = _get_close_series(df)
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
