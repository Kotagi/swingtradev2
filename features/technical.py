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


def _get_column(df: DataFrame, col_name: str) -> Series:
    """
    Get a column from DataFrame with case-insensitive matching.
    
    Args:
        df: Input DataFrame.
        col_name: Column name to find (case-insensitive).
    
    Returns:
        Series from the DataFrame.
    
    Raises:
        KeyError: If column not found in any case variation.
    """
    col_lower = col_name.lower()
    for col in df.columns:
        if col.lower() == col_lower:
            return df[col]
    raise KeyError(f"DataFrame must contain '{col_name}' column (case-insensitive)")


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

def feature_close_vs_ma20(df: DataFrame) -> Series:
    """
    Compute the ratio of today's close to its 20-day simple moving average.

    Steps:
      1. Retrieve the closing price series (handles upper/lower case).
      2. Compute the 20-day SMA (min_periods=1 so you don’t introduce NaNs early).
      3. Divide close_t by SMA20 and name the result.
    Args:
        df: Input DataFrame with a 'close' or 'Close' column.
    Returns:
        Series named 'close_vs_ma20' of close/SMA20 ratios.
    """
    close = _get_close_series(df)
    sma20 = close.rolling(window=20, min_periods=1).mean()
    ratio = close / sma20
    ratio.name = "close_vs_ma20"
    return ratio

def feature_close_zscore_20(df: DataFrame) -> Series:
    """
    Compute the 20-day rolling z-score of today’s close:
      (close_t − mean(close_{t-19..t})) / std(close_{t-19..t}).

    Steps:
      1. Retrieve closing-price series (handles upper/lower case).
      2. Compute 20-day rolling mean & std (min_periods=1 to avoid NaNs early).
      3. Subtract mean from close and divide by std.
      4. Name the Series 'close_zscore_20'.

    Args:
        df: Input DataFrame with a 'close' or 'Close' column.

    Returns:
        Series named 'close_zscore_20', centered at 0 with unit variance.
    """
    close = _get_close_series(df)
    mean20 = close.rolling(window=20, min_periods=1).mean()
    std20  = close.rolling(window=20, min_periods=1).std().replace(0, np.nan)
    zscore = (close - mean20) / std20
    zscore.name = "close_zscore_20"
    return zscore

def feature_price_percentile_20d(df: DataFrame) -> Series:
    """
    Compute the 20-day rolling percentile rank of today’s close.

    Steps:
      1. Retrieve the closing-price series (handles upper/lower case).
      2. For each day t, take the last 20 closes (including t).
      3. Compute the percentile of close_t among that window:
         (# of window values ≤ close_t − 1) / (window_size − 1)
      4. Name the resulting Series for downstream use.

    Args:
        df: Input DataFrame with a 'close' column.

    Returns:
        Series named 'price_percentile_20d' with values in [0,1].
    """
    close = _get_close_series(df)
    def pct_rank(window: np.ndarray) -> float:
        # window[-1] is today's close; rank among window
        today = window[-1]
        # count how many are less than today
        less_equal = np.sum(window <= today) - 1
        denom = len(window) - 1
        return float(less_equal / denom) if denom > 0 else 0.5

    pct = close.rolling(window=20, min_periods=1) \
               .apply(pct_rank, raw=True)
    pct.name = "price_percentile_20d"
    return pct

def feature_gap_up_pct(df: DataFrame) -> Series:
    """
    Compute the percent gap-up at open relative to prior close:
      (open_t / close_{t-1} - 1) * 100.

    Steps:
      1. Retrieve the closing-price series (handles ‘close’/‘Close’).
      2. Retrieve today’s opening price (df['open']).
      3. Shift close by 1 day to get prior close.
      4. Compute (open_t / close_{t-1} − 1) * 100.
      5. Name the Series 'gap_up_pct'.

    Args:
        df: Input DataFrame with 'open' and 'close' columns.

    Returns:
        Series named 'gap_up_pct' with values in percent.
    """
    close = _get_close_series(df)
    openp = _get_open_series(df)
    gap = (openp / close.shift(1) - 1) * 100
    gap.name = "gap_up_pct"
    return gap

def feature_daily_range_pct(df: DataFrame) -> Series:
    """
    Compute the percent size of the intraday high-low range relative to open:
      ((high_t − low_t) / open_t) × 100.

    Steps:
      1. Retrieve today’s open (handles ‘open’/‘Open’), high, and low series.
      2. Compute range = high_t − low_t.
      3. Divide by open_t and multiply by 100.
      4. Name the Series 'daily_range_pct'.

    Args:
        df: DataFrame with 'open','high','low' columns.

    Returns:
        Series named 'daily_range_pct' giving range% per bar.
    """
    openp = _get_open_series(df)
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)

    # compute percent range
    pct = ((highp - lowp) / openp) * 100
    pct.name = "daily_range_pct"
    return pct

def feature_candle_body_pct(df: DataFrame) -> Series:
    """
    Compute the percent size of the candle body relative to the high-low range:
      ((close_t − open_t) / (high_t − low_t)) * 100

    Steps:
      1. Retrieve open, high, low, and close series (case-insensitive).
      2. Compute body = close − open.
      3. Compute range = high − low, replacing any 0 with NaN to avoid divide-by-zero.
      4. Compute (body / range) * 100 and name the Series.
    Args:
        df: Input DataFrame with 'open','high','low','close' columns.
    Returns:
        Series named 'candle_body_pct' with values in [-∞,∞], clipped by typical range.
    """
    openp = _get_open_series(df)
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)
    close = _get_close_series(df)

    body = close - openp
    rng  = (highp - lowp).replace(0, np.nan)
    pct  = (body / rng) * 100
    pct.name = "candle_body_pct"
    return pct

def feature_close_position_in_range(df: DataFrame) -> Series:
    """
    Compute the relative position of the close within the intraday range:
      (close_t − low_t) / (high_t − low_t).

    Steps:
      1. Retrieve closing-price series (handles 'close'/'Close').
      2. Retrieve high and low series (handles lower/upper case).
      3. Compute range = high_t − low_t, replacing zeros with NaN to avoid divide-by-zero.
      4. Compute (close_t − low_t) / range.
      5. Name the Series 'close_position_in_range'.

    Args:
        df: Input DataFrame with 'close','high','low' columns.

    Returns:
        Series named 'close_position_in_range' in [0,1], where 0 means close==low, 1 means close==high.
    """
    close = _get_close_series(df)
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)

    # avoid division by zero
    rng = (highp - lowp).replace(0, np.nan)

    pos = (close - lowp) / rng
    pos.name = "close_position_in_range"
    return pos

def feature_high_vs_close(df: DataFrame) -> Series:
    """
    Compute the percent difference between the intraday high and the close:
      (high_t / close_t - 1) * 100

    Steps:
      1. Retrieve today’s close via our _get_close_series helper.
      2. Retrieve today’s high price (handles 'high'/'High').
      3. Compute (high_t / close_t - 1) * 100.
      4. Name the Series 'high_vs_close'.

    Args:
        df: DataFrame with 'high' and 'close' columns.

    Returns:
        Series named 'high_vs_close' giving the intraday extension above close.
    """
    close = _get_close_series(df)
    highp = _get_high_series(df)
    pct = (highp / close - 1) * 100
    pct.name = "high_vs_close"
    return pct

def feature_rolling_max_5d_breakout(df: DataFrame) -> Series:
    """
    Compute the percent that today’s close exceeds the max close of the prior 5 days:
      max_prev5 = max(close_{t-5..t-1})
      breakout_pct = max( (close_t / max_prev5 - 1) , 0 ) * 100

    Steps:
      1. Retrieve the closing-price series (via _get_close_series).
      2. Shift by 1 day and take a 5-day rolling max (min_periods=1).
      3. Divide today’s close by that prior-5-day max, subtract 1.
      4. Clip negative values to 0 (only positive breakouts).
      5. Multiply by 100 and name the Series.

    Args:
        df: Input DataFrame with a 'close' or 'Close' column.

    Returns:
        Series named 'rolling_max_5d_breakout' giving % breakout over the prior 5-day high.
    """
    close = _get_close_series(df)
    # prior 5-day high
    prev_max5 = close.shift(1).rolling(window=5, min_periods=1).max()
    # percent above that high, clip negatives to zero
    pct = ((close / prev_max5 - 1).clip(lower=0)) * 100
    pct.name = "rolling_max_5d_breakout"
    return pct

def feature_rolling_min_5d_breakdown(df: DataFrame) -> Series:
    """
    Compute the percent that today’s close falls below the min close of the prior 5 days:
      min_prev5 = min(close_{t-5..t-1})
      breakdown_pct = max((min_prev5 - close_t) / min_prev5, 0) * 100

    Steps:
      1. Retrieve the closing-price series via _get_close_series.
      2. Shift by 1 day and take a 5-day rolling minimum (min_periods=1).
      3. Compute (min_prev5 - close_t) / min_prev5.
      4. Clip negative values to 0 (only positive breakdowns).
      5. Multiply by 100 and name the Series.

    Args:
        df: Input DataFrame with 'close' or 'Close' column.

    Returns:
        Series named 'rolling_min_5d_breakdown'.
    """
    close = _get_close_series(df)
    prev_min5 = close.shift(1).rolling(window=5, min_periods=1).min()
    pct = ((prev_min5 - close) / prev_min5).clip(lower=0) * 100
    pct.name = "rolling_min_5d_breakdown"
    return pct

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
        high=_get_high_series(df),
        low=_get_low_series(df),
        close=_get_close_series(df),
        length=period
    )
    s.name = f"atr_{period}"
    return s

def feature_atr_pct_of_price(df: DataFrame) -> Series:
    """
    Compute the Average True Range as a percent of today's close:
      atr_pct_of_price = (ATR_t / close_t) * 100

    Steps:
      1. Compute ATR (using our existing feature_atr helper).
      2. Retrieve the closing-price series via _get_close_series.
      3. Divide ATR by close and multiply by 100.
      4. Name the Series 'atr_pct_of_price'.

    Args:
        df: Input DataFrame with 'high','low','close' columns.

    Returns:
        Series named 'atr_pct_of_price'.
    """
    # 1) get raw ATR
    atr = feature_atr(df)
    # 2) get close
    close = _get_close_series(df)
    # 3) pct of price
    pct = (atr / close) * 100
    pct.name = "atr_pct_of_price"
    return pct

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
    s = ta.obv(_get_close_series(df), _get_volume_series(df))
    s.name = "obv"
    return s


def feature_obv_pct(df: DataFrame, length: int = 1) -> Series:
    """
    Compute daily percent change of OBV via Rate-of-Change.
    
    Handles edge cases where OBV transitions from 0 to non-zero (which would
    produce infinity in ROC calculation).

    Args:
        df: Input DataFrame with 'close' and 'volume' columns.
        length: Period for percent change (default 1 day).

    Returns:
        Series named 'obv_pct' of percent changes, with infinities replaced by NaN.
    """
    obv = ta.obv(_get_close_series(df), _get_volume_series(df))
    
    # Compute ROC
    s = ta.roc(obv, length=length)
    
    # Handle infinity cases: when OBV goes from 0 to non-zero, ROC = infinity
    # Replace infinities with NaN for downstream handling
    s = s.replace([np.inf, -np.inf], np.nan)
    
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

def feature_volume_avg_ratio_5d(df: DataFrame) -> Series:
    """
    Compute the ratio of today's volume to the prior 5-day average volume.

    Steps:
      1. Retrieve today’s volume (handles 'volume'/'Volume').
      2. Shift by 1 day and take a 5-day rolling mean (min_periods=1).
      3. Replace any zero averages with NaN to avoid divide-by-zero.
      4. Divide today’s volume by that prior-5-day average.
      5. Replace infinities and NaNs with 0.0.
      6. Name the Series 'volume_avg_ratio_5d'.

    Args:
        df: Input DataFrame with a 'volume' column.

    Returns:
        Series named 'volume_avg_ratio_5d'.
    """
    import numpy as np

    vol = _get_volume_series(df)

    # 2) Compute prior 5-day average volume
    avg5 = vol.shift(1).rolling(window=5, min_periods=1).mean()

    # 3) Prevent divide-by-zero
    avg5 = avg5.replace(0, np.nan)

    # 4) Compute ratio
    ratio = vol / avg5

    # 5) Clean up infinities and NaNs
    ratio = ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # 6) Name and return
    ratio.name = "volume_avg_ratio_5d"
    return ratio

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

def feature_rsi_slope(df: DataFrame, period: int = 14, length: int = 3) -> Series:
    """
    Compute the average daily change (“slope”) of the period‐RSI over a lookback window.

    Steps:
      1. Compute the RSI via pandas‐ta over `period` days.
      2. Shift that series by `length` days.
      3. Subtract and divide by `length` to get per‐day slope.
      4. Name the Series 'rsi_slope'.

    Args:
        df: Input DataFrame with a 'close' column.
        period: Lookback length for RSI (default 14).  
        length: Number of days over which to measure slope (default 5).

    Returns:
        Series named 'rsi_slope' of per‐day RSI changes.
    """
    # 1) Calculate RSI  
    rsi_series = ta.rsi(df["close"], length=period)  # :contentReference[oaicite:0]{index=0}

    # 2) Compute N-day difference, then annualize as per-day slope
    slope = (rsi_series - rsi_series.shift(length)) / length

    # 3) Clean up any NaNs (initial periods) if you like, or leave them  
    slope.name = "rsi_slope"
    return slope

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

def feature_macd_line(df: DataFrame, fast: int = 12, slow: int = 26) -> Series:
    """
    Compute the MACD line: the difference between fast and slow EMAs of close.

    Steps:
      1. Retrieve the closing‐price series via _get_close_series.
      2. Compute the fast EMA (default length=12).
      3. Compute the slow EMA (default length=26).
      4. Subtract: fast_ema − slow_ema.
      5. Name the Series 'macd_line'.

    Args:
        df: Input DataFrame with a 'close' column.
        fast: Window for the fast EMA (default 12).
        slow: Window for the slow EMA (default 26).

    Returns:
        Series named 'macd_line'.
    """
    close = _get_close_series(df)
    # pandas_ta EMAs
    ema_fast = ta.ema(close, length=fast)
    ema_slow = ta.ema(close, length=slow)
    macd = ema_fast - ema_slow
    macd.name = "macd_line"
    return macd

def feature_macd_histogram(
    df: DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Series:
    """
    Compute the MACD histogram: the difference between the MACD line and its signal line.

    Steps:
      1. Retrieve close series via _get_close_series.
      2. Compute fast EMA (length=fast) and slow EMA (length=slow).
      3. macd_line = fast_ema − slow_ema.
      4. signal_line = EMA(macd_line, length=signal).
      5. histogram = macd_line − signal_line.
      6. Name the Series 'macd_histogram'.

    Args:
        df: Input DataFrame with a 'close' column.
        fast: Window for the fast EMA (default 12).
        slow: Window for the slow EMA (default 26).
        signal: Window for the signal‐line EMA (default 9).

    Returns:
        Series named 'macd_histogram'.
    """
    close = _get_close_series(df)
    ema_fast    = ta.ema(close, length=fast)
    ema_slow    = ta.ema(close, length=slow)
    macd_line   = ema_fast - ema_slow
    signal_line = ta.ema(macd_line, length=signal)
    hist        = macd_line - signal_line
    hist.name   = "macd_histogram"
    return hist

def feature_macd_cross_signal(
    df: DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Series:
    """
    Compute a -1/0/+1 signal for MACD line crossing its signal line.

    Steps:
      1. Retrieve close series via _get_close_series.
      2. Compute fast and slow EMAs, then macd_line = fast_ema - slow_ema.
      3. Compute signal_line = EMA(macd_line, length=signal).
      4. Compute diff = macd_line - signal_line.
      5. Detect cross:
           +1 where diff shifts from ≤0 to >0 (bullish crossover)
           -1 where diff shifts from ≥0 to <0 (bearish crossover)
           0 otherwise
      6. Name the Series 'macd_cross_signal'.

    Args:
        df: Input DataFrame with a 'close' column.
        fast: Window for the fast EMA (default 12).
        slow: Window for the slow EMA (default 26).
        signal: Window for the signal‐line EMA (default 9).

    Returns:
        Series named 'macd_cross_signal' of int values in {-1,0,1}.
    """
    close       = _get_close_series(df)
    ema_fast    = ta.ema(close, length=fast)
    ema_slow    = ta.ema(close, length=slow)
    macd_line   = ema_fast - ema_slow
    signal_line = ta.ema(macd_line, length=signal)
    diff        = macd_line - signal_line

    # 1-day lagged diff
    prev_diff = diff.shift(1)

    # bullish cross: prev_diff <= 0, diff > 0
    bullish = (prev_diff <= 0) & (diff > 0)
    # bearish cross: prev_diff >= 0, diff < 0
    bearish = (prev_diff >= 0) & (diff < 0)

    signal = pd.Series(0, index=diff.index)
    signal[bullish] = 1
    signal[bearish] = -1
    signal.name = "macd_cross_signal"
    return signal

def feature_stoch_k(
    df: DataFrame,
    length: int = 14,
    smooth_k: int = 3
) -> Series:
    """
    Compute the %K line of the Stochastic Oscillator:
      %K_t = 100 × (close_t − lowest_low_{t−length+1…t}) 
                    / (highest_high_{t−length+1…t} − lowest_low_{t−length+1…t})
      then smoothed by a moving average of window `smooth_k`.

    Steps:
      1. Retrieve close, high, and low via _get_close_series / df.
      2. Compute rolling lowest low and highest high over `length` bars.
      3. Compute raw %K.
      4. Smooth %K with an SMA of window `smooth_k` (min_periods=1).
      5. Name the Series 'stoch_k'.

    Args:
        df: Input DataFrame with 'high','low','close' columns.
        length: Lookback length for the oscillator (default 14).
        smooth_k: Smoothing window for %K (default 3).

    Returns:
        Series named 'stoch_k' with values in [0,100].
    """
    close = _get_close_series(df)
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)

    # 1) rolling extremes
    lowest_low  = lowp.rolling(window=length, min_periods=1).min()
    highest_high = highp.rolling(window=length, min_periods=1).max()

    # 2) raw %K
    raw_k = (close - lowest_low) / (highest_high - lowest_low) * 100

    # 3) smooth
    stoch_k = raw_k.rolling(window=smooth_k, min_periods=1).mean()
    stoch_k.name = "stoch_k"
    return stoch_k

def feature_stoch_d(
    df: DataFrame,
    length: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Series:
    """
    Compute the %D (signal) line of the Stochastic Oscillator:
      1) %K_t  = 100 × (close_t − lowest_low_{t−length+1…t})
                   / (highest_high_{t−length+1…t} − lowest_low_{t−length+1…t})
      2) stoch_k_smoothed = SMA(%K, window=smooth_k)
      3) %D_t  = SMA(stoch_k_smoothed, window=smooth_d)

    Steps:
      1. Retrieve close, high, low via _get_close_series / df.
      2. Compute raw %K over `length` bars.
      3. Smooth raw %K by `smooth_k` to get stoch_k.
      4. Further smooth stoch_k by `smooth_d` to get stoch_d.
      5. Name the Series 'stoch_d'.

    Args:
        df: Input DataFrame with 'high', 'low', 'close' columns.
        length: Lookback length for the oscillator (default 14).
        smooth_k: Smoothing window for %K (default 3).
        smooth_d: Smoothing window for %D (default 3).

    Returns:
        Series named 'stoch_d' with values in [0,100].
    """
    close = _get_close_series(df)
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)

    # 1) rolling extremes
    lowest_low   = lowp.rolling(window=length, min_periods=1).min()
    highest_high = highp.rolling(window=length, min_periods=1).max()

    # 2) raw %K
    raw_k = (close - lowest_low) / (highest_high - lowest_low) * 100

    # 3) smooth %K
    stoch_k = raw_k.rolling(window=smooth_k, min_periods=1).mean()

    # 4) smooth %D
    stoch_d = stoch_k.rolling(window=smooth_d, min_periods=1).mean()
    stoch_d.name = "stoch_d"
    return stoch_d

def feature_stoch_cross(
    df: DataFrame,
    length: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Series:
    """
    Compute a -1/0/+1 signal when %K crosses its %D line.

    Steps:
      1. Compute stoch_k via rolling %K and SMA(length=smooth_k).
      2. Compute stoch_d via SMA(stoch_k, window=smooth_d).
      3. diff = stoch_k − stoch_d.
      4. prev_diff = diff.shift(1).
      5. Bullish cross (prev_diff <= 0 & diff > 0): +1  
         Bearish cross (prev_diff >= 0 & diff < 0): −1  
         Else: 0.
      6. Name as 'stoch_cross'.

    Args:
        df: DataFrame with 'high','low','close'.
        length: Lookback for raw %K (default 14).  
        smooth_k: SMA window for %K (default 3).  
        smooth_d: SMA window for %D (default 3).

    Returns:
        Series named 'stoch_cross' of int values in {-1,0,1}.
    """
    close   = _get_close_series(df)
    highp   = df.get("high", df.get("High"))
    lowp    = df.get("low",  df.get("Low"))

    # raw %K
    lowest = lowp.rolling(window=length, min_periods=1).min()
    highest= highp.rolling(window=length, min_periods=1).max()
    raw_k  = (close - lowest) / (highest - lowest) * 100

    # smooth %K and %D
    stoch_k = raw_k.rolling(window=smooth_k, min_periods=1).mean()
    stoch_d = stoch_k.rolling(window=smooth_d, min_periods=1).mean()

    diff     = stoch_k - stoch_d
    prev_diff= diff.shift(1)

    bullish  = (prev_diff <= 0) & (diff > 0)
    bearish  = (prev_diff >= 0) & (diff < 0)

    signal = pd.Series(0, index=diff.index)
    signal[bullish] = 1
    signal[bearish] = -1
    signal.name = "stoch_cross"
    return signal

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
        high=_get_high_series(df),
        low=_get_low_series(df),
        close=_get_close_series(df),
        length=period
    )
    s = adx_df[f"ADX_{period}"]
    s.name = f"adx_{period}"
    return s

def feature_ichimoku_conversion(df: DataFrame, period: int = 9) -> Series:
    """
    Conversion Line (Tenkan-sen): midpoint of highest high and lowest low over `period` bars.

    Steps:
      1. Retrieve high/low via df.
      2. Compute rolling max(high, window=period) and min(low, window=period).
      3. Take their average.
      4. Name Series 'ichimoku_conversion'.

    Args:
        df: DataFrame with 'high' and 'low' columns.
        period: Lookback for conversion line (default 9).

    Returns:
        Series named 'ichimoku_conversion'.
    """
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)
    conv = (highp.rolling(period, min_periods=1).max() +
            lowp.rolling(period,  min_periods=1).min()) / 2
    conv.name = "ichimoku_conversion"
    return conv


def feature_ichimoku_base(df: DataFrame, period: int = 26) -> Series:
    """
    Base Line (Kijun-sen): midpoint of highest high and lowest low over `period` bars.

    Steps analogous to conversion line but with longer period.
    """
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)
    base = (highp.rolling(period, min_periods=1).max() +
            lowp.rolling(period,  min_periods=1).min()) / 2
    base.name = "ichimoku_base"
    return base


def feature_ichimoku_lead_span_a(
    df: DataFrame,
    conv_period: int = 9,
    base_period: int = 26,
    shift: int = 26
) -> Series:
    """
    Leading Span A (Senkou Span A): midpoint of Conversion and Base lines.
    
    WARNING: Original Ichimoku shifts this forward, which causes lookahead bias.
    This version does NOT shift forward to avoid using future data in ML models.
    
    Steps:
      1. Compute conversion & base via their feature functions.
      2. Average them (without forward shift).
    """
    conv = feature_ichimoku_conversion(df, period=conv_period)
    base = feature_ichimoku_base(df,    period=base_period)
    # Do NOT shift forward - that would be lookahead bias
    span_a = (conv + base) / 2
    span_a.name = "ichimoku_lead_span_a"
    return span_a


def feature_ichimoku_lead_span_b(
    df: DataFrame,
    period: int = 52,
    shift: int = 26
) -> Series:
    """
    Leading Span B (Senkou Span B): midpoint of highest high & lowest low over `period` bars.
    
    WARNING: Original Ichimoku shifts this forward, which causes lookahead bias.
    This version does NOT shift forward to avoid using future data in ML models.
    
    Steps:
      1. Compute rolling max(high, window=period) and min(low, window=period).
      2. Average them (without forward shift).
    """
    highp = _get_high_series(df)
    lowp  = _get_low_series(df)
    # Do NOT shift forward - that would be lookahead bias
    span_b = (highp.rolling(period, min_periods=1).max() +
              lowp.rolling(period,  min_periods=1).min()) / 2
    span_b.name = "ichimoku_lead_span_b"
    return span_b


def feature_ichimoku_lagging_span(
    df: DataFrame,
    shift: int = 26
) -> Series:
    """
    Lagging Span (Chikou Span): today's close shifted backward by `shift` bars.
    
    WARNING: This uses past data shifted forward, which can cause issues in ML.
    Consider using this only for visualization, not as a predictive feature.
    
    Steps:
      1. Retrieve close via _get_close_series.
      2. Shift backward (negative shift) by `shift` bars.
    """
    close = _get_close_series(df)
    lag = close.shift(-shift)
    lag.name = "ichimoku_lagging_span"
    return lag

def feature_bullish_engulfing(df: DataFrame) -> Series:
    """
    Flag when today’s bullish candle fully engulfs yesterday’s bearish candle.

    Steps:
      1. Retrieve today’s open/close and yesterday’s open/close (shifted by 1).
      2. Identify a bearish prior bar: prev_close < prev_open.
      3. Identify a bullish current bar: close_t > open_t.
      4. Check engulf:
         - today’s open < prev_close
         - today’s close > prev_open
      5. Combine conditions into a binary Series.

    Args:
        df: Input DataFrame with 'open' and 'close' columns.

    Returns:
        Series named 'bullish_engulfing' with values 1.0 or 0.0.
    """
    import numpy as np

    openp  = _get_open_series(df)
    closep = _get_close_series(df)

    # yesterday’s values
    prev_open  = openp.shift(1)
    prev_close = closep.shift(1)

    # 2) prior bearish
    prior_bear = prev_close < prev_open
    # 3) current bullish
    curr_bull = closep > openp
    # 4) engulf conditions
    engulps = (openp < prev_close) & (closep > prev_open)

    # 5) flag
    flag = (prior_bear & curr_bull & engulps).astype(float)
    flag.name = "bullish_engulfing"
    return flag

def feature_bearish_engulfing(df: DataFrame) -> Series:
    """
    Flag when today’s bearish candle fully engulfs yesterday’s bullish candle.

    Steps:
      1. Get today’s open/close and yesterday’s (shifted by 1).
      2. Prior bullish: prev_close > prev_open.
      3. Current bearish: close_t < open_t.
      4. Engulf: today’s open > prev_close AND today’s close < prev_open.
      5. Combine into 1.0/0.0 flag.
    """
    openp   = _get_open_series(df)
    closep  = _get_close_series(df)
    prev_o  = openp.shift(1); prev_c = closep.shift(1)
    prior_bull = prev_c > prev_o
    curr_bear  = closep < openp
    engulps    = (openp > prev_c) & (closep < prev_o)
    flag       = (prior_bull & curr_bear & engulps).astype(float)
    flag.name  = "bearish_engulfing"
    return flag


def feature_hammer_signal(df: DataFrame, body_pct: float = 0.3, shadow_ratio: float = 2.0) -> Series:
    """
    Flag hammer candles: small real body near top with long lower shadow.

    Steps:
      1. Get open/high/low/close.
      2. Compute body = abs(close−open), range = high−low.
      3. Lower shadow = min(open,close) − low.
      4. Conditions:
         • body <= body_pct * range
         • lower_shadow >= shadow_ratio * body
         • upper_shadow <= body_pct * range
      5. Flag 1.0/0.0.
    """
    openp  = _get_open_series(df)
    closep = _get_close_series(df)
    highp  = _get_high_series(df)
    lowp   = _get_low_series(df)
    body   = (closep - openp).abs()
    rng    = highp - lowp
    lower  = np.minimum(openp, closep) - lowp
    upper  = highp - np.maximum(openp, closep)
    cond   = (
        (body <= body_pct * rng) &
        (lower >= shadow_ratio * body) &
        (upper <= body_pct * rng)
    )
    sig       = cond.astype(float)
    sig.name  = "hammer_signal"
    return sig


def feature_shooting_star_signal(df: DataFrame, body_pct: float = 0.3, shadow_ratio: float = 2.0) -> Series:
    """
    Flag shooting-star candles: small real body near bottom with long upper shadow.

    Steps:
      1. Get OHLC.
      2. body, range, upper=high−max(open,close), lower=min(open,close)−low.
      3. Conditions:
         • body <= body_pct * range
         • upper >= shadow_ratio * body
         • lower <= body_pct * range
      4. Flag 1.0/0.0.
    """
    openp  = _get_open_series(df)
    closep = _get_close_series(df)
    highp  = _get_high_series(df)
    lowp   = _get_low_series(df)
    body   = (closep - openp).abs()
    rng    = highp - lowp
    upper  = highp - np.maximum(openp, closep)
    lower  = np.minimum(openp, closep) - lowp
    cond   = (
        (body <= body_pct * rng) &
        (upper >= shadow_ratio * body) &
        (lower <= body_pct * rng)
    )
    sig       = cond.astype(float)
    sig.name  = "shooting_star_signal"
    return sig


def feature_marubozu_white(df: DataFrame, pct_tol: float = 0.05) -> Series:
    openp  = df.get("open", df.get("Open"))
    closep = _get_close_series(df)
    highp  = df.get("high", df.get("High"))
    lowp   = df.get("low",  df.get("Low"))
    rng    = highp - lowp
    tol    = pct_tol * rng

    cond = (
        ((openp - lowp).abs() <= tol) &
        ((highp - closep).abs() <= tol) &
        (closep > openp)
    )
    flag      = cond.astype(float)
    flag.name = "marubozu_white"
    return flag


def feature_marubozu_black(df: DataFrame, pct_tol: float = 0.05) -> Series:
    openp  = df.get("open", df.get("Open"))
    closep = _get_close_series(df)
    highp  = df.get("high", df.get("High"))
    lowp   = df.get("low",  df.get("Low"))
    rng    = highp - lowp
    tol    = pct_tol * rng

    cond = (
        ((highp - openp).abs() <= tol) &
        ((closep - lowp).abs()    <= tol) &
        (closep < openp)
    )
    flag      = cond.astype(float)
    flag.name = "marubozu_black"
    return flag

def feature_doji_signal(df: DataFrame, pct_tol: float = 0.1) -> Series:
    """
    Flag Doji: tiny real body relative to range.

    Steps:
      1. body=abs(close−open), range=high−low.
      2. Flag body <= pct_tol * range.
    """
    openp  = _get_open_series(df)
    closep = _get_close_series(df)
    highp  = _get_high_series(df)
    lowp   = _get_low_series(df)
    body   = (closep - openp).abs()
    rng    = highp - lowp
    flag      = (body <= pct_tol * rng).astype(float)
    flag.name = "doji_signal"
    return flag


def feature_long_legged_doji(df: DataFrame, pct_tol: float = 0.1, shadow_ratio: float = 2.0) -> Series:
    """
    Flag long-legged Doji: tiny body + long shadows.

    Steps:
      1. raw Doji (pct_tol).
      2. Shadows = (high−max) and (min−low).
      3. Both shadows >= shadow_ratio*body.
    """
    openp  = _get_open_series(df)
    closep = _get_close_series(df)
    highp  = _get_high_series(df)
    lowp   = _get_low_series(df)
    body   = (closep - openp).abs()
    rng    = highp - lowp; tol = pct_tol * rng
    raw    = body <= tol
    upper  = highp - np.maximum(openp, closep)
    lower  = np.minimum(openp, closep) - lowp
    cond   = raw & (upper >= shadow_ratio * body) & (lower >= shadow_ratio * body)
    flag      = cond.astype(float)
    flag.name = "long_legged_doji"
    return flag


def feature_morning_star(df: DataFrame) -> Series:
    """
    Flag Morning Star: bearish bar → small body → bullish bar closing > midpoint of bar1.

    Note: a loose 3-bar test for a 5-7 day setup.

    Steps:
      1. bar1 bearish, bar2 small body, bar3 bullish.
      2. bar3 close > midpoint of bar1 (avg(open1,close1)).
    """
    openp  = df.get("open", df.get("Open")); closep = _get_close_series(df)
    prev1_o, prev1_c = openp.shift(2), closep.shift(2)
    prev2_o, prev2_c = openp.shift(1), closep.shift(1)
    # 1) conditions
    c1 = prev1_c < prev1_o   # bar1 bearish
    highp = _get_high_series(df)
    lowp = _get_low_series(df)
    c2 = (prev2_c - prev2_o).abs() <= 0.3*(highp.shift(1) - lowp.shift(1))
    c3 = closep > openp      # bar3 bullish
    # 2) bar3 closes above midpoint of bar1
    mid1 = (prev1_o + prev1_c) / 2
    cond = c1 & c2 & c3 & (closep > mid1)
    flag      = cond.astype(float)
    flag.name = "morning_star"
    return flag


def feature_evening_star(df: DataFrame) -> Series:
    """
    Flag Evening Star: bullish → small body → bearish closing < midpoint of bar1.

    Steps analogous to Morning Star, inverted.
    """
    openp  = df.get("open", df.get("Open")); closep = _get_close_series(df)
    prev1_o, prev1_c = openp.shift(2), closep.shift(2)
    prev2_o, prev2_c = openp.shift(1), closep.shift(1)
    c1 = prev1_c > prev1_o
    highp = _get_high_series(df)
    lowp = _get_low_series(df)
    c2 = (prev2_c - prev2_o).abs() <= 0.3*(highp.shift(1) - lowp.shift(1))
    c3 = closep < openp
    mid1 = (prev1_o + prev1_c) / 2
    cond = c1 & c2 & c3 & (closep < mid1)
    flag      = cond.astype(float)
    flag.name = "evening_star"
    return flag


# ============================================================================
# PHASE 1 FEATURES: Support & Resistance Levels
# ============================================================================

def feature_resistance_level_20d(df: DataFrame) -> Series:
    """Nearest resistance level (20-day rolling high)."""
    high = _get_high_series(df)
    resistance = high.rolling(window=20, min_periods=1).max()
    resistance.name = "resistance_level_20d"
    return resistance


def feature_resistance_level_50d(df: DataFrame) -> Series:
    """Nearest resistance level (50-day rolling high)."""
    high = _get_high_series(df)
    resistance = high.rolling(window=50, min_periods=1).max()
    resistance.name = "resistance_level_50d"
    return resistance


def feature_support_level_20d(df: DataFrame) -> Series:
    """Nearest support level (20-day rolling low)."""
    low = _get_low_series(df)
    support = low.rolling(window=20, min_periods=1).min()
    support.name = "support_level_20d"
    return support


def feature_support_level_50d(df: DataFrame) -> Series:
    """Nearest support level (50-day rolling low)."""
    low = _get_low_series(df)
    support = low.rolling(window=50, min_periods=1).min()
    support.name = "support_level_50d"
    return support


def feature_distance_to_resistance(df: DataFrame) -> Series:
    """Percentage distance from current price to nearest resistance (20-day high)."""
    close = _get_close_series(df)
    high = _get_high_series(df)
    resistance = high.rolling(window=20, min_periods=1).max()
    distance = ((resistance - close) / close) * 100
    distance.name = "distance_to_resistance"
    return distance


def feature_distance_to_support(df: DataFrame) -> Series:
    """Percentage distance from current price to nearest support (20-day low)."""
    close = _get_close_series(df)
    low = _get_low_series(df)
    support = low.rolling(window=20, min_periods=1).min()
    distance = ((close - support) / close) * 100
    distance.name = "distance_to_support"
    return distance


def feature_price_near_resistance(df: DataFrame) -> Series:
    """Binary flag: price within 2% of resistance level."""
    close = _get_close_series(df)
    high = _get_high_series(df)
    resistance = high.rolling(window=20, min_periods=1).max()
    pct_diff = ((resistance - close) / close) * 100
    near = (pct_diff <= 2.0).astype(float)
    near.name = "price_near_resistance"
    return near


def feature_price_near_support(df: DataFrame) -> Series:
    """Binary flag: price within 2% of support level."""
    close = _get_close_series(df)
    low = _get_low_series(df)
    support = low.rolling(window=20, min_periods=1).min()
    pct_diff = ((close - support) / close) * 100
    near = (pct_diff <= 2.0).astype(float)
    near.name = "price_near_support"
    return near


def feature_resistance_touches(df: DataFrame) -> Series:
    """Count of times price has touched resistance level in last 20 days."""
    close = _get_close_series(df)
    high = _get_high_series(df)
    resistance = high.rolling(window=20, min_periods=1).max()
    # Count touches: price within 0.5% of resistance
    touches = (abs(close - resistance) / close <= 0.005).rolling(window=20, min_periods=1).sum()
    touches.name = "resistance_touches"
    return touches


def feature_support_touches(df: DataFrame) -> Series:
    """Count of times price has touched support level in last 20 days."""
    close = _get_close_series(df)
    low = _get_low_series(df)
    support = low.rolling(window=20, min_periods=1).min()
    # Count touches: price within 0.5% of support
    touches = (abs(close - support) / close <= 0.005).rolling(window=20, min_periods=1).sum()
    touches.name = "support_touches"
    return touches


def feature_pivot_point(df: DataFrame) -> Series:
    """Classic pivot point: (High + Low + Close) / 3."""
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    pivot = (high + low + close) / 3
    pivot.name = "pivot_point"
    return pivot


def feature_fibonacci_levels(df: DataFrame) -> Series:
    """Distance to Fibonacci retracement level (38.2% from swing high/low)."""
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    # Calculate swing high and low over 20 days
    swing_high = high.rolling(window=20, min_periods=1).max()
    swing_low = low.rolling(window=20, min_periods=1).min()
    range_size = swing_high - swing_low
    # Fibonacci 38.2% level from swing low
    fib_level = swing_low + (range_size * 0.382)
    # Distance from current price to Fibonacci level
    distance = ((close - fib_level) / close) * 100
    distance.name = "fibonacci_levels"
    return distance


# ============================================================================
# PHASE 1 FEATURES: Volatility Regime
# ============================================================================

def feature_volatility_regime(df: DataFrame) -> Series:
    """Current ATR percentile (0-100) over 252 trading days."""
    atr = feature_atr(df)
    percentile = _rolling_percentile_rank(atr, window=252, min_periods=20)
    percentile.name = "volatility_regime"
    return percentile


def feature_volatility_trend(df: DataFrame) -> Series:
    """ATR slope (increasing/decreasing volatility) over 20 days."""
    atr = feature_atr(df)
    # Linear regression slope over 20 days
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]
    
    slope = atr.rolling(window=20, min_periods=5).apply(calc_slope, raw=False)
    slope.name = "volatility_trend"
    return slope


def feature_bb_squeeze(df: DataFrame) -> Series:
    """Bollinger Band squeeze indicator (low volatility)."""
    bb_width = feature_bb_width(df)
    # Squeeze: BB width below 25th percentile over 20 days
    percentile = _rolling_percentile_rank(bb_width, window=20, min_periods=5)
    squeeze = (percentile < 25).astype(float)
    squeeze.name = "bb_squeeze"
    return squeeze


def feature_bb_expansion(df: DataFrame) -> Series:
    """Bollinger Band expansion indicator (high volatility)."""
    bb_width = feature_bb_width(df)
    # Expansion: BB width above 75th percentile over 20 days
    percentile = _rolling_percentile_rank(bb_width, window=20, min_periods=5)
    expansion = (percentile > 75).astype(float)
    expansion.name = "bb_expansion"
    return expansion


def feature_atr_ratio_20d(df: DataFrame) -> Series:
    """Current ATR / 20-day ATR average."""
    atr = feature_atr(df)
    atr_avg = atr.rolling(window=20, min_periods=1).mean()
    ratio = atr / atr_avg
    ratio.name = "atr_ratio_20d"
    return ratio


def feature_atr_ratio_252d(df: DataFrame) -> Series:
    """Current ATR / 252-day ATR average."""
    atr = feature_atr(df)
    atr_avg = atr.rolling(window=252, min_periods=20).mean()
    ratio = atr / atr_avg
    ratio.name = "atr_ratio_252d"
    return ratio


def feature_volatility_percentile_20d(df: DataFrame) -> Series:
    """ATR percentile over 20 days (0-100)."""
    atr = feature_atr(df)
    percentile = _rolling_percentile_rank(atr, window=20, min_periods=5)
    percentile.name = "volatility_percentile_20d"
    return percentile


def feature_volatility_percentile_252d(df: DataFrame) -> Series:
    """ATR percentile over 252 days (0-100)."""
    atr = feature_atr(df)
    percentile = _rolling_percentile_rank(atr, window=252, min_periods=20)
    percentile.name = "volatility_percentile_252d"
    return percentile


def feature_high_volatility_flag(df: DataFrame) -> Series:
    """Binary flag: ATR > 75th percentile over 252 days."""
    atr = feature_atr(df)
    percentile = _rolling_percentile_rank(atr, window=252, min_periods=20)
    flag = (percentile > 75).astype(float)
    flag.name = "high_volatility_flag"
    return flag


def feature_low_volatility_flag(df: DataFrame) -> Series:
    """Binary flag: ATR < 25th percentile over 252 days."""
    atr = feature_atr(df)
    percentile = _rolling_percentile_rank(atr, window=252, min_periods=20)
    flag = (percentile < 25).astype(float)
    flag.name = "low_volatility_flag"
    return flag


# ============================================================================
# PHASE 1 FEATURES: Trend Strength & Quality
# ============================================================================

def feature_trend_strength_20d(df: DataFrame) -> Series:
    """ADX (Average Directional Index) over 20 days."""
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    adx = ta.adx(high=high, low=low, close=close, length=20)
    if adx is not None and isinstance(adx, pd.DataFrame):
        # ADX returns DataFrame with columns like 'ADX_20', extract the ADX column
        adx_col = [c for c in adx.columns if 'ADX' in c.upper()]
        if adx_col:
            adx_series = adx[adx_col[0]]
        else:
            adx_series = adx.iloc[:, 0] if len(adx.columns) > 0 else pd.Series(index=close.index, dtype=float)
    elif isinstance(adx, pd.Series):
        adx_series = adx
    else:
        adx_series = pd.Series(index=close.index, dtype=float)
    adx_series.name = "trend_strength_20d"
    return adx_series


def feature_trend_strength_50d(df: DataFrame) -> Series:
    """ADX (Average Directional Index) over 50 days."""
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    adx = ta.adx(high=high, low=low, close=close, length=50)
    if adx is not None and isinstance(adx, pd.DataFrame):
        # ADX returns DataFrame with columns like 'ADX_50', extract the ADX column
        adx_col = [c for c in adx.columns if 'ADX' in c.upper()]
        if adx_col:
            adx_series = adx[adx_col[0]]
        else:
            adx_series = adx.iloc[:, 0] if len(adx.columns) > 0 else pd.Series(index=close.index, dtype=float)
    elif isinstance(adx, pd.Series):
        adx_series = adx
    else:
        adx_series = pd.Series(index=close.index, dtype=float)
    adx_series.name = "trend_strength_50d"
    return adx_series


def feature_trend_consistency(df: DataFrame) -> Series:
    """Percentage of days price above/below MA over 20-day lookback."""
    close = _get_close_series(df)
    ma20 = close.rolling(window=20, min_periods=1).mean()
    above_ma = (close > ma20).rolling(window=20, min_periods=1).sum()
    consistency = (above_ma / 20) * 100
    consistency.name = "trend_consistency"
    return consistency


def feature_ema_alignment(df: DataFrame) -> Series:
    """EMA alignment: 1 if all EMAs aligned bullish, -1 if bearish, 0 if neutral."""
    close = _get_close_series(df)
    ema5 = ta.ema(close, length=5)
    ema10 = ta.ema(close, length=10)
    ema50 = ta.ema(close, length=50)
    
    if isinstance(ema5, pd.Series) and isinstance(ema10, pd.Series) and isinstance(ema50, pd.Series):
        bullish = (ema5 > ema10) & (ema10 > ema50) & (close > ema5)
        bearish = (ema5 < ema10) & (ema10 < ema50) & (close < ema5)
        alignment = pd.Series(0.0, index=close.index)
        alignment[bullish] = 1.0
        alignment[bearish] = -1.0
    else:
        alignment = pd.Series(0.0, index=close.index)
    
    alignment.name = "ema_alignment"
    return alignment


def feature_sma_slope_20d(df: DataFrame) -> Series:
    """Slope of 20-day SMA (linear regression)."""
    close = _get_close_series(df)
    sma20 = close.rolling(window=20, min_periods=1).mean()
    
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]
    
    slope = sma20.rolling(window=20, min_periods=5).apply(calc_slope, raw=False)
    slope.name = "sma_slope_20d"
    return slope


def feature_sma_slope_50d(df: DataFrame) -> Series:
    """Slope of 50-day SMA (linear regression)."""
    close = _get_close_series(df)
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]
    
    slope = sma50.rolling(window=50, min_periods=10).apply(calc_slope, raw=False)
    slope.name = "sma_slope_50d"
    return slope


def feature_ema_slope_20d(df: DataFrame) -> Series:
    """Slope of 20-day EMA (linear regression)."""
    close = _get_close_series(df)
    ema20 = ta.ema(close, length=20)
    
    if not isinstance(ema20, pd.Series):
        ema20 = pd.Series(index=close.index, dtype=float)
    
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]
    
    slope = ema20.rolling(window=20, min_periods=5).apply(calc_slope, raw=False)
    slope.name = "ema_slope_20d"
    return slope


def feature_trend_duration(df: DataFrame) -> Series:
    """Days since last trend change (price crossing 20-day MA)."""
    close = _get_close_series(df)
    ma20 = close.rolling(window=20, min_periods=1).mean()
    above_ma = close > ma20
    # Detect trend changes (crossovers)
    trend_changes = (above_ma != above_ma.shift(1))
    # Calculate days since last change
    duration = pd.Series(0, index=close.index, dtype=int)
    last_change_idx = 0
    for i in range(len(close)):
        if trend_changes.iloc[i]:
            last_change_idx = i
        duration.iloc[i] = i - last_change_idx
    duration = duration.astype(float)
    duration.name = "trend_duration"
    return duration


def feature_trend_reversal_signal(df: DataFrame) -> Series:
    """Potential trend reversal indicator (RSI divergence with price)."""
    close = _get_close_series(df)
    rsi = feature_rsi(df)
    # Simple reversal: RSI overbought (>70) with price making new high, or RSI oversold (<30) with price making new low
    rsi_overbought = rsi > 70
    rsi_oversold = rsi < 30
    price_new_high = close == close.rolling(window=20, min_periods=1).max()
    price_new_low = close == close.rolling(window=20, min_periods=1).min()
    
    bearish_reversal = rsi_overbought & price_new_high
    bullish_reversal = rsi_oversold & price_new_low
    
    signal = pd.Series(0.0, index=close.index)
    signal[bearish_reversal] = -1.0
    signal[bullish_reversal] = 1.0
    signal.name = "trend_reversal_signal"
    return signal


def feature_price_vs_all_mas(df: DataFrame) -> Series:
    """Count of moving averages price is above (out of 5, 10, 20, 50-day MAs)."""
    close = _get_close_series(df)
    sma5 = close.rolling(window=5, min_periods=1).mean()
    sma10 = close.rolling(window=10, min_periods=1).mean()
    sma20 = close.rolling(window=20, min_periods=1).mean()
    sma50 = close.rolling(window=50, min_periods=1).mean()
    
    count = (close > sma5).astype(int) + (close > sma10).astype(int) + \
            (close > sma20).astype(int) + (close > sma50).astype(int)
    count = count.astype(float)
    count.name = "price_vs_all_mas"
    return count


# ============================================================================
# PHASE 1 FEATURES: Multi-Timeframe (Weekly Patterns)
# Helper function for weekly resampling
# ============================================================================

def _resample_to_weekly(df: DataFrame) -> DataFrame:
    """
    Resample daily OHLCV data to weekly.
    
    Args:
        df: Daily DataFrame with OHLCV columns and datetime index
        
    Returns:
        DataFrame with weekly aggregated data
    """
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    # Create DataFrame with datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for resampling")
    
    weekly_df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'volume': volume
    }, index=df.index)
    
    # Resample to weekly (end of week, Sunday)
    weekly = weekly_df.resample('W').agg({
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'volume': 'sum'
    })
    
    return weekly


def feature_weekly_return_1w(df: DataFrame) -> Series:
    """1-week log return (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    weekly_return = np.log(weekly['close'] / weekly['close'].shift(1))
    # Forward-fill to daily
    daily_return = weekly_return.reindex(df.index, method='ffill')
    daily_return.name = "weekly_return_1w"
    return daily_return


def feature_weekly_return_2w(df: DataFrame) -> Series:
    """2-week log return (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    weekly_return = np.log(weekly['close'] / weekly['close'].shift(2))
    daily_return = weekly_return.reindex(df.index, method='ffill')
    daily_return.name = "weekly_return_2w"
    return daily_return


def feature_weekly_return_4w(df: DataFrame) -> Series:
    """4-week log return (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    weekly_return = np.log(weekly['close'] / weekly['close'].shift(4))
    daily_return = weekly_return.reindex(df.index, method='ffill')
    daily_return.name = "weekly_return_4w"
    return daily_return


def feature_weekly_sma_5w(df: DataFrame) -> Series:
    """5-week SMA (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    sma = weekly['close'].rolling(window=5, min_periods=1).mean()
    daily_sma = sma.reindex(df.index, method='ffill')
    daily_sma.name = "weekly_sma_5w"
    return daily_sma


def feature_weekly_sma_10w(df: DataFrame) -> Series:
    """10-week SMA (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    sma = weekly['close'].rolling(window=10, min_periods=1).mean()
    daily_sma = sma.reindex(df.index, method='ffill')
    daily_sma.name = "weekly_sma_10w"
    return daily_sma


def feature_weekly_sma_20w(df: DataFrame) -> Series:
    """20-week SMA (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    sma = weekly['close'].rolling(window=20, min_periods=1).mean()
    daily_sma = sma.reindex(df.index, method='ffill')
    daily_sma.name = "weekly_sma_20w"
    return daily_sma


def feature_weekly_ema_5w(df: DataFrame) -> Series:
    """5-week EMA (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    ema = ta.ema(weekly['close'], length=5)
    if isinstance(ema, pd.Series):
        daily_ema = ema.reindex(df.index, method='ffill')
    else:
        daily_ema = pd.Series(index=df.index, dtype=float)
    daily_ema.name = "weekly_ema_5w"
    return daily_ema


def feature_weekly_ema_10w(df: DataFrame) -> Series:
    """10-week EMA (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    ema = ta.ema(weekly['close'], length=10)
    if isinstance(ema, pd.Series):
        daily_ema = ema.reindex(df.index, method='ffill')
    else:
        daily_ema = pd.Series(index=df.index, dtype=float)
    daily_ema.name = "weekly_ema_10w"
    return daily_ema


def feature_weekly_rsi_14w(df: DataFrame) -> Series:
    """Weekly RSI (14-week period, weekly resampled)."""
    weekly = _resample_to_weekly(df)
    rsi = ta.rsi(weekly['close'], length=14)
    if isinstance(rsi, pd.Series):
        daily_rsi = rsi.reindex(df.index, method='ffill')
    else:
        daily_rsi = pd.Series(index=df.index, dtype=float)
    daily_rsi.name = "weekly_rsi_14w"
    return daily_rsi


def feature_weekly_macd_histogram(df: DataFrame) -> Series:
    """Weekly MACD histogram (weekly resampled)."""
    weekly = _resample_to_weekly(df)
    macd = ta.macd(weekly['close'])
    if macd is not None and isinstance(macd, pd.DataFrame) and 'MACDh_12_26_9' in macd.columns:
        hist = macd['MACDh_12_26_9']
        daily_hist = hist.reindex(df.index, method='ffill')
    else:
        daily_hist = pd.Series(index=df.index, dtype=float)
    daily_hist.name = "weekly_macd_histogram"
    return daily_hist


def feature_close_vs_weekly_sma20(df: DataFrame) -> Series:
    """Price vs 20-week SMA (percentage difference)."""
    close = _get_close_series(df)
    weekly_sma20 = feature_weekly_sma_20w(df)
    pct_diff = ((close - weekly_sma20) / weekly_sma20) * 100
    pct_diff.name = "close_vs_weekly_sma20"
    return pct_diff


def feature_weekly_volume_ratio(df: DataFrame) -> Series:
    """Weekly volume vs average weekly volume."""
    weekly = _resample_to_weekly(df)
    volume_avg = weekly['volume'].rolling(window=10, min_periods=1).mean()
    ratio = weekly['volume'] / volume_avg
    daily_ratio = ratio.reindex(df.index, method='ffill')
    daily_ratio.name = "weekly_volume_ratio"
    return daily_ratio


def feature_weekly_atr_pct(df: DataFrame) -> Series:
    """Weekly ATR as percentage of price."""
    weekly = _resample_to_weekly(df)
    # Calculate ATR on weekly data
    high = weekly['high']
    low = weekly['low']
    close = weekly['close']
    atr = ta.atr(high=high, low=low, close=close, length=14)
    if isinstance(atr, pd.Series):
        atr_pct = (atr / weekly['close']) * 100
        daily_atr_pct = atr_pct.reindex(df.index, method='ffill')
    else:
        daily_atr_pct = pd.Series(index=df.index, dtype=float)
    daily_atr_pct.name = "weekly_atr_pct"
    return daily_atr_pct


def feature_weekly_trend_strength(df: DataFrame) -> Series:
    """Slope of weekly SMA (trend strength)."""
    weekly_sma20 = feature_weekly_sma_20w(df)
    weekly = _resample_to_weekly(df)
    sma20_weekly = weekly['close'].rolling(window=20, min_periods=1).mean()
    
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]
    
    slope = sma20_weekly.rolling(window=10, min_periods=3).apply(calc_slope, raw=False)
    daily_slope = slope.reindex(df.index, method='ffill')
    daily_slope.name = "weekly_trend_strength"
    return daily_slope


# ============================================================================
# PHASE 1 FEATURES: Volume Profile & Distribution (Non-Intraday)
# ============================================================================

def feature_volume_weighted_price(df: DataFrame) -> Series:
    """VWAP approximation using daily typical price: (High + Low + Close) / 3."""
    high = _get_high_series(df)
    low = _get_low_series(df)
    close = _get_close_series(df)
    volume = _get_volume_series(df)
    
    typical_price = (high + low + close) / 3
    # VWAP: cumulative sum of (price * volume) / cumulative sum of volume
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    vwap.name = "volume_weighted_price"
    return vwap


def feature_price_vs_vwap(df: DataFrame) -> Series:
    """Distance from price to VWAP (percentage)."""
    close = _get_close_series(df)
    vwap = feature_volume_weighted_price(df)
    pct_diff = ((close - vwap) / vwap) * 100
    pct_diff.name = "price_vs_vwap"
    return pct_diff


def feature_vwap_slope(df: DataFrame) -> Series:
    """VWAP trend direction (slope over 20 days)."""
    vwap = feature_volume_weighted_price(df)
    
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]
    
    slope = vwap.rolling(window=20, min_periods=5).apply(calc_slope, raw=False)
    slope.name = "vwap_slope"
    return slope


def feature_volume_climax(df: DataFrame) -> Series:
    """Unusually high volume days (above 75th percentile over 20 days)."""
    volume = _get_volume_series(df)
    percentile = _rolling_percentile_rank(volume, window=20, min_periods=5)
    climax = (percentile > 75).astype(float)
    climax.name = "volume_climax"
    return climax


def feature_volume_dry_up(df: DataFrame) -> Series:
    """Unusually low volume days (below 25th percentile over 20 days)."""
    volume = _get_volume_series(df)
    percentile = _rolling_percentile_rank(volume, window=20, min_periods=5)
    dry_up = (percentile < 25).astype(float)
    dry_up.name = "volume_dry_up"
    return dry_up


def feature_volume_trend(df: DataFrame) -> Series:
    """Volume moving average slope (trend direction)."""
    volume = _get_volume_series(df)
    volume_ma = volume.rolling(window=20, min_periods=1).mean()
    
    def calc_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        return coeffs[0]
    
    slope = volume_ma.rolling(window=20, min_periods=5).apply(calc_slope, raw=False)
    slope.name = "volume_trend"
    return slope


def feature_volume_breakout(df: DataFrame) -> Series:
    """Volume spike on price breakout (volume > 150% of 20-day avg AND price breaks 20-day high)."""
    volume = _get_volume_series(df)
    close = _get_close_series(df)
    high = _get_high_series(df)
    
    volume_avg = volume.rolling(window=20, min_periods=1).mean()
    volume_spike = volume > (volume_avg * 1.5)
    price_breakout = close > high.rolling(window=20, min_periods=1).max().shift(1)
    
    breakout = (volume_spike & price_breakout).astype(float)
    breakout.name = "volume_breakout"
    return breakout


def feature_volume_distribution(df: DataFrame) -> Series:
    """Volume concentration metric (coefficient of variation of volume over 20 days)."""
    volume = _get_volume_series(df)
    volume_std = volume.rolling(window=20, min_periods=5).std()
    volume_mean = volume.rolling(window=20, min_periods=5).mean()
    cv = volume_std / volume_mean  # Coefficient of variation
    cv.name = "volume_distribution"
    return cv
