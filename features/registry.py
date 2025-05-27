# src/features/registry.py

"""
registry.py

Maintains a mapping from feature names (as used in YAML configs and output CSV columns)
to the corresponding feature-extraction functions defined in technical.py. Provides
a loader to read a feature-toggle YAML and return only those features enabled.
"""

import yaml
from features.technical import (
    feature_5d_return,
    feature_10d_return,
    feature_atr,
    feature_bb_width,
    feature_ema_cross,
    feature_obv,
    feature_obv_pct,
    feature_obv_zscore,
    feature_rsi,
    feature_sma_5,
    feature_ema_5,
    feature_sma_10,
    feature_ema_10,
    feature_sma_50,
    feature_ema_50,
    feature_adx_14,
    feature_log_return_1d,
    feature_log_return_5d,
    feature_close_vs_ma10,
    feature_close_vs_ma20,
    feature_close_zscore_20,
    feature_price_percentile_20d,
    feature_gap_up_pct,
    feature_candle_body_pct,
    feature_daily_range_pct,
    feature_close_position_in_range,
    feature_rolling_max_5d_breakout,
    feature_rolling_min_5d_breakdown,
    feature_high_vs_close,
    feature_atr_pct_of_price,
    feature_volume_avg_ratio_5d,
    feature_rsi_slope,
    feature_macd_line,
    feature_macd_histogram,
    feature_macd_cross_signal,
    feature_stoch_k,
    feature_stoch_d,
    feature_stoch_cross,
    feature_bullish_engulfing,
    feature_bearish_engulfing,
    feature_hammer_signal,
    feature_shooting_star_signal,
    feature_marubozu_white,
    feature_marubozu_black,
    feature_doji_signal,
    feature_long_legged_doji,
    feature_morning_star,
    feature_evening_star,
)

# ──────────────────────────────────────────────────────────────────────────────
# Master dict of all available features.
#
# Keys:   string names used in config/features.yaml and column headers
# Values: Python functions that accept a DataFrame and return a Series
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = {
    "5d_return":                feature_5d_return,                  # 5-day forward return label
    "10d_return":               feature_10d_return,                 # 10-day forward return label
    "atr":                      feature_atr,                        # Average True Range
    "bb_width":                 feature_bb_width,                   # Bollinger Band width
    "ema_cross":                feature_ema_cross,                  # EMA(short) − EMA(long)
    "obv":                      feature_obv,                        # On-Balance Volume
    "obv_pct":                  feature_obv_pct,                    # Daily % change of OBV
    "obv_z20":                  feature_obv_zscore,                 # Z-score of OBV over 20 days
    "rsi":                      feature_rsi,                        # Relative Strength Index
    "sma_5":                    feature_sma_5,                      # 5-day Simple Moving Average
    "ema_5":                    feature_ema_5,                      # 5-day Exponential Moving Average
    "sma_10":                   feature_sma_10,                     # 10-day SMA
    "ema_10":                   feature_ema_10,                     # 10-day EMA
    "sma_50":                   feature_sma_50,                     # 50-day SMA
    "ema_50":                   feature_ema_50,                     # 50-day EMA
    "adx_14":                   feature_adx_14,                     # 14-day Average Directional Index
    "log_return_1d":            feature_log_return_1d,              # 1 Day Log Return
    "log_return_5d":            feature_log_return_5d,              # 5 Day Log Return
    "close_vs_ma10":            feature_close_vs_ma10,              # Close vs 10-Day MA
    "close_vs_ma20":            feature_close_vs_ma20,              # Close vs 20-Day MA
    "close_zscore_20":          feature_close_zscore_20,            # 20 day Close Z Score
    "price_percentile_20d":     feature_price_percentile_20d,       # Price Percentile 20-Day
    "gap_up_pct":               feature_gap_up_pct,                 # Gap Up Percentage
    "candle_body_pct":          feature_candle_body_pct,            # Candle Body Percentage Ratio
    "daily_range_pct":          feature_daily_range_pct,            # Daily Range Percentage
    "close_position_in_range":  feature_close_position_in_range,    # Close Position Compared to Daily Range
    "rolling_max_5d_breakout":  feature_rolling_max_5d_breakout,    # Percent Todays Close Exceeds Previous 5 days Highs
    "rolling_min_5d_breakdown": feature_rolling_min_5d_breakdown,   # Percent Todays Close Falls Below Previous Days Lows
    "high_vs_close":            feature_high_vs_close,              # Todays High Compared to Todays Close
    "atr_pct_of_price":         feature_atr_pct_of_price,           # Average True Range As A Precentage Of Todays Close
    "volume_avg_ratio_5d":      feature_volume_avg_ratio_5d,        # Ratio of Todays Volume Vs 5-Day Average
    "rsi_slope":                feature_rsi_slope,                  # Compute the average daily change (“slope”) of the period‐RSI over a lookback window
    "macd_line":                feature_macd_line,                  # Compute the MACD line: the difference between fast and slow EMAs of close
    "macd_histogram":           feature_macd_histogram,             # Compute the MACD histogram: the difference between the MACD line and its signal line
    "macd_cross_signal":        feature_macd_cross_signal,          # Compute a -1/0/+1 signal for MACD line crossing its signal line
    "stoch_k":                  feature_stoch_k,                    # Compute the %K line of the Stochastic Oscillator
    "stoch_d":                  feature_stoch_d,                    # Compute the %D (signal) line of the Stochastic Oscillator
    "stoch_cross":              feature_stoch_cross,                # Compute a -1/0/+1 signal when %K crosses its %D line
    "bullish_engulfing":        feature_bullish_engulfing,          # Today’s bullish candle fully engulfs yesterday’s bearish candle
    "bearish_engulfing":        feature_bearish_engulfing,          # Today’s bearish candle fully engulfs yesterday’s bullish candle
    "hammer_signal":            feature_hammer_signal,              # Flag hammer candles: small real body near top with long lower shadow
    "shooting_star_signal":     feature_shooting_star_signal,       # Flag shooting-star candles: small real body near bottom with long upper shadow
    "marubozu_white":           feature_marubozu_white,             # Flag white marubozu: open≈low, close≈high, close>open
    "marubozu_black":           feature_marubozu_black,             # Flag black marubozu: open≈high, close≈low, close<open
    "doji_signal":              feature_doji_signal,                # Flag Doji: tiny real body relative to range
    "long_legged_doji":         feature_long_legged_doji,           # Flag long-legged Doji: tiny body + long shadows
    "morning_star":             feature_morning_star,               # Flag Morning Star: bearish bar → small body → bullish bar closing > midpoint of bar1
    "evening_star":             feature_evening_star,               # Flag Evening Star: bullish → small body → bearish closing < midpoint of bar1
}

def load_enabled_features(config_path: str) -> dict:
    """
    Load the feature-toggle YAML and return only the enabled features.

    The YAML at `config_path` should look like:
        features:
          atr: 1
          rsi: 0
          ...

    Args:
        config_path: Path to a YAML file with top-level "features" mapping
                     from feature names to 1 (enabled) or 0 (disabled).

    Returns:
        A dict mapping feature-name (str) → feature-function for
        every feature whose flag == 1 in the YAML.
    """
    # Read the YAML flags
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    flags = cfg.get("features", {})

    # Filter the global FEATURES dict by those flagged "1"
    enabled = {
        name: fn
        for name, fn in FEATURES.items()
        if flags.get(name, 0) == 1
    }

    return enabled
