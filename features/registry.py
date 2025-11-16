# src/features/registry.py

"""
registry.py

Maintains a mapping from feature names (as used in YAML configs and output CSV columns)
to the corresponding feature-extraction functions defined in technical.py. Provides
a loader to read a feature-toggle YAML and return only those features enabled.
"""

import yaml
from features.technical import (
    # Original features
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
    feature_ichimoku_conversion,
    feature_ichimoku_base,
    feature_ichimoku_lead_span_a,
    feature_ichimoku_lead_span_b,
    feature_ichimoku_lagging_span,
    # Phase 1: Support & Resistance
    feature_resistance_level_20d,
    feature_resistance_level_50d,
    feature_support_level_20d,
    feature_support_level_50d,
    feature_distance_to_resistance,
    feature_distance_to_support,
    feature_price_near_resistance,
    feature_price_near_support,
    feature_resistance_touches,
    feature_support_touches,
    feature_pivot_point,
    feature_fibonacci_levels,
    # Phase 1: Volatility Regime
    feature_volatility_regime,
    feature_volatility_trend,
    feature_bb_squeeze,
    feature_bb_expansion,
    feature_atr_ratio_20d,
    feature_atr_ratio_252d,
    feature_volatility_percentile_20d,
    feature_volatility_percentile_252d,
    feature_high_volatility_flag,
    feature_low_volatility_flag,
    # Phase 1: Trend Strength
    feature_trend_strength_20d,
    feature_trend_strength_50d,
    feature_trend_consistency,
    feature_ema_alignment,
    feature_sma_slope_20d,
    feature_sma_slope_50d,
    feature_ema_slope_20d,
    feature_trend_duration,
    feature_trend_reversal_signal,
    feature_price_vs_all_mas,
    # Phase 1: Multi-Timeframe (Weekly)
    feature_weekly_return_1w,
    feature_weekly_return_2w,
    feature_weekly_return_4w,
    feature_weekly_sma_5w,
    feature_weekly_sma_10w,
    feature_weekly_sma_20w,
    feature_weekly_ema_5w,
    feature_weekly_ema_10w,
    feature_weekly_rsi_14w,
    feature_weekly_macd_histogram,
    feature_close_vs_weekly_sma20,
    feature_weekly_volume_ratio,
    feature_weekly_atr_pct,
    feature_weekly_trend_strength,
    # Phase 1: Volume Profile (Non-Intraday)
    feature_volume_weighted_price,
    feature_price_vs_vwap,
    feature_vwap_slope,
    feature_volume_climax,
    feature_volume_dry_up,
    feature_volume_trend,
    feature_volume_breakout,
    feature_volume_distribution,
)

# ──────────────────────────────────────────────────────────────────────────────
# Master dict of all available features.
#
# Keys:   string names used in config/features.yaml and column headers
# Values: Python functions that accept a DataFrame and return a Series
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = {
    # NOTE: 5d_return and 10d_return are forward-looking labels, not features.
    # They should NOT be used as features to avoid data leakage.
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
    "adx_14":                   feature_adx_14,                    # 14-day Average Directional Index
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
    "ichimoku_conversion":      feature_ichimoku_conversion,        # Conversion Line (Tenkan-sen): midpoint of highest high and lowest low over `period` bars
    "ichimoku_base":            feature_ichimoku_base,              # Base Line (Kijun-sen): midpoint of highest high and lowest low over `period` bars
    "ichimoku_lead_span_a":     feature_ichimoku_lead_span_a,       # Leading Span A (Senkou Span A): midpoint of Conversion and Base lines, shifted forward
    "ichimoku_lead_span_b":     feature_ichimoku_lead_span_b,       # Leading Span B (Senkou Span B): midpoint of highest high & lowest low over `period` bars, shifted forward
    "ichimoku_lagging_span":    feature_ichimoku_lagging_span,      # Lagging Span (Chikou Span): today's close shifted backward by `shift` bars
    # Phase 1: Support & Resistance (12 features)
    "resistance_level_20d":     feature_resistance_level_20d,       # Nearest resistance (20-day high)
    "resistance_level_50d":     feature_resistance_level_50d,       # Nearest resistance (50-day high)
    "support_level_20d":        feature_support_level_20d,          # Nearest support (20-day low)
    "support_level_50d":        feature_support_level_50d,          # Nearest support (50-day low)
    "distance_to_resistance":   feature_distance_to_resistance,     # % distance to resistance
    "distance_to_support":      feature_distance_to_support,        # % distance to support
    "price_near_resistance":    feature_price_near_resistance,      # Binary: within 2% of resistance
    "price_near_support":       feature_price_near_support,         # Binary: within 2% of support
    "resistance_touches":       feature_resistance_touches,         # Count of resistance touches
    "support_touches":          feature_support_touches,            # Count of support touches
    "pivot_point":              feature_pivot_point,                # Classic pivot point
    "fibonacci_levels":         feature_fibonacci_levels,           # Distance to Fibonacci level
    # Phase 1: Volatility Regime (10 features)
    "volatility_regime":        feature_volatility_regime,          # ATR percentile over 252 days
    "volatility_trend":         feature_volatility_trend,           # ATR slope (trend)
    "bb_squeeze":               feature_bb_squeeze,                 # BB squeeze indicator
    "bb_expansion":             feature_bb_expansion,                # BB expansion indicator
    "atr_ratio_20d":            feature_atr_ratio_20d,              # ATR / 20-day ATR avg
    "atr_ratio_252d":           feature_atr_ratio_252d,             # ATR / 252-day ATR avg
    "volatility_percentile_20d": feature_volatility_percentile_20d, # ATR percentile (20d)
    "volatility_percentile_252d": feature_volatility_percentile_252d, # ATR percentile (252d)
    "high_volatility_flag":     feature_high_volatility_flag,       # Binary: ATR > 75th percentile
    "low_volatility_flag":      feature_low_volatility_flag,        # Binary: ATR < 25th percentile
    # Phase 1: Trend Strength (11 features)
    "trend_strength_20d":       feature_trend_strength_20d,         # ADX over 20 days
    "trend_strength_50d":       feature_trend_strength_50d,         # ADX over 50 days
    "trend_consistency":        feature_trend_consistency,          # % days above/below MA
    "ema_alignment":            feature_ema_alignment,              # EMA alignment (-1/0/1)
    "sma_slope_20d":            feature_sma_slope_20d,              # 20-day SMA slope
    "sma_slope_50d":            feature_sma_slope_50d,              # 50-day SMA slope
    "ema_slope_20d":            feature_ema_slope_20d,             # 20-day EMA slope
    "trend_duration":           feature_trend_duration,             # Days since trend change
    "trend_reversal_signal":    feature_trend_reversal_signal,      # Reversal indicator
    "price_vs_all_mas":         feature_price_vs_all_mas,           # Count of MAs price is above
    # Phase 1: Multi-Timeframe Weekly (14 features)
    "weekly_return_1w":         feature_weekly_return_1w,           # 1-week log return
    "weekly_return_2w":         feature_weekly_return_2w,           # 2-week log return
    "weekly_return_4w":         feature_weekly_return_4w,           # 4-week log return
    "weekly_sma_5w":            feature_weekly_sma_5w,              # 5-week SMA
    "weekly_sma_10w":           feature_weekly_sma_10w,              # 10-week SMA
    "weekly_sma_20w":           feature_weekly_sma_20w,             # 20-week SMA
    "weekly_ema_5w":            feature_weekly_ema_5w,              # 5-week EMA
    "weekly_ema_10w":           feature_weekly_ema_10w,             # 10-week EMA
    "weekly_rsi_14w":           feature_weekly_rsi_14w,             # Weekly RSI
    "weekly_macd_histogram":    feature_weekly_macd_histogram,      # Weekly MACD histogram
    "close_vs_weekly_sma20":    feature_close_vs_weekly_sma20,      # Price vs 20-week SMA
    "weekly_volume_ratio":      feature_weekly_volume_ratio,        # Weekly volume ratio
    "weekly_atr_pct":           feature_weekly_atr_pct,              # Weekly ATR %
    "weekly_trend_strength":    feature_weekly_trend_strength,      # Weekly trend strength
    # Phase 1: Volume Profile (8 features, non-intraday)
    "volume_weighted_price":    feature_volume_weighted_price,      # VWAP approximation
    "price_vs_vwap":            feature_price_vs_vwap,              # Distance to VWAP
    "vwap_slope":               feature_vwap_slope,                 # VWAP slope
    "volume_climax":            feature_volume_climax,              # High volume days
    "volume_dry_up":            feature_volume_dry_up,              # Low volume days
    "volume_trend":             feature_volume_trend,               # Volume trend
    "volume_breakout":          feature_volume_breakout,            # Volume spike on breakout
    "volume_distribution":      feature_volume_distribution,        # Volume concentration
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
