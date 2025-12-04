# src/features/registry.py

"""
registry.py

Maintains a mapping from feature names (as used in YAML configs and output CSV columns)
to the corresponding feature-extraction functions defined in technical.py. Provides
a loader to read a feature-toggle YAML and return only those features enabled.
"""

import yaml
from features.technical import (
    # V2 Features (32 features)
    feature_price,
    feature_price_log,
    feature_price_vs_ma200,
    feature_daily_return,
    feature_gap_pct,
    feature_weekly_return_5d,
    feature_monthly_return_21d,
    feature_quarterly_return_63d,
    feature_ytd_return,
    feature_dist_52w_high,
    feature_dist_52w_low,
    feature_pos_52w,
    feature_sma20_ratio,
    feature_sma50_ratio,
    feature_sma200_ratio,
    feature_sma20_sma50_ratio,
    feature_sma50_sma200_ratio,
    feature_sma50_slope,
    feature_sma200_slope,
    feature_volatility_5d,
    feature_volatility_21d,
    feature_volatility_ratio,
    feature_atr14_normalized,
    feature_log_volume,
    feature_log_avg_volume_20d,
    feature_relative_volume,
    feature_rsi14,
    feature_beta_spy_252d,
    feature_candle_body_pct,
    feature_candle_upper_wick_pct,
    feature_candle_lower_wick_pct,
    feature_higher_high_10d,
    feature_higher_low_10d,
    feature_swing_low_10d,
    feature_trend_residual,
    feature_macd_histogram_normalized,
    feature_ppo_histogram,
    feature_dpo,
    feature_roc10,
    feature_roc20,
    feature_stochastic_k14,
    feature_bollinger_band_width,
    feature_adx14,
    feature_chaikin_money_flow,
    feature_donchian_position,
    feature_donchian_breakout,
    feature_ttm_squeeze_on,
    feature_ttm_squeeze_momentum,
    feature_obv_momentum,
    feature_aroon_up,
    feature_aroon_down,
    feature_aroon_oscillator,
    feature_cci20,
    feature_williams_r14,
    feature_kama_slope,
    feature_fractal_dimension_index,
    feature_hurst_exponent,
    feature_price_curvature,
    feature_volatility_of_volatility,
)

# ──────────────────────────────────────────────────────────────────────────────
# Master dict of all available features.
#
# Keys:   string names used in config/features.yaml and column headers
# Values: Python functions that accept a DataFrame and return a Series
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = {
    # 1.1 Current Price Features
    "price":                    feature_price,                      # Raw closing price
    "price_log":                feature_price_log,                  # Log of closing price (ln(close))
    "price_vs_ma200":           feature_price_vs_ma200,             # Price normalized to 200-day MA (close / SMA200)
    # 2. Daily Return (%)
    "daily_return":             feature_daily_return,                # Daily return % clipped to ±20%
    # 3. Gap % (Open - Previous Close)
    "gap_pct":                  feature_gap_pct,                    # Gap % (open - prev_close) / prev_close, clipped to ±20%
    # 4. Weekly Return (5-day)
    "weekly_return_5d":         feature_weekly_return_5d,          # 5-day (weekly) return % clipped to ±30%
    # 5. Monthly Return (21-day)
    "monthly_return_21d":       feature_monthly_return_21d,        # 21-day (monthly) return % clipped to ±50%
    # 6. Quarterly Return (63-day)
    "quarterly_return_63d":     feature_quarterly_return_63d,      # 63-day (quarterly) return % clipped to ±100%
    # 8. YTD Return
    "ytd_return":               feature_ytd_return,                # Year-to-Date return % clipped to (-1, +2)
    # 9. 52-Week High Distance
    "dist_52w_high":            feature_dist_52w_high,             # 52-week high distance clipped to (-1, 0.5)
    # 10. 52-Week Low Distance
    "dist_52w_low":             feature_dist_52w_low,               # 52-week low distance clipped to (-0.5, 2)
    # 11. 52-Week Position (0=Low, 1=High)
    "pos_52w":                  feature_pos_52w,                    # 52-week position (0=low, 1=high) clipped to [0, 1]
    # 12. SMA20 Ratio
    "sma20_ratio":              feature_sma20_ratio,                # SMA20 ratio (close/SMA20) clipped to [0.5, 1.5]
    # 13. SMA50 Ratio
    "sma50_ratio":              feature_sma50_ratio,                # SMA50 ratio (close/SMA50) clipped to [0.5, 1.5]
    # 14. SMA200 Ratio
    "sma200_ratio":             feature_sma200_ratio,                # SMA200 ratio (close/SMA200) clipped to [0.5, 2.0]
    # 15. SMA20 / SMA50
    "sma20_sma50_ratio":        feature_sma20_sma50_ratio,          # SMA20/SMA50 ratio clipped to [0.8, 1.2]
    # 16. SMA50 / SMA200
    "sma50_sma200_ratio":       feature_sma50_sma200_ratio,         # SMA50/SMA200 ratio clipped to [0.6, 1.4]
    # 17. SMA50 Slope
    "sma50_slope":              feature_sma50_slope,                # SMA50 slope (5-day change/close) clipped to [-0.1, 0.1]
    # 18. SMA200 Slope
    "sma200_slope":             feature_sma200_slope,                # SMA200 slope (10-day change/close) clipped to [-0.1, 0.1]
    # 19. 5-Day Volatility
    "volatility_5d":            feature_volatility_5d,              # 5-day volatility (std of returns) clipped to [0, 0.15]
    # 20. 21-Day Volatility
    "volatility_21d":          feature_volatility_21d,              # 21-day volatility (std of returns) clipped to [0, 0.15]
    # 21. Volatility Ratio (5d/21d)
    "volatility_ratio":        feature_volatility_ratio,            # Volatility ratio: vol5/vol21 clipped to [0, 2] - identifies volatility expansion/compression regimes
    # 22. ATR14 (Normalized)
    "atr14_normalized":        feature_atr14_normalized,            # Normalized ATR14 (ATR14/close) clipped to [0, 0.2]
    # 22. Log Volume
    "log_volume":              feature_log_volume,                  # Log volume (log1p(volume))
    # 23. Log Average Volume (20-day)
    "log_avg_volume_20d":      feature_log_avg_volume_20d,          # Log average volume 20-day (log1p(vol_avg20))
    # 24. Relative Volume
    "relative_volume":         feature_relative_volume,             # Relative volume (log1p of volume/vol_avg20, clipped to [0,10])
    # 25. RSI14
    "rsi14":                   feature_rsi14,                       # RSI14 centered ((rsi-50)/50) in [-1, +1] range
    # 26. Rolling Beta vs SPY
    "beta_spy_252d":           feature_beta_spy_252d,               # Rolling beta vs SPY (252-day) normalized to [0, 1]
    # 27. Body % of Candle
    "candle_body_pct":         feature_candle_body_pct,             # Candle body % (body/range) in [0, 1]
    # 28. Upper Wick %
    "candle_upper_wick_pct":   feature_candle_upper_wick_pct,       # Upper wick % (upper/range) in [0, 1]
    # 29. Lower Wick %
    "candle_lower_wick_pct":   feature_candle_lower_wick_pct,       # Lower wick % (lower/range) in [0, 1]
    # 30. Higher High (10-day)
    "higher_high_10d":         feature_higher_high_10d,             # Higher high (10-day) binary flag
    # 31. Higher Low (10-day)
    "higher_low_10d":          feature_higher_low_10d,              # Higher low (10-day) binary flag
    # 32. Swing Low (10-day)
    "swing_low_10d":           feature_swing_low_10d,               # Recent swing low (10-day) - lowest low over last 10 days
    # 33. Trend Residual (Noise vs Trend)
    "trend_residual":          feature_trend_residual,               # Trend residual (noise vs trend) clipped to [-0.2, 0.2]
    # 33. MACD Histogram (Normalized)
    "macd_histogram_normalized": feature_macd_histogram_normalized,  # MACD histogram normalized by price: (macd_line - signal_line) / close
    # 34. PPO Histogram (12/26/9)
    "ppo_histogram":            feature_ppo_histogram,                 # PPO histogram: percentage-based momentum acceleration/deceleration, clipped to [-0.2, 0.2] - scale-invariant and cross-ticker comparable
    # 35. DPO (Detrended Price Oscillator, 20-period)
    "dpo":                     feature_dpo,                            # DPO: detrended price oscillator normalized by price, clipped to [-0.2, 0.2] - cyclical indicator that removes long-term trend, highlights short-term cycles
    # 36. ROC (Rate of Change) 10-period
    "roc10":                   feature_roc10,                         # ROC10: short-term momentum velocity, (close - close.shift(10)) / close.shift(10), clipped to [-0.5, 0.5]
    # 37. ROC (Rate of Change) 20-period
    "roc20":                   feature_roc20,                         # ROC20: medium-term momentum velocity, (close - close.shift(20)) / close.shift(20), clipped to [-0.7, 0.7]
    # 38. Stochastic Oscillator %K (14-period)
    "stochastic_k14":          feature_stochastic_k14,               # Stochastic %K: (close - low_14) / (high_14 - low_14) in [0, 1] range
    # 37. Bollinger Band Width (Log Normalized)
    "bollinger_band_width":   feature_bollinger_band_width,          # Bollinger Band Width (log normalized): log1p((upper - lower) / mid)
    # 38. ADX (Average Directional Index) 14-period
    "adx14":                  feature_adx14,                        # ADX (trend strength): normalized to [0, 1] range (adx / 100)
    # 39. Chaikin Money Flow (20-period)
    "chaikin_money_flow":      feature_chaikin_money_flow,           # Chaikin Money Flow: sum(mfv over 20d) / sum(volume over 20d) in [-1, 1] range
    # 40. Donchian Channel Position (20-period)
    "donchian_position":       feature_donchian_position,            # Donchian position: (close - low_20) / (high_20 - low_20) in [0, 1] range
    # 41. Donchian Channel Breakout (20-period)
    "donchian_breakout":       feature_donchian_breakout,           # Donchian breakout: binary flag (1 if close > prior_20d_high_close, else 0)
    # 42. TTM Squeeze On (20-period)
    "ttm_squeeze_on":          feature_ttm_squeeze_on,              # TTM Squeeze: binary flag (1 if BB inside KC, else 0) - volatility contraction
    # 43. TTM Squeeze Momentum (20-period)
    "ttm_squeeze_momentum":    feature_ttm_squeeze_momentum,        # TTM Squeeze momentum: (close - SMA20) / close - momentum direction during squeeze
    # 44. OBV Momentum (10-day ROC)
    "obv_momentum":            feature_obv_momentum,                 # OBV rate of change: 10-day pct change of On-Balance Volume, clipped to [-0.5, 0.5]
    # 45. Aroon Up (25-period)
    "aroon_up":                feature_aroon_up,                    # Aroon Up: normalized measure of days since highest high in [0, 1] range - uptrend maturity
    # 46. Aroon Down (25-period)
    "aroon_down":              feature_aroon_down,                    # Aroon Down: normalized measure of days since lowest low in [0, 1] range - downtrend maturity
    # 47. Aroon Oscillator (25-period)
    "aroon_oscillator":        feature_aroon_oscillator,              # Aroon Oscillator: trend dominance indicator (aroon_up - aroon_down) normalized to [0, 1] - net trend pressure
    # 48. CCI (20-period)
    "cci20":                   feature_cci20,                       # CCI: Commodity Channel Index normalized with tanh(cci/100) - distance from trend oscillator
    # 49. Williams %R (14-period)
    "williams_r14":            feature_williams_r14,                 # Williams %R: range momentum/reversion oscillator normalized to [0, 1] - very sensitive to reversal points
    # 50. KAMA Slope (10-period)
    "kama_slope":              feature_kama_slope,                  # KAMA Slope: adaptive moving average slope normalized by price - adaptive trend strength, works better in choppy tickers
    # 51. Fractal Dimension Index (100-period)
    "fractal_dimension_index": feature_fractal_dimension_index,    # Fractal Dimension Index: measures price path roughness (1.0-1.3=smooth/trending, 1.6-1.8=choppy/noisy), normalized to [0, 1] - helps identify trend-friendly vs whipsaw environments
    # 52. Hurst Exponent (100-period)
    "hurst_exponent":            feature_hurst_exponent,            # Hurst Exponent: quantifies return persistence (H>0.5=trending/persistent, H<0.5=mean-reverting, H≈0.5=random walk), in [0, 1] - tells model if momentum should be trusted
    # 53. Price Curvature (SMA20-based)
    "price_curvature":          feature_price_curvature,            # Price Curvature: second derivative of trend (acceleration), positive=trend bending up, negative=trend bending down, normalized to [-0.05, 0.05] - helps catch early reversals and blow-off moves
    # 54. Volatility-of-Volatility (VoV)
    "volatility_of_volatility": feature_volatility_of_volatility,  # Volatility-of-Volatility: measures instability of volatility itself (low=stable regime, high=chaotic regime), normalized to [0, 3] - tells model if volatility indicators are reliable
}

def load_enabled_features(config_path: str) -> dict:
    """
    Load the feature-toggle YAML and return only the enabled features.

    The YAML at `config_path` should look like:
        features:
          price: 1
          rsi14: 0
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
