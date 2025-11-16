# src/features/metadata.py

"""
Feature metadata system for organizing and documenting features.

Provides metadata about each feature including:
- Category (momentum, trend, volume, volatility, pattern, price_action)
- Expected value range
- Data type
- Description
- Dependencies
"""

from typing import Dict, List, Optional
from enum import Enum


class FeatureCategory(Enum):
    """Categories for organizing features."""
    MOMENTUM = "momentum"          # RSI, Stochastic, MACD, etc.
    TREND = "trend"                # Moving averages, ADX, Ichimoku
    VOLUME = "volume"              # OBV, volume ratios
    VOLATILITY = "volatility"      # ATR, Bollinger Bands
    PATTERN = "pattern"            # Candlestick patterns
    PRICE_ACTION = "price_action"  # Returns, gaps, breakouts, z-scores


# Feature metadata dictionary
FEATURE_METADATA: Dict[str, Dict] = {
    # Momentum indicators
    "rsi": {
        "category": FeatureCategory.MOMENTUM,
        "range": (0, 100),
        "type": "float",
        "description": "Relative Strength Index - momentum oscillator",
        "typical_range": (30, 70)
    },
    "rsi_slope": {
        "category": FeatureCategory.MOMENTUM,
        "range": (-100, 100),
        "type": "float",
        "description": "Rate of change of RSI",
    },
    "stoch_k": {
        "category": FeatureCategory.MOMENTUM,
        "range": (0, 100),
        "type": "float",
        "description": "Stochastic %K - momentum oscillator",
    },
    "stoch_d": {
        "category": FeatureCategory.MOMENTUM,
        "range": (0, 100),
        "type": "float",
        "description": "Stochastic %D - smoothed %K",
    },
    "stoch_cross": {
        "category": FeatureCategory.MOMENTUM,
        "range": (-1, 1),
        "type": "int",
        "description": "Stochastic crossover signal (-1, 0, 1)",
    },
    "macd_line": {
        "category": FeatureCategory.MOMENTUM,
        "range": None,  # Unbounded
        "type": "float",
        "description": "MACD line - difference between fast and slow EMAs",
    },
    "macd_histogram": {
        "category": FeatureCategory.MOMENTUM,
        "range": None,
        "type": "float",
        "description": "MACD histogram - difference between MACD and signal line",
    },
    "macd_cross_signal": {
        "category": FeatureCategory.MOMENTUM,
        "range": (-1, 1),
        "type": "int",
        "description": "MACD crossover signal (-1, 0, 1)",
    },
    
    # Trend indicators
    "sma_5": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "5-day Simple Moving Average",
    },
    "ema_5": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "5-day Exponential Moving Average",
    },
    "sma_10": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "10-day Simple Moving Average",
    },
    "ema_10": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "10-day Exponential Moving Average",
    },
    "sma_50": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "50-day Simple Moving Average",
    },
    "ema_50": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "50-day Exponential Moving Average",
    },
    "ema_cross": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "Difference between short and long EMAs",
    },
    "adx_14": {
        "category": FeatureCategory.TREND,
        "range": (0, 100),
        "type": "float",
        "description": "14-day Average Directional Index - trend strength",
    },
    "ichimoku_conversion": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "Ichimoku Conversion Line (Tenkan-sen)",
    },
    "ichimoku_base": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "Ichimoku Base Line (Kijun-sen)",
    },
    "ichimoku_lead_span_a": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "Ichimoku Leading Span A (modified to avoid lookahead bias)",
    },
    "ichimoku_lead_span_b": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "Ichimoku Leading Span B (modified to avoid lookahead bias)",
    },
    "ichimoku_lagging_span": {
        "category": FeatureCategory.TREND,
        "range": None,
        "type": "float",
        "description": "Ichimoku Lagging Span (use with caution in ML)",
    },
    
    # Volume indicators
    "obv": {
        "category": FeatureCategory.VOLUME,
        "range": None,
        "type": "float",
        "description": "On-Balance Volume - cumulative volume indicator",
    },
    "obv_pct": {
        "category": FeatureCategory.VOLUME,
        "range": None,
        "type": "float",
        "description": "Percent change in OBV",
    },
    "obv_z20": {
        "category": FeatureCategory.VOLUME,
        "range": None,
        "type": "float",
        "description": "Z-score of OBV over 20 days",
    },
    "volume_avg_ratio_5d": {
        "category": FeatureCategory.VOLUME,
        "range": (0, None),
        "type": "float",
        "description": "Ratio of today's volume to 5-day average",
    },
    
    # Volatility indicators
    "atr": {
        "category": FeatureCategory.VOLATILITY,
        "range": (0, None),
        "type": "float",
        "description": "Average True Range - volatility measure",
    },
    "atr_pct_of_price": {
        "category": FeatureCategory.VOLATILITY,
        "range": (0, None),
        "type": "float",
        "description": "ATR as percentage of price",
    },
    "bb_width": {
        "category": FeatureCategory.VOLATILITY,
        "range": (0, None),
        "type": "float",
        "description": "Bollinger Band width - volatility measure",
    },
    
    # Price action features
    "log_return_1d": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": None,
        "type": "float",
        "description": "1-day log return",
    },
    "log_return_5d": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": None,
        "type": "float",
        "description": "5-day log return",
    },
    "close_vs_ma10": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, None),
        "type": "float",
        "description": "Close price relative to 10-day MA",
    },
    "close_vs_ma20": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, None),
        "type": "float",
        "description": "Close price relative to 20-day MA",
    },
    "close_zscore_20": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": None,
        "type": "float",
        "description": "Z-score of close price over 20 days",
    },
    "price_percentile_20d": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, 1),
        "type": "float",
        "description": "Percentile rank of close in 20-day window",
    },
    "gap_up_pct": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": None,
        "type": "float",
        "description": "Gap up percentage at open",
    },
    "rolling_max_5d_breakout": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, None),
        "type": "float",
        "description": "Percent above 5-day high",
    },
    "rolling_min_5d_breakdown": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, None),
        "type": "float",
        "description": "Percent below 5-day low",
    },
    "high_vs_close": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, None),
        "type": "float",
        "description": "High price relative to close",
    },
    "daily_range_pct": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, None),
        "type": "float",
        "description": "Daily high-low range as percentage",
    },
    "candle_body_pct": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": None,
        "type": "float",
        "description": "Candle body as percentage of range",
    },
    "close_position_in_range": {
        "category": FeatureCategory.PRICE_ACTION,
        "range": (0, 1),
        "type": "float",
        "description": "Close position within daily range (0=low, 1=high)",
    },
    
    # Pattern features (binary)
    "bullish_engulfing": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Bullish engulfing candlestick pattern",
    },
    "bearish_engulfing": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Bearish engulfing candlestick pattern",
    },
    "hammer_signal": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Hammer candlestick pattern",
    },
    "shooting_star_signal": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Shooting star candlestick pattern",
    },
    "marubozu_white": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "White marubozu pattern",
    },
    "marubozu_black": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Black marubozu pattern",
    },
    "doji_signal": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Doji candlestick pattern",
    },
    "long_legged_doji": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Long-legged doji pattern",
    },
    "morning_star": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Morning star 3-candle pattern",
    },
    "evening_star": {
        "category": FeatureCategory.PATTERN,
        "range": (0, 1),
        "type": "float",
        "description": "Evening star 3-candle pattern",
    },
}


def get_feature_metadata(feature_name: str) -> Optional[Dict]:
    """Get metadata for a feature."""
    return FEATURE_METADATA.get(feature_name)


def get_features_by_category(category: FeatureCategory) -> List[str]:
    """Get all feature names in a given category."""
    return [
        name for name, meta in FEATURE_METADATA.items()
        if meta.get("category") == category
    ]


def get_all_categories() -> List[FeatureCategory]:
    """Get all feature categories."""
    return list(FeatureCategory)


def get_feature_summary() -> Dict[str, int]:
    """Get summary of features by category."""
    summary = {}
    for meta in FEATURE_METADATA.values():
        cat = meta.get("category")
        if cat:
            cat_name = cat.value if isinstance(cat, FeatureCategory) else str(cat)
            summary[cat_name] = summary.get(cat_name, 0) + 1
    return summary

