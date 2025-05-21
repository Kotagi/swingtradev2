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
)

# ──────────────────────────────────────────────────────────────────────────────
# Master dict of all available features.
#
# Keys:   string names used in config/features.yaml and column headers
# Values: Python functions that accept a DataFrame and return a Series
# ──────────────────────────────────────────────────────────────────────────────
FEATURES = {
    "5d_return":    feature_5d_return,    # 5-day forward return label
    "10d_return":   feature_10d_return,   # 10-day forward return label
    "atr":          feature_atr,          # Average True Range
    "bb_width":     feature_bb_width,     # Bollinger Band width
    "ema_cross":    feature_ema_cross,    # EMA(short) − EMA(long)
    "obv":          feature_obv,          # On-Balance Volume
    "obv_pct":      feature_obv_pct,      # Daily % change of OBV
    "obv_z20":      feature_obv_zscore,   # Z-score of OBV over 20 days
    "rsi":          feature_rsi,          # Relative Strength Index
    "sma_5":        feature_sma_5,        # 5-day Simple Moving Average
    "ema_5":        feature_ema_5,        # 5-day Exponential Moving Average
    "sma_10":       feature_sma_10,       # 10-day SMA
    "ema_10":       feature_ema_10,       # 10-day EMA
    "sma_50":       feature_sma_50,       # 50-day SMA
    "ema_50":       feature_ema_50,       # 50-day EMA
    "adx_14":       feature_adx_14,       # 14-day Average Directional Index
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
