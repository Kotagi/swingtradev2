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
    
)

FEATURES = {
    "5d_return":    feature_5d_return,
    "10d_return":   feature_10d_return,
    "atr":          feature_atr,
    "bb_width":     feature_bb_width,
    "ema_cross":    feature_ema_cross,
    "obv":          feature_obv,
    "obv_pct":      feature_obv_pct,
    "obv_z20":      feature_obv_zscore,   # default 20-day window
    "rsi":          feature_rsi,
    "sma_5":        feature_sma_5,
    "ema_5":        feature_ema_5,
    "sma_10":       feature_sma_10,
    "ema_10":       feature_ema_10,
    "sma_50":       feature_sma_50,
    "ema_50":       feature_ema_50,


}

def load_enabled_features(config_path: str):
    """
    Read config/features.yaml and return a dict of {feature_name: function}
    for all FLAGS set to 1.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    flags = cfg.get("features", {})
    return {
        name: fn
        for name, fn in FEATURES.items()
        if flags.get(name, 0) == 1
    }
