import yaml
from features.technical import (
    feature_5d_return, feature_10d_return, feature_atr,
    feature_bb_width, feature_ema_cross, feature_obv, feature_rsi
)

FEATURES = {
    "5d_return": feature_5d_return,
    "10d_return": feature_10d_return,
    "atr": feature_atr,
    "bb_width": feature_bb_width,
    "ema_cross": feature_ema_cross,
    "obv": feature_obv,
    "rsi": feature_rsi,
    # ...you can add placeholders here for future phases
}

def load_enabled_features(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    flags = cfg.get("features", {})
    return {name: fn for name, fn in FEATURES.items() if flags.get(name) == 1}
