import pandas as pd
import numpy as np
import pytest
import yaml

from features.technical import (
    feature_5d_return,
    feature_10d_return,
    feature_atr,
    feature_bb_width,
    feature_ema_cross,
    feature_obv,
    feature_rsi
)
from features.registry import load_enabled_features

def test_feature_5d_return():
    # 7 points to allow two valid 5-day returns at indices 0 and 1
    df = pd.DataFrame({'close': [100, 110,  90, 105, 120, 130, 140]})
    result = feature_5d_return(df)
    expected = pd.Series([
        (130 / 100) - 1,  # index 0: close[5] / close[0]
        (140 / 110) - 1,  # index 1: close[6] / close[1]
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan
    ], index=df.index)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_10d_return():
    # 22 points to allow 10-day returns for i=0..11
    close_values = list(range(1, 23))
    df = pd.DataFrame({'close': close_values})
    result = feature_10d_return(df)
    expected = pd.Series(df['close']).shift(-10) / pd.Series(df['close']) - 1
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_atr():
    # Synthetic data for ATR period=3
    data = {
        'High':  [10, 12, 14, 13],
        'Low':   [ 8,  9, 10, 11],
        'Close': [ 9, 11, 13, 12]
    }
    df = pd.DataFrame(data)
    # TR = [2,3,4,2] -> ATR(period=3) = [nan, nan, 3.0, 3.0]
    expected = pd.Series([np.nan, np.nan, 3.0, 3.0], index=df.index)
    result = feature_atr(df, period=3)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_bb_width():
    # Synthetic data for Bollinger Band width, period=3, std_dev=2
    df = pd.DataFrame({'close': [1,2,3,4,5,6,7]})
    close = df['close']
    sma = close.rolling(window=3).mean()
    std = close.rolling(window=3).std(ddof=0)
    expected = (2 * 2.0 * std) / sma
    result = feature_bb_width(df, period=3, std_dev=2.0)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_ema_cross():
    # EMA crossover for small series
    df = pd.DataFrame({'close': [1,2,3,4,5,6,7,8]})
    close = df['close']
    result = feature_ema_cross(df, span_short=12, span_long=26)
    expected = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_obv():
    # Build a tiny price + volume series
    data = {
        'close': [10, 12, 11, 13, 13],
        'volume': [100, 200, 150, 300, 250]
    }
    df = pd.DataFrame(data)
    # Manually compute OBV:
    #  start at 0,
    #  close↑: add vol, close↓: subtract vol, equal: no change.
    #  steps: +100, +200, -150, +300, 0 → [100, 300, 150, 450, 450]
    expected = pd.Series([100, 300, 150, 450, 450], index=df.index)
    result = feature_obv(df)
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_load_enabled_features(tmp_path):
    # Config with mixed flags
    cfg = {
        'features': {
            '5d_return': 1,
            '10d_return': 0,
            'atr': 1,
            'bb_width': 1,
            'ema_cross': 1
        }
    }
    config_file = tmp_path / "features.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    enabled = load_enabled_features(str(config_file))
    assert '5d_return' in enabled
    assert 'atr' in enabled
    assert 'bb_width' in enabled
    assert 'ema_cross' in enabled
    assert '10d_return' not in enabled
    for fn in enabled.values():
        assert callable(fn)
