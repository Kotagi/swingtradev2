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
    df = pd.DataFrame({
        'close': [100, 110,  90, 105, 120, 130]
    })
    result = feature_5d_return(df)
    expected = pd.Series([
        (120 / 100) - 1,
        (130 / 110) - 1,
        np.nan,
        np.nan,
        np.nan,
        np.nan
    ], index=df.index)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_10d_return():
    close_values = list(range(1, 16))  # 15 points
    df = pd.DataFrame({'close': close_values})
    result = feature_10d_return(df)

    # Build expected with shift
    expected = pd.Series(close_values).shift(-10) / pd.Series(close_values) - 1
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_atr():
    # Simple synthetic data for ATR period=3
    data = {
        'High':  [10, 12, 14, 13],
        'Low':   [ 8,  9, 10, 11],
        'Close': [ 9, 11, 13, 12]
    }
    df = pd.DataFrame(data)
    # True Range TR = [2,3,4,2]; ATR for period=3 is rolling mean of TR
    expected = pd.Series([np.nan, np.nan, 3.0, 3.0], index=df.index)
    result = feature_atr(df, period=3)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_bb_width():
    # Example DataFrame
    data = {'close': [1, 2, 3, 4, 5, 6, 7]}
    df = pd.DataFrame(data)
    # period=3, std_dev=2
    # middle = [NaN, NaN, 2,3,4,5,6]
    # std =   [NaN, NaN, 0.816,0.816,0.816,0.816,0.816]
    # width = 2*2*std / middle
    expected = pd.Series([np.nan, np.nan] + [
        (2*2*0.816496580927726) / m
        for m in (2,3,4,5,6)
    ])
    result = feature_bb_width(df, period=3, std_dev=2.0)
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_load_enabled_features(tmp_path):
    # Create a temporary config with mixed flags
    cfg = {
        'features': {
            '5d_return': 1,
            '10d_return': 0,
            'atr': 1,
            'bb_width': 0
        }
    }
    config_file = tmp_path / "features.yaml"
    config_file.write_text(yaml.safe_dump(cfg))

    enabled = load_enabled_features(str(config_file))
    assert '5d_return' in enabled
    assert 'atr' in enabled
    assert '10d_return' not in enabled
    assert 'bb_width' not in enabled
    for fn in enabled.values():
        assert callable(fn)
