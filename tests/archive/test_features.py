# tests/test_features.py

import pandas as pd
import pytest
import yaml

from features.technical import feature_5d_return, feature_10d_return
from features.registry import load_enabled_features

def test_feature_5d_return():
    # Create a close price series where 5-day return is known
    df = pd.DataFrame({'close': [100, 110, 90, 105, 120, 130]})
    result = feature_5d_return(df)
    expected = pd.Series([
        (120 / 100) - 1,
        (130 / 110) - 1,
        None, None, None, None
    ])
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_feature_10d_return():
    # Create a longer series for 10-day return
    close_values = list(range(1, 22))
    df = pd.DataFrame({'close': close_values})
    result = feature_10d_return(df)
    # First element: (11/1)-1, rest should be NaN/None
    expected = pd.Series([(11 / 1) - 1] + [None] * (len(close_values) - 1))
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_load_enabled_features(tmp_path):
    # Write a temporary config YAML with mixed flags
    config_file = tmp_path / "features.yaml"
    config_data = {
        'features': {
            '5d_return': 1,
            '10d_return': 0,
            'atr': 1,
            'bb_width': 0
        }
    }
    config_file.write_text(yaml.safe_dump(config_data))

    # Load enabled features
    enabled = load_enabled_features(str(config_file))

    # Assertions
    assert '5d_return' in enabled
    assert 'atr' in enabled
    assert '10d_return' not in enabled
    assert 'bb_width' not in enabled
    # Ensure the loaded entries are callables
    for fn in enabled.values():
        assert callable(fn)
