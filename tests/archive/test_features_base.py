import pandas as pd
import numpy as np
from src.feature_engineering import compute_features

def make_dummy_df():
    # 100-day increasing dummy data
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    data = {
        "Open": np.linspace(100, 200, 100),
        "High": np.linspace(101, 201, 100),
        "Low":  np.linspace(99, 199, 100),
        "Close":np.linspace(100, 200, 100),
        "Volume": np.random.randint(1e6, 1e7, 100),
    }
    return pd.DataFrame(data, index=dates)

def test_compute_features_columns_exist():
    df = make_dummy_df()
    df_feat = compute_features(df.copy())

    # Check that every indicator column is present
    expected = {
        "SMA_10", "SMA_20", "SMA_50",
        "EMA_10", "EMA_20", "EMA_50",
        "RSI_14", "MACD", "MACD_signal", "ATR_14"
    }
    assert expected.issubset(set(df_feat.columns))

def test_values_not_all_nan():
    df = make_dummy_df()
    df_feat = compute_features(df.copy())
    # At least one non-NaN in each
    for col in ["SMA_10", "RSI_14", "MACD", "ATR_14"]:
        assert df_feat[col].notna().any()
