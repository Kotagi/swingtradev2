# src/utils/labeling.py

"""
labeling.py

Provides utilities to generate binary labels based on future returns in a time-series DataFrame.
These labels indicate whether the price will rise above a threshold after a specified horizon.
"""

import pandas as pd
from pandas import DataFrame


def label_future_return(
    df: DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    horizon: int = 5,
    threshold: float = 0.0,
    label_name: str = "label_5d"
) -> DataFrame:
    """
    Add a binary label column to indicate if the high price reached the threshold during the horizon.

    For each row t, computes:
        max_high = max(high_{t+1}, high_{t+2}, ..., high_{t+horizon})
        future_return = (max_high / close_t) - 1
    and sets label to 1 if future_return > threshold, else 0.

    This checks if the stock's HIGH reached the target gain at ANY point during the horizon,
    not just if it closed at that level. This is more appropriate for swing trading because
    you want to know if you could have exited at a profit.

    Args:
        df: Input DataFrame with a datetime index and price columns.
        close_col: Name of the closing price column in df.
        high_col: Name of the high price column in df.
        horizon: Number of periods (rows) to look ahead for return calculation.
        threshold: Minimum return required to assign a positive label.
        label_name: Name for the new binary label column.

    Returns:
        DataFrame: The same DataFrame with a new column `label_name` appended.

    Raises:
        KeyError: If `close_col` or `high_col` is not found in df.columns.

    Side Effects:
        Modifies the input df in-place by adding the label column.
    """
    # Verify that the required columns exist
    if close_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{close_col}' column")
    if high_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{high_col}' column")

    # Extract the price series
    close = df[close_col]
    high = df[high_col]

    # For each row t, find the maximum high price in the window from t+1 to t+horizon
    # We use a forward-looking approach by reversing the series, using rolling.max(),
    # then reversing back and adjusting
    
    # Reverse the high series to enable forward-looking windows
    high_reversed = high[::-1].reset_index(drop=True)
    
    # Use rolling.max() with window=horizon to look at next horizon rows
    # After reversing, rolling looks backward in reversed series = forward in original
    max_high_reversed = high_reversed.rolling(window=horizon, min_periods=1).max()
    
    # Reverse back to original order
    max_high_forward = pd.Series(max_high_reversed.values[::-1], index=high.index)
    
    # The above gives us max(high[t], high[t+1], ..., high[t+horizon-1])
    # But we want max(high[t+1], ..., high[t+horizon])
    # So we shift by -1 to get the next row's value, which represents max from t+1 to t+horizon
    max_high_forward = max_high_forward.shift(-1)
    
    # Alternative: compute max including current row, then compare with current high
    # and take max of (current max, next period's high) - but simpler to just shift
    
    # Actually, let's use a more direct approach: for each row, compute max of shifted high values
    # Create shifted versions and take max
    shifted_highs = []
    for i in range(1, horizon + 1):
        shifted_highs.append(high.shift(-i))
    
    # Take max across all shifted versions (this gives max from t+1 to t+horizon)
    max_high_forward = pd.concat(shifted_highs, axis=1).max(axis=1)

    # Compute forward return using max high reached during horizon
    # If max_high_forward is NaN (insufficient future data), future_ret will be NaN
    future_ret = max_high_forward / close - 1

    # Create binary label: 1 if return > threshold, otherwise 0
    # NaN values (insufficient future data) will be labeled as 0
    df[label_name] = (future_ret > threshold).astype(int).fillna(0)

    return df


def label_future_return_regression(
    df: DataFrame,
    close_col: str = "close",
    horizon: int = 5,
    label_name: str = "return_5d"
) -> DataFrame:
    """
    Add a continuous target column representing the future return.

    For each row t, computes:
        return = (close_{t+horizon} / close_t) - 1

    Args:
        df: Input DataFrame with a datetime index and price columns.
        close_col: Name of the closing price column in df.
        horizon: Number of periods to look ahead for return calculation.
        label_name: Name for the new return column.

    Returns:
        DataFrame: The same DataFrame with a new column `label_name` containing raw returns.

    Raises:
        KeyError: If `close_col` is not found in df.columns.

    Side Effects:
        Modifies the input df in-place by adding the return column.
    """
    # Verify that the close price column exists
    if close_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{close_col}' column")

    # Extract the close price series
    close = df[close_col]

    # Compute and assign raw forward return
    df[label_name] = close.shift(-horizon) / close - 1

    return df
