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
    horizon: int = 5,
    threshold: float = 0.0,
    label_name: str = "label_5d"
) -> DataFrame:
    """
    Add a binary label column to indicate if the forward return exceeds a threshold.

    For each row t, computes:
        future_return = (close_{t+horizon} / close_t) - 1
    and sets label to 1 if future_return > threshold, else 0.

    Args:
        df: Input DataFrame with a datetime index and price columns.
        close_col: Name of the closing price column in df.
        horizon: Number of periods (rows) to look ahead for return calculation.
        threshold: Minimum return required to assign a positive label.
        label_name: Name for the new binary label column.

    Returns:
        DataFrame: The same DataFrame with a new column `label_name` appended.

    Raises:
        KeyError: If `close_col` is not found in df.columns.

    Side Effects:
        Modifies the input df in-place by adding the label column.
    """
    # Verify that the close price column exists
    if close_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{close_col}' column")

    # Extract the close price series
    close = df[close_col]

    # Compute forward return over the specified horizon
    future_ret = close.shift(-horizon) / close - 1

    # Create binary label: 1 if return > threshold, otherwise 0
    df[label_name] = (future_ret > threshold).astype(int)

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
