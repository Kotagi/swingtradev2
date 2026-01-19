# features/sets/v3_New_Dawn/technical.py

"""
Technical feature functions for feature set 'v3_New_Dawn'.

This file is empty by default. Add your feature functions here.
Each function should take a DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
and return a pandas Series.
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from typing import Optional

# Import shared utilities
from features.shared.utils import (
    _load_spy_data,
    _get_column,
    _get_close_series,
    _get_open_series,
    _get_high_series,
    _get_low_series,
    _get_volume_series,
    _rolling_percentile_rank,
)

# Add your feature functions here
# Example:
# def feature_my_custom_feature(df: DataFrame) -> Series:
#     """My custom feature."""
#     close = _get_close_series(df)
#     return close * 2
