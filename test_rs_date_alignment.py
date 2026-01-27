#!/usr/bin/env python3
"""Test date alignment issue in relative_strength_vs_sector"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '.')

from features.shared.utils import _load_sector_etf_data

# Load AAPL cleaned data
df_aapl = pd.read_parquet('data/clean/AAPL.parquet')
print("AAPL cleaned data index:")
print(f"  Type: {type(df_aapl.index)}")
print(f"  First 5 dates: {df_aapl.index[:5]}")
print(f"  Has time component: {any(hasattr(d, 'hour') and d.hour != 0 for d in df_aapl.index[:10])}")

# Load XLK data
xlk_data = _load_sector_etf_data('XLK')
xlk_close = xlk_data['Close']
print("\nXLK data index:")
print(f"  Type: {type(xlk_close.index)}")
print(f"  First 5 dates: {xlk_close.index[:5]}")
print(f"  Has time component: {any(hasattr(d, 'hour') and d.hour != 0 for d in xlk_close.index[:10])}")

# Test reindex as done in the feature
print("\nTesting reindex as done in feature (line 11817):")
print("  etf_close_aligned = etf_close.reindex(df.index, method='ffill')")

# Check if indices match exactly
print("\nIndex comparison:")
print(f"  AAPL index type: {type(df_aapl.index)}")
print(f"  XLK index type: {type(xlk_close.index)}")
print(f"  Types match: {type(df_aapl.index) == type(xlk_close.index)}")

# Normalize both indices
aapl_dates_norm = pd.to_datetime(df_aapl.index).normalize()
xlk_dates_norm = pd.to_datetime(xlk_close.index).normalize()

print(f"\nAfter normalization:")
print(f"  AAPL dates normalized: {aapl_dates_norm[:5]}")
print(f"  XLK dates normalized: {xlk_dates_norm[:5]}")
print(f"  First dates match: {aapl_dates_norm[0] == xlk_dates_norm[0]}")

# Test reindex with original indices
print("\n1. Reindex with original indices (current implementation):")
try:
    aligned1 = xlk_close.reindex(df_aapl.index, method='ffill')
    print(f"  Success: {aligned1.shape}")
    print(f"  Non-null: {aligned1.notna().sum()}")
    print(f"  First 5 values: {aligned1.head(5).values}")
except Exception as e:
    print(f"  Error: {e}")

# Test reindex with normalized indices
print("\n2. Reindex with normalized indices (safer approach):")
try:
    xlk_close_norm = xlk_close.copy()
    xlk_close_norm.index = pd.to_datetime(xlk_close_norm.index).normalize()
    aligned2 = xlk_close_norm.reindex(aapl_dates_norm, method='ffill')
    print(f"  Success: {aligned2.shape}")
    print(f"  Non-null: {aligned2.notna().sum()}")
    print(f"  First 5 values: {aligned2.head(5).values}")
except Exception as e:
    print(f"  Error: {e}")

# Check if there are any date mismatches
print("\n3. Checking for date mismatches:")
overlap = set(aapl_dates_norm) & set(xlk_dates_norm)
missing_in_xlk = set(aapl_dates_norm) - set(xlk_dates_norm)
missing_in_aapl = set(xlk_dates_norm) - set(aapl_dates_norm)

print(f"  Overlapping dates: {len(overlap)}")
print(f"  Dates in AAPL but not in XLK: {len(missing_in_xlk)}")
if len(missing_in_xlk) > 0:
    print(f"    First 10 missing: {sorted(list(missing_in_xlk))[:10]}")
print(f"  Dates in XLK but not in AAPL: {len(missing_in_aapl)}")
if len(missing_in_aapl) > 0:
    print(f"    First 10 missing: {sorted(list(missing_in_aapl))[:10]}")

print("\nTest complete")
