#!/usr/bin/env python3
"""Test relative_strength_vs_sector calculation logic"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '.')

from features.sets.v3_New_Dawn.technical import feature_relative_strength_vs_sector

# Load AAPL cleaned data
print("=" * 80)
print("Testing relative_strength_vs_sector calculation")
print("=" * 80)

# Load cleaned AAPL data (what the feature function receives)
df_aapl_clean = pd.read_parquet('data/clean/AAPL.parquet')
print(f"AAPL cleaned data:")
print(f"  Shape: {df_aapl_clean.shape}")
print(f"  Index type: {type(df_aapl_clean.index)}")
print(f"  Date range: {df_aapl_clean.index.min()} to {df_aapl_clean.index.max()}")
print(f"  Columns: {list(df_aapl_clean.columns)}")

# Calculate the feature
print("\nCalculating feature_relative_strength_vs_sector with ticker='AAPL'...")
rs_result = feature_relative_strength_vs_sector(df_aapl_clean, ticker='AAPL')

print(f"\nResult:")
print(f"  Shape: {rs_result.shape}")
print(f"  Non-null: {rs_result.notna().sum()}")
print(f"  Null: {rs_result.isna().sum()}")
print(f"  Unique values: {rs_result.nunique()}")
print(f"  Value range: {rs_result.min()} to {rs_result.max()}")
print(f"  Mean: {rs_result.mean()}")
print(f"  Std: {rs_result.std()}")

# Check first 30 values (should have zeros for first 20)
print(f"\nFirst 30 values:")
print(rs_result.head(30))

# Check values after day 30 (should have variation)
print(f"\nValues from day 30-60 (should show variation):")
print(rs_result.iloc[30:60])

# Check recent values
print(f"\nLast 30 values:")
print(rs_result.tail(30))

# Check if there's a pattern - are most values near zero?
print(f"\nValue distribution analysis:")
non_zero = rs_result[rs_result != 0.0]
print(f"  Zero values: {(rs_result == 0.0).sum()} ({((rs_result == 0.0).sum() / len(rs_result) * 100):.1f}%)")
print(f"  Non-zero values: {len(non_zero)} ({len(non_zero) / len(rs_result) * 100:.1f}%)")
if len(non_zero) > 0:
    print(f"  Non-zero mean: {non_zero.mean()}")
    print(f"  Non-zero std: {non_zero.std()}")
    print(f"  Non-zero range: {non_zero.min()} to {non_zero.max()}")
    
    # Check if non-zero values are very small (might indicate calculation issue)
    very_small = (non_zero.abs() < 0.001).sum()
    print(f"  Very small values (abs < 0.001): {very_small} ({very_small / len(non_zero) * 100:.1f}%)")

# Test without ticker (should use SPY fallback)
print("\n" + "=" * 80)
print("Testing without ticker (SPY fallback)")
print("=" * 80)
rs_result_spy = feature_relative_strength_vs_sector(df_aapl_clean, ticker=None)
print(f"  Shape: {rs_result_spy.shape}")
print(f"  Unique values: {rs_result_spy.nunique()}")
print(f"  Value range: {rs_result_spy.min()} to {rs_result_spy.max()}")
print(f"  Mean: {rs_result_spy.mean()}")
print(f"  Std: {rs_result_spy.std()}")

# Compare the two
print("\n" + "=" * 80)
print("Comparison: With ticker vs Without ticker")
print("=" * 80)
if len(rs_result) == len(rs_result_spy):
    diff = (rs_result - rs_result_spy).abs()
    print(f"  Max difference: {diff.max()}")
    print(f"  Mean difference: {diff.mean()}")
    print(f"  Values that differ: {(diff > 0.0001).sum()} ({(diff > 0.0001).sum() / len(rs_result) * 100:.1f}%)")
    
    if (diff > 0.0001).sum() == 0:
        print("  ⚠️  WARNING: Results are identical! Feature might always be using SPY fallback!")
    else:
        print("  ✓ Results differ - feature is using sector ETF correctly")

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
