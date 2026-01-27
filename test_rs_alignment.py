#!/usr/bin/env python3
"""Test relative_strength_vs_sector alignment"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '.')

from features.shared.utils import _load_sector_mapping, _load_sector_etf_data, _load_spy_data

# Load AAPL feature data
print("=" * 80)
print("1. Loading AAPL feature data")
print("=" * 80)
df_aapl = pd.read_parquet('data/features_labeled_v3_New_Dawn/AAPL.parquet')
print(f"AAPL feature data:")
print(f"  Shape: {df_aapl.shape}")
print(f"  Index type: {type(df_aapl.index)}")
print(f"  Date range: {df_aapl.index.min()} to {df_aapl.index.max()}")
print(f"  Has relative_strength_vs_sector: {'relative_strength_vs_sector' in df_aapl.columns}")

if 'relative_strength_vs_sector' in df_aapl.columns:
    rs = df_aapl['relative_strength_vs_sector']
    print(f"\nRelative Strength vs Sector:")
    print(f"  Non-null count: {rs.notna().sum()}")
    print(f"  Null count: {rs.isna().sum()}")
    print(f"  Unique values: {rs.nunique()}")
    print(f"  Value range: {rs.min()} to {rs.max()}")
    print(f"  Mean: {rs.mean()}")
    print(f"  Std: {rs.std()}")
    print(f"  First 20 values:")
    print(rs.head(20))
    print(f"  Last 20 values:")
    print(rs.tail(20))
    
    # Check if all values are the same
    if rs.nunique() == 1:
        print(f"\n⚠️  WARNING: All values are the same! Value = {rs.iloc[0]}")
    elif rs.nunique() < 10:
        print(f"\n⚠️  WARNING: Only {rs.nunique()} unique values - likely alignment issue!")

# Load sector mapping
print("\n" + "=" * 80)
print("2. Checking sector mapping for AAPL")
print("=" * 80)
sector_mapping = _load_sector_mapping()
aapl_sector = sector_mapping.get('AAPL')
print(f"AAPL sector: {aapl_sector}")

# Load XLK data (Technology sector ETF)
print("\n" + "=" * 80)
print("3. Loading XLK (Technology ETF) data")
print("=" * 80)
xlk_data = _load_sector_etf_data('XLK')
if xlk_data is not None:
    print(f"XLK data:")
    print(f"  Shape: {xlk_data.shape}")
    print(f"  Index type: {type(xlk_data.index)}")
    print(f"  Date range: {xlk_data.index.min()} to {xlk_data.index.max()}")
    print(f"  Columns: {list(xlk_data.columns)}")
    
    # Get close column
    xlk_close = None
    for col in ['Close', 'close', 'Adj Close', 'AdjClose', 'Price']:
        if col in xlk_data.columns:
            xlk_close = xlk_data[col]
            print(f"  Using column: {col}")
            break
    
    if xlk_close is not None:
        print(f"  Close price range: {xlk_close.min()} to {xlk_close.max()}")
        print(f"  First 10 dates and values:")
        print(xlk_close.head(10))
        print(f"  Last 10 dates and values:")
        print(xlk_close.tail(10))
else:
    print("  ⚠️  XLK data not found!")

# Test alignment
print("\n" + "=" * 80)
print("4. Testing alignment between AAPL dates and XLK dates")
print("=" * 80)
if xlk_close is not None:
    # Check date overlap
    aapl_dates = pd.to_datetime(df_aapl.index).normalize()
    xlk_dates = pd.to_datetime(xlk_close.index).normalize()
    
    overlap = set(aapl_dates) & set(xlk_dates)
    print(f"AAPL dates: {len(aapl_dates)}")
    print(f"XLK dates: {len(xlk_dates)}")
    print(f"Overlapping dates: {len(overlap)}")
    print(f"Overlap percentage: {len(overlap) / len(aapl_dates) * 100:.1f}%")
    
    # Test reindex
    print("\nTesting reindex with method='ffill':")
    xlk_aligned = xlk_close.reindex(aapl_dates, method='ffill')
    print(f"  Aligned shape: {xlk_aligned.shape}")
    print(f"  Non-null count: {xlk_aligned.notna().sum()}")
    print(f"  Null count: {xlk_aligned.isna().sum()}")
    print(f"  First 20 aligned values:")
    print(xlk_aligned.head(20))
    
    # Check if all values are the same after alignment
    if xlk_aligned.notna().sum() > 0:
        non_null_values = xlk_aligned.dropna()
        if non_null_values.nunique() == 1:
            print(f"\n⚠️  WARNING: All non-null aligned values are the same! Value = {non_null_values.iloc[0]}")
        elif non_null_values.nunique() < 10:
            print(f"\n⚠️  WARNING: Only {non_null_values.nunique()} unique values after alignment!")

# Test SPY data
print("\n" + "=" * 80)
print("5. Testing SPY data")
print("=" * 80)
spy_data = _load_spy_data()
if spy_data is not None:
    print(f"SPY data:")
    print(f"  Shape: {spy_data.shape}")
    print(f"  Index type: {type(spy_data.index)}")
    print(f"  Date range: {spy_data.index.min()} to {spy_data.index.max()}")
    print(f"  Columns: {list(spy_data.columns)}")
    
    # Get close column
    spy_close = None
    for col in ['Close', 'close', 'Adj Close', 'AdjClose', 'Price']:
        if col in spy_data.columns:
            spy_close = spy_data[col]
            print(f"  Using column: {col}")
            break
    
    if spy_close is not None:
        print(f"  Close price range: {spy_close.min()} to {spy_close.max()}")
        
        # Test alignment
        spy_dates = pd.to_datetime(spy_close.index).normalize()
        overlap = set(aapl_dates) & set(spy_dates)
        print(f"\n  SPY dates: {len(spy_dates)}")
        print(f"  Overlapping dates with AAPL: {len(overlap)}")
        print(f"  Overlap percentage: {len(overlap) / len(aapl_dates) * 100:.1f}%")
        
        # Test reindex
        spy_aligned = spy_close.reindex(aapl_dates, method='ffill')
        print(f"\n  Aligned shape: {spy_aligned.shape}")
        print(f"  Non-null count: {spy_aligned.notna().sum()}")
        print(f"  Null count: {spy_aligned.isna().sum()}")
        
        if spy_aligned.notna().sum() > 0:
            non_null_values = spy_aligned.dropna()
            if non_null_values.nunique() == 1:
                print(f"\n  ⚠️  WARNING: All non-null aligned values are the same! Value = {non_null_values.iloc[0]}")
else:
    print("  ⚠️  SPY data not found!")

print("\n" + "=" * 80)
print("Test complete")
print("=" * 80)
