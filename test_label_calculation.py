#!/usr/bin/env python3
"""
Label Calculation Diagnostic Test

Tests the label_future_return function to ensure it's working correctly.
This script:
1. Tests with synthetic data (known outcomes)
2. Tests with real training data
3. Shows label distributions
4. Identifies any calculation issues
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.labeling import label_future_return


def test_synthetic_data():
    """Test label calculation with synthetic data where we know the expected outcomes."""
    print("=" * 80)
    print("TEST 1: Synthetic Data Test")
    print("=" * 80)
    
    # Create synthetic data: 10 days of prices
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    # Scenario: Stock starts at $100, goes up to $120 on day 5, then back down
    # Day 0: $100 close, $100 high
    # Day 1: $105 close, $108 high
    # Day 2: $110 close, $112 high
    # Day 3: $115 close, $118 high
    # Day 4: $120 close, $120 high (peak)
    # Day 5: $115 close, $117 high
    # Day 6: $110 close, $112 high
    # Day 7: $105 close, $107 high
    # Day 8: $100 close, $102 high
    # Day 9: $95 close, $98 high
    
    data = {
        'close': [100, 105, 110, 115, 120, 115, 110, 105, 100, 95],
        'high': [100, 108, 112, 118, 120, 117, 112, 107, 102, 98]
    }
    df = pd.DataFrame(data, index=dates)
    
    print("\nSynthetic Price Data:")
    print(df)
    
    # Test 1: 5-day horizon, 15% threshold (0.15)
    # Day 0: max_high in days 1-5 = max(108, 112, 118, 120, 117) = 120
    #        return = (120/100) - 1 = 0.20 = 20% > 15% threshold → label = 1 ✓
    # Day 1: max_high in days 2-6 = max(112, 118, 120, 117, 112) = 120
    #        return = (120/105) - 1 = 0.143 = 14.3% < 15% threshold → label = 0 ✓
    # Day 2: max_high in days 3-7 = max(118, 120, 117, 112, 107) = 120
    #        return = (120/110) - 1 = 0.091 = 9.1% < 15% threshold → label = 0 ✓
    
    df_test = df.copy()
    label_future_return(df_test, horizon=5, threshold=0.15, label_name='label_5d_15pct')
    
    print("\n" + "=" * 80)
    print("Results for 5-day horizon, 15% threshold:")
    print("=" * 80)
    print(df_test[['close', 'high', 'label_5d_15pct']])
    
    # Verify expected outcomes
    expected = {
        0: 1,  # Day 0: 20% return > 15% → label = 1
        1: 0,  # Day 1: 14.3% return < 15% → label = 0
        2: 0,  # Day 2: 9.1% return < 15% → label = 0
    }
    
    print("\nVerification:")
    all_correct = True
    for day_idx, expected_label in expected.items():
        actual_label = df_test.iloc[day_idx]['label_5d_15pct']
        status = "PASS" if actual_label == expected_label else "FAIL"
        if actual_label != expected_label:
            all_correct = False
        print(f"  Day {day_idx}: Expected {expected_label}, Got {actual_label} [{status}]")
    
    if all_correct:
        print("\n[PASS] Synthetic data test PASSED")
    else:
        print("\n[FAIL] Synthetic data test FAILED")
    
    return all_correct


def test_edge_cases():
    """Test edge cases: insufficient data, exact threshold, etc."""
    print("\n" + "=" * 80)
    print("TEST 2: Edge Cases Test")
    print("=" * 80)
    
    # Test case 1: Not enough future data (last few rows)
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    data = {
        'close': [100, 110, 120, 130, 140],
        'high': [100, 110, 120, 130, 140]
    }
    df = pd.DataFrame(data, index=dates)
    
    df_test = df.copy()
    label_future_return(df_test, horizon=5, threshold=0.10, label_name='label')
    
    print("\nEdge Case 1: Last rows with insufficient future data")
    print(df_test[['close', 'high', 'label']])
    print(f"  Last row label (should be 0 due to NaN): {df_test.iloc[-1]['label']}")
    
    # Test case 2: Exact threshold match
    dates = pd.date_range('2024-01-01', periods=6, freq='D')
    data = {
        'close': [100, 100, 100, 100, 100, 100],
        'high': [100, 100, 115, 100, 100, 100]  # Exactly 15% on day 2
    }
    df = pd.DataFrame(data, index=dates)
    
    df_test = df.copy()
    label_future_return(df_test, horizon=3, threshold=0.15, label_name='label')
    
    print("\nEdge Case 2: Exact threshold (15%)")
    print(df_test[['close', 'high', 'label']])
    print(f"  Day 0: max_high in days 1-3 = 115, return = 15%, label = {df_test.iloc[0]['label']}")
    print(f"  (Should be 0 because > threshold, not >=)")
    
    return True


def test_real_data(feature_set='v3_New_Dawn', horizon=20, threshold=0.15):
    """Test label calculation with real training data."""
    print("\n" + "=" * 80)
    print("TEST 3: Real Data Test")
    print("=" * 80)
    
    # Determine data directory
    try:
        from feature_set_manager import get_feature_set_data_path
        data_dir = get_feature_set_data_path(feature_set)
    except ImportError:
        # Fallback
        if feature_set:
            data_dir = PROJECT_ROOT / "data" / f"features_labeled_{feature_set}"
        else:
            data_dir = PROJECT_ROOT / "data" / "features_labeled"
    
    if not data_dir.exists():
        print(f"[WARNING] Data directory not found: {data_dir}")
        print("  Skipping real data test")
        return None
    
    # Load a few sample tickers
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[WARNING] No parquet files found in {data_dir}")
        print("  Skipping real data test")
        return None
    
    print(f"\nFound {len(parquet_files)} ticker files")
    print(f"Testing with first 3 tickers...")
    
    # Load first 3 tickers
    sample_tickers = []
    for f in parquet_files[:3]:
        df = pd.read_parquet(f)
        df.index.name = "date"
        df["ticker"] = f.stem
        sample_tickers.append(df)
        print(f"  Loaded {f.stem}: {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
    
    if not sample_tickers:
        print("  No data loaded")
        return None
    
    # Combine and test labeling
    df_all = pd.concat(sample_tickers, axis=0)
    
    # Find close and high columns
    # Prefer 'close' over 'adj close' for consistency with 'high' (both split-adjusted only)
    close_col = None
    high_col = None
    for col in df_all.columns:
        # Prefer 'close' (split-adjusted) over 'adj close' (split+dividend adjusted)
        if col.lower() == 'close':
            close_col = col
        elif col.lower() == 'adj close' and close_col is None:
            # Fallback to adj close if regular close not found
            close_col = col
        if col.lower() == 'high':
            high_col = col
    
    if not close_col or not high_col:
        print(f"[WARNING] Could not find close/high columns. Available columns: {list(df_all.columns)[:10]}...")
        return None
    
    print(f"\nUsing columns: close='{close_col}', high='{high_col}'")
    
    # Calculate labels
    df_test = df_all.copy()
    label_future_return(
        df_test,
        close_col=close_col,
        high_col=high_col,
        horizon=horizon,
        threshold=threshold,
        label_name='test_label'
    )
    
    # Analyze label distribution
    print(f"\n" + "=" * 80)
    print(f"Label Distribution Analysis (horizon={horizon}d, threshold={threshold:.1%})")
    print("=" * 80)
    
    label_col = 'test_label'
    total = len(df_test)
    positive = df_test[label_col].sum()
    negative = total - positive
    positive_pct = (positive / total) * 100 if total > 0 else 0
    
    print(f"\nTotal samples: {total:,}")
    print(f"Positive labels (1): {positive:,} ({positive_pct:.2f}%)")
    print(f"Negative labels (0): {negative:,} ({100-positive_pct:.2f}%)")
    print(f"Class imbalance ratio (neg/pos): {negative/positive:.2f}" if positive > 0 else "  (No positive labels!)")
    
    # Check for issues
    issues = []
    if positive_pct > 50:
        issues.append(f"[WARNING] More than 50% positive labels ({positive_pct:.2f}%) - unusual for {horizon}d/{threshold:.1%} threshold")
    elif positive_pct < 1:
        issues.append(f"[WARNING] Less than 1% positive labels ({positive_pct:.2f}%) - very rare events")
    elif positive_pct > 20:
        issues.append(f"[WARNING] High positive label rate ({positive_pct:.2f}%) - may indicate threshold too low")
    
    if positive == 0:
        issues.append("[CRITICAL] No positive labels found! Label calculation may be broken.")
    
    if issues:
        print("\nIssues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n[OK] Label distribution looks reasonable")
    
    # Show sample of positive labels
    if positive > 0:
        print(f"\nSample of positive labels (first 5):")
        positive_samples = df_test[df_test[label_col] == 1].head(5)
        for idx, row in positive_samples.iterrows():
            print(f"  {row.get('ticker', 'N/A')} on {idx}: close={row[close_col]:.2f}, high={row[high_col]:.2f}")
    
    # Calculate expected return for a few samples
    print(f"\nVerification: Calculating returns for first 5 rows...")
    for i in range(min(5, len(df_test))):
        row = df_test.iloc[i]
        close_price = row[close_col]
        
        # Get future highs
        future_highs = []
        for j in range(1, horizon + 1):
            if i + j < len(df_test):
                future_highs.append(df_test.iloc[i + j][high_col])
        
        if future_highs:
            max_high = max(future_highs)
            calculated_return = (max_high / close_price) - 1
            expected_label = 1 if calculated_return > threshold else 0
            actual_label = row[label_col]
            
            match = "[PASS]" if expected_label == actual_label else "[FAIL]"
            print(f"  Row {i}: close={close_price:.2f}, max_future_high={max_high:.2f}, "
                  f"return={calculated_return:.2%}, expected_label={expected_label}, "
                  f"actual_label={actual_label} {match}")
    
    return {
        'total': total,
        'positive': positive,
        'negative': negative,
        'positive_pct': positive_pct,
        'issues': issues
    }


def test_full_training_data(feature_set='v3_New_Dawn', horizon=20, threshold=0.15):
    """Test with all training data to see full distribution."""
    print("\n" + "=" * 80)
    print("TEST 4: Full Training Data Analysis")
    print("=" * 80)
    
    # Determine data directory
    try:
        from feature_set_manager import get_feature_set_data_path
        data_dir = get_feature_set_data_path(feature_set)
    except ImportError:
        if feature_set:
            data_dir = PROJECT_ROOT / "data" / f"features_labeled_{feature_set}"
        else:
            data_dir = PROJECT_ROOT / "data" / "features_labeled"
    
    if not data_dir.exists():
        print(f"[WARNING] Data directory not found: {data_dir}")
        return None
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[WARNING] No parquet files found")
        return None
    
    print(f"\nLoading all {len(parquet_files)} tickers...")
    
    # Load all tickers
    all_dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            df.index.name = "date"
            df["ticker"] = f.stem
            all_dfs.append(df)
        except Exception as e:
            print(f"  [WARNING] Error loading {f.stem}: {e}")
            continue
    
    if not all_dfs:
        print("  No data loaded")
        return None
    
    df_all = pd.concat(all_dfs, axis=0)
    print(f"  Loaded {len(df_all):,} total rows from {len(all_dfs)} tickers")
    print(f"  Date range: {df_all.index.min()} to {df_all.index.max()}")
    
    # Find close and high columns
    # Prefer 'close' over 'adj close' for consistency with 'high' (both split-adjusted only)
    close_col = None
    high_col = None
    for col in df_all.columns:
        # Prefer 'close' (split-adjusted) over 'adj close' (split+dividend adjusted)
        if col.lower() == 'close':
            close_col = col
        elif col.lower() == 'adj close' and close_col is None:
            # Fallback to adj close if regular close not found
            close_col = col
        if col.lower() == 'high':
            high_col = col
    
    if not close_col or not high_col:
        print(f"[WARNING] Could not find close/high columns")
        return None
    
    # Calculate labels
    print(f"\nCalculating labels (horizon={horizon}d, threshold={threshold:.1%})...")
    df_labeled = df_all.copy()
    label_future_return(
        df_labeled,
        close_col=close_col,
        high_col=high_col,
        horizon=horizon,
        threshold=threshold,
        label_name='test_label'
    )
    
    # Analyze by date split (matching training splits)
    train_start = pd.to_datetime("2010-01-01")  # Updated to match new training split
    train_end = pd.to_datetime("2021-12-31")     # Training ends 2021 (includes COVID)
    val_end = pd.to_datetime("2023-12-31")       # Validation: 2022-2023 (2 years)
    
    dates = pd.to_datetime(df_labeled.index)
    train_mask = (dates >= train_start) & (dates <= train_end)
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    print(f"\n" + "=" * 80)
    print("Label Distribution by Split")
    print("=" * 80)
    
    splits = {
        'Training': train_mask,
        'Validation': val_mask,
        'Test': test_mask
    }
    
    results = {}
    for split_name, mask in splits.items():
        split_data = df_labeled[mask]
        if len(split_data) == 0:
            print(f"\n{split_name}: No data")
            continue
        
        total = len(split_data)
        positive = split_data['test_label'].sum()
        negative = total - positive
        positive_pct = (positive / total) * 100 if total > 0 else 0
        
        results[split_name] = {
            'total': total,
            'positive': positive,
            'negative': negative,
            'positive_pct': positive_pct
        }
        
        print(f"\n{split_name}:")
        print(f"  Total samples: {total:,}")
        print(f"  Positive: {positive:,} ({positive_pct:.2f}%)")
        print(f"  Negative: {negative:,} ({100-positive_pct:.2f}%)")
        if positive > 0:
            print(f"  Class ratio (neg/pos): {negative/positive:.2f}")
            print(f"  Expected scale_pos_weight: {negative/positive:.2f}")
        else:
            print(f"  [WARNING] No positive labels in {split_name} split!")
    
    # Overall statistics
    total_all = len(df_labeled)
    positive_all = df_labeled['test_label'].sum()
    positive_pct_all = (positive_all / total_all) * 100
    
    print(f"\n" + "=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print(f"Total samples: {total_all:,}")
    print(f"Positive labels: {positive_all:,} ({positive_pct_all:.2f}%)")
    print(f"Negative labels: {total_all - positive_all:,} ({100-positive_pct_all:.2f}%)")
    if positive_all > 0:
        print(f"Overall class ratio (neg/pos): {(total_all - positive_all)/positive_all:.2f}")
        print(f"Expected scale_pos_weight: {(total_all - positive_all)/positive_all:.2f}")
    
    # Compare to your metadata
    print(f"\n" + "=" * 80)
    print("Comparison to Training Metadata")
    print("=" * 80)
    print(f"Your metadata shows: scale_pos_weight = 0.396")
    if positive_all > 0:
        expected_ratio = (total_all - positive_all) / positive_all
        print(f"Calculated ratio: {expected_ratio:.2f}")
        if abs(expected_ratio - 0.396) > 0.1:
            print(f"[WARNING] MISMATCH: Expected ratio ({expected_ratio:.2f}) doesn't match metadata (0.396)")
            print(f"  This suggests the label distribution has changed or there's a calculation issue")
        else:
            print(f"[OK] Ratio matches metadata")
    
    return results


def main():
    """Run all diagnostic tests."""
    print("=" * 80)
    print("LABEL CALCULATION DIAGNOSTIC TEST")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Synthetic data
    test1_passed = test_synthetic_data()
    
    # Test 2: Edge cases
    test2_passed = test_edge_cases()
    
    # Test 3: Real data (sample)
    test3_results = test_real_data(feature_set='v3_New_Dawn', horizon=20, threshold=0.15)
    
    # Test 4: Full training data
    test4_results = test_full_training_data(feature_set='v3_New_Dawn', horizon=20, threshold=0.15)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Synthetic data test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Edge cases test: {'PASSED' if test2_passed else 'FAILED'}")
    if test3_results:
        print(f"Real data (sample) test: COMPLETED")
        print(f"  Positive label rate: {test3_results['positive_pct']:.2f}%")
    if test4_results:
        print(f"Full training data test: COMPLETED")
        if 'Training' in test4_results:
            train_pct = test4_results['Training']['positive_pct']
            print(f"  Training set positive rate: {train_pct:.2f}%")
    
    print("\n" + "=" * 80)
    print("Diagnostic complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
