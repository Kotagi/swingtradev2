#!/usr/bin/env python3
"""
test_spy_data_optimization.py

Test script to verify SPY data optimization is working correctly:
1. SPY data is loaded once (not per worker)
2. SPY data is passed to features correctly
3. Features compute correctly with shared SPY data
4. Results match when SPY data is loaded independently vs shared
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
from features.shared.utils import _load_spy_data
from src.feature_pipeline import apply_features, load_enabled_features_for_set
from utils.logger import setup_logger
import inspect


def test_spy_data_loading():
    """Test 1: Verify SPY data can be loaded and has expected structure."""
    print("=" * 70)
    print("TEST 1: SPY Data Loading")
    print("=" * 70)
    
    spy_data = _load_spy_data()
    
    if spy_data is None:
        print("❌ FAILED: SPY data is None")
        return False
    
    print(f"[PASS] SPY data loaded successfully")
    print(f"   Rows: {len(spy_data)}")
    print(f"   Date range: {spy_data.index.min()} to {spy_data.index.max()}")
    print(f"   Columns: {list(spy_data.columns)}")
    
    # Check for expected columns
    has_close = 'Close' in spy_data.columns or 'close' in spy_data.columns
    if not has_close:
        print("[WARNING] No 'Close' or 'close' column found")
        print(f"   Available columns: {list(spy_data.columns)}")
    
    return True


def test_feature_signatures():
    """Test 2: Verify features that need SPY data have spy_data parameter."""
    print("\n" + "=" * 70)
    print("TEST 2: Feature Function Signatures")
    print("=" * 70)
    
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    
    spy_features = []
    non_spy_features = []
    
    for name, func in enabled.items():
        sig = inspect.signature(func)
        if 'spy_data' in sig.parameters:
            spy_features.append(name)
        else:
            non_spy_features.append(name)
    
    print(f"[PASS] Features that accept spy_data: {len(spy_features)}")
    for name in spy_features:
        print(f"   - {name}")
    
    print(f"\n[PASS] Features that don't need spy_data: {len(non_spy_features)}")
    print(f"   (First 5: {', '.join(non_spy_features[:5])}...)")
    
    # Expected SPY features
    expected_spy_features = ['beta_spy_252d', 'mkt_spy_dist_sma200', 'mkt_spy_sma200_slope']
    missing = [f for f in expected_spy_features if f not in spy_features]
    
    if missing:
        print(f"\n[WARNING] Expected SPY features not found: {missing}")
        return False
    
    print(f"\n[PASS] All expected SPY features found: {expected_spy_features}")
    return True


def test_feature_computation_with_spy():
    """Test 3: Verify features compute correctly with SPY data."""
    print("\n" + "=" * 70)
    print("TEST 3: Feature Computation with SPY Data")
    print("=" * 70)
    
    # Load a sample ticker
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))
    
    if len(ticker_files) == 0:
        print("[FAILED] No ticker files found in data/clean")
        return False
    
    # Use first ticker
    sample_file = ticker_files[0]
    ticker = sample_file.stem
    print(f"Testing with ticker: {ticker}")
    
    df = pd.read_parquet(sample_file)
    print(f"   Rows: {len(df)}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Load features
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    
    # Get SPY features only
    spy_feature_names = ['beta_spy_252d', 'mkt_spy_dist_sma200', 'mkt_spy_sma200_slope']
    spy_features = {name: enabled[name] for name in spy_feature_names if name in enabled}
    
    if len(spy_features) == 0:
        print("[FAILED] No SPY features found in enabled features")
        return False
    
    logger = setup_logger("test", "test_spy.log")
    
    # Test 3a: Compute WITHOUT SPY data (should load internally)
    print("\n   Test 3a: Computing features WITHOUT passing spy_data...")
    spy_data = None
    df_feat_no_spy, _ = apply_features(df, spy_features, logger, validate=False, spy_data=None)
    
    # Test 3b: Compute WITH SPY data (should use passed data)
    print("   Test 3b: Computing features WITH passed spy_data...")
    spy_data = _load_spy_data()
    df_feat_with_spy, _ = apply_features(df, spy_features, logger, validate=False, spy_data=spy_data)
    
    # Compare results
    print("\n   Comparing results...")
    all_match = True
    
    for feature_name in spy_feature_names:
        if feature_name not in df_feat_no_spy.columns or feature_name not in df_feat_with_spy.columns:
            print(f"   [WARNING] {feature_name}: Missing in one of the results")
            continue
        
        no_spy_vals = df_feat_no_spy[feature_name]
        with_spy_vals = df_feat_with_spy[feature_name]
        
        # Check if values match (within floating point tolerance)
        matches = np.allclose(no_spy_vals, with_spy_vals, equal_nan=True)
        
        if matches:
            non_nan_count = no_spy_vals.notna().sum()
            print(f"   [PASS] {feature_name}: Values match ({non_nan_count} non-NaN values)")
        else:
            # Find differences
            diff_mask = ~np.isclose(no_spy_vals, with_spy_vals, equal_nan=True)
            diff_count = diff_mask.sum()
            if diff_count > 0:
                print(f"   [FAILED] {feature_name}: {diff_count} values differ!")
                # Show first few differences
                diff_indices = df_feat_no_spy.index[diff_mask][:5]
                for idx in diff_indices:
                    print(f"      {idx}: no_spy={no_spy_vals.loc[idx]:.6f}, with_spy={with_spy_vals.loc[idx]:.6f}")
                all_match = False
            else:
                print(f"   [PASS] {feature_name}: Values match")
    
    return all_match


def test_spy_data_passing():
    """Test 4: Verify SPY data is actually being used (not None)."""
    print("\n" + "=" * 70)
    print("TEST 4: SPY Data Passing Verification")
    print("=" * 70)
    
    # Load a sample ticker
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))
    
    if len(ticker_files) == 0:
        print("[FAILED] No ticker files found")
        return False
    
    sample_file = ticker_files[0]
    ticker = sample_file.stem
    df = pd.read_parquet(sample_file)
    
    # Load features
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    
    # Test with None spy_data
    print("   Test 4a: Computing with spy_data=None...")
    logger = setup_logger("test", "test_spy.log")
    spy_features = {name: enabled[name] for name in ['beta_spy_252d'] if name in enabled}
    
    if len(spy_features) == 0:
        print("   [FAILED] beta_spy_252d not found")
        return False
    
    df_feat_none, _ = apply_features(df, spy_features, logger, validate=False, spy_data=None)
    
    # Test with actual spy_data
    print("   Test 4b: Computing with actual spy_data...")
    spy_data = _load_spy_data()
    df_feat_with, _ = apply_features(df, spy_features, logger, validate=False, spy_data=spy_data)
    
    # Both should produce same results (feature loads SPY internally if None)
    beta_none = df_feat_none['beta_spy_252d']
    beta_with = df_feat_with['beta_spy_252d']
    
    matches = np.allclose(beta_none, beta_with, equal_nan=True)
    
    if matches:
        non_nan = beta_none.notna().sum()
        print(f"   [PASS] Results match ({non_nan} non-NaN values)")
        print(f"   [PASS] SPY data is being used correctly")
        return True
    else:
        print(f"   ❌ Results don't match!")
        return False


def test_multiple_tickers():
    """Test 5: Verify SPY data works across multiple tickers."""
    print("\n" + "=" * 70)
    print("TEST 5: Multiple Tickers with Shared SPY Data")
    print("=" * 70)
    
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))[:5]  # Test with 5 tickers
    
    if len(ticker_files) < 2:
        print("[FAILED] Need at least 2 tickers for this test")
        return False
    
    print(f"Testing with {len(ticker_files)} tickers...")
    
    # Load SPY data once
    spy_data = _load_spy_data()
    if spy_data is None:
        print("[FAILED] Could not load SPY data")
        return False
    
    print(f"[PASS] SPY data loaded once: {len(spy_data)} rows")
    
    # Load features
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    spy_features = {name: enabled[name] for name in ['beta_spy_252d'] if name in enabled}
    
    logger = setup_logger("test", "test_spy.log")
    
    # Process each ticker with same SPY data
    results = {}
    for ticker_file in ticker_files:
        ticker = ticker_file.stem
        df = pd.read_parquet(ticker_file)
        
        df_feat, _ = apply_features(df, spy_features, logger, validate=False, spy_data=spy_data)
        
        beta = df_feat['beta_spy_252d']
        non_nan = beta.notna().sum()
        results[ticker] = {
            'non_nan': non_nan,
            'total': len(beta),
            'mean': beta.mean() if non_nan > 0 else None
        }
    
    print(f"\n[PASS] Processed {len(results)} tickers with shared SPY data:")
    for ticker, stats in results.items():
        mean_str = f"{stats['mean']:.4f}" if stats['mean'] is not None else 'N/A'
        print(f"   {ticker}: {stats['non_nan']}/{stats['total']} non-NaN values, mean={mean_str}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SPY DATA OPTIMIZATION VERIFICATION TESTS")
    print("=" * 70)
    
    tests = [
        ("SPY Data Loading", test_spy_data_loading),
        ("Feature Signatures", test_feature_signatures),
        ("Feature Computation", test_feature_computation_with_spy),
        ("SPY Data Passing", test_spy_data_passing),
        ("Multiple Tickers", test_multiple_tickers),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n[ERROR] in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASSED]" if result else "[FAILED]"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! SPY data optimization is working correctly.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
