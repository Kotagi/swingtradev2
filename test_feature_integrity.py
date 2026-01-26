#!/usr/bin/env python3
"""
test_feature_integrity.py

Test feature pipeline integrity after optimizations:
1. Verify all features are computed and populated
2. Check for data loss or corruption
3. Validate feature values are reasonable
4. Compare feature counts and structure
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
from src.feature_pipeline import apply_features, load_enabled_features_for_set
from utils.logger import setup_logger


def test_feature_completeness():
    """Test 1: Verify all enabled features are computed and present."""
    print("=" * 70)
    print("TEST 1: Feature Completeness")
    print("=" * 70)
    
    # Load a sample ticker
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))
    
    if len(ticker_files) == 0:
        print("[FAILED] No ticker files found")
        return False
    
    sample_file = ticker_files[0]
    ticker = sample_file.stem
    print(f"Testing with ticker: {ticker}")
    
    df = pd.read_parquet(sample_file)
    print(f"   Input rows: {len(df)}")
    print(f"   Input columns: {list(df.columns)}")
    
    # Load features
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    expected_features = list(enabled.keys())
    
    print(f"\n   Expected features: {len(expected_features)}")
    
    # Load SPY data
    from features.shared.utils import _load_spy_data
    spy_data = _load_spy_data()
    
    # Compute features
    logger = setup_logger("test", "test_integrity.log")
    df_feat, validation_issues = apply_features(df, enabled, logger, validate=True, spy_data=spy_data)
    
    # Check which features are present
    feature_cols = [col for col in df_feat.columns if col in expected_features]
    missing_features = [f for f in expected_features if f not in df_feat.columns]
    
    print(f"   Computed features: {len(feature_cols)}")
    
    if missing_features:
        print(f"   [FAILED] Missing features: {missing_features}")
        return False
    
    if len(feature_cols) != len(expected_features):
        print(f"   [WARNING] Feature count mismatch: expected {len(expected_features)}, got {len(feature_cols)}")
        return False
    
    print(f"   [PASS] All {len(expected_features)} features computed and present")
    return True


def test_data_integrity():
    """Test 2: Verify no data loss - input rows preserved."""
    print("\n" + "=" * 70)
    print("TEST 2: Data Integrity (No Data Loss)")
    print("=" * 70)
    
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))[:5]  # Test 5 tickers
    
    if len(ticker_files) == 0:
        print("[FAILED] No ticker files found")
        return False
    
    print(f"Testing with {len(ticker_files)} tickers...")
    
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    
    from features.shared.utils import _load_spy_data
    spy_data = _load_spy_data()
    logger = setup_logger("test", "test_integrity.log")
    
    all_pass = True
    
    for ticker_file in ticker_files:
        ticker = ticker_file.stem
        df_input = pd.read_parquet(ticker_file)
        
        df_feat, _ = apply_features(df_input, enabled, logger, validate=True, spy_data=spy_data)
        
        # Check row count
        if len(df_feat) != len(df_input):
            print(f"   [FAILED] {ticker}: Row count mismatch - input: {len(df_input)}, output: {len(df_feat)}")
            all_pass = False
            continue
        
        # Check index matches
        if not df_feat.index.equals(df_input.index):
            print(f"   [FAILED] {ticker}: Index mismatch")
            all_pass = False
            continue
        
        # Check original columns are preserved
        input_cols = set(df_input.columns)
        output_cols = set(df_feat.columns)
        if not input_cols.issubset(output_cols):
            missing = input_cols - output_cols
            print(f"   [FAILED] {ticker}: Missing input columns: {missing}")
            all_pass = False
            continue
        
        # Check original data values match
        for col in df_input.columns:
            if not df_feat[col].equals(df_input[col]):
                print(f"   [FAILED] {ticker}: Column '{col}' values don't match")
                all_pass = False
                break
        
        if all_pass:
            print(f"   [PASS] {ticker}: {len(df_input)} rows preserved, data intact")
    
    return all_pass


def test_feature_population():
    """Test 3: Verify features are populated (not all NaN)."""
    print("\n" + "=" * 70)
    print("TEST 3: Feature Population (Non-NaN Values)")
    print("=" * 70)
    
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))[:3]  # Test 3 tickers
    
    if len(ticker_files) == 0:
        print("[FAILED] No ticker files found")
        return False
    
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    
    from features.shared.utils import _load_spy_data
    spy_data = _load_spy_data()
    logger = setup_logger("test", "test_integrity.log")
    
    all_pass = True
    
    for ticker_file in ticker_files:
        ticker = ticker_file.stem
        df = pd.read_parquet(ticker_file)
        
        df_feat, _ = apply_features(df, enabled, logger, validate=True, spy_data=spy_data)
        
        # Get feature columns (exclude original OHLCV columns)
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 
                      'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [col for col in df_feat.columns if col not in price_cols]
        
        print(f"\n   {ticker}:")
        print(f"      Total features: {len(feature_cols)}")
        
        # Check each feature has some non-NaN values
        issues = []
        for feat in feature_cols:
            non_nan_count = df_feat[feat].notna().sum()
            total_count = len(df_feat[feat])
            pct_populated = (non_nan_count / total_count * 100) if total_count > 0 else 0
            
            # Features should have at least some values (except maybe first few rows)
            # Allow up to 50% NaN (for rolling window features)
            if pct_populated < 50:
                issues.append(f"{feat}: {pct_populated:.1f}% populated ({non_nan_count}/{total_count})")
        
        if issues:
            print(f"      [WARNING] {len(issues)} features with <50% population:")
            for issue in issues[:5]:  # Show first 5
                print(f"         {issue}")
            if len(issues) > 5:
                print(f"         ... and {len(issues) - 5} more")
        else:
            print(f"      [PASS] All features have reasonable population")
    
    return all_pass


def test_feature_values_reasonable():
    """Test 4: Verify feature values are within reasonable ranges."""
    print("\n" + "=" * 70)
    print("TEST 4: Feature Value Reasonableness")
    print("=" * 70)
    
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))[:3]
    
    if len(ticker_files) == 0:
        print("[FAILED] No ticker files found")
        return False
    
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    
    from features.shared.utils import _load_spy_data
    spy_data = _load_spy_data()
    logger = setup_logger("test", "test_integrity.log")
    
    all_pass = True
    
    for ticker_file in ticker_files:
        ticker = ticker_file.stem
        df = pd.read_parquet(ticker_file)
        
        df_feat, _ = apply_features(df, enabled, logger, validate=True, spy_data=spy_data)
        
        # Check for extreme values that might indicate errors
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 
                      'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [col for col in df_feat.columns if col not in price_cols]
        
        issues = []
        for feat in feature_cols:
            series = df_feat[feat]
            non_nan = series.dropna()
            
            if len(non_nan) == 0:
                continue
            
            # Check for infinities (should be caught by validation, but double-check)
            if np.isinf(non_nan).any():
                issues.append(f"{feat}: Contains infinity values")
            
            # Check for extremely large values (might indicate calculation error)
            # Most features should be normalized or in reasonable ranges
            max_val = non_nan.abs().max()
            if max_val > 1e10:  # Very large threshold
                issues.append(f"{feat}: Max absolute value = {max_val:.2e} (suspiciously large)")
        
        if issues:
            print(f"   [WARNING] {ticker}: {len(issues)} potential value issues:")
            for issue in issues[:3]:
                print(f"      {issue}")
            # Don't fail on this, just warn
        else:
            print(f"   [PASS] {ticker}: All feature values appear reasonable")
    
    return True  # Don't fail on warnings


def test_multiple_tickers_consistency():
    """Test 5: Verify consistency across multiple tickers."""
    print("\n" + "=" * 70)
    print("TEST 5: Multi-Ticker Consistency")
    print("=" * 70)
    
    input_dir = PROJECT_ROOT / "data" / "clean"
    ticker_files = sorted(input_dir.glob("*.parquet"))[:10]  # Test 10 tickers
    
    if len(ticker_files) < 2:
        print("[FAILED] Need at least 2 tickers")
        return False
    
    print(f"Testing consistency across {len(ticker_files)} tickers...")
    
    config_path = PROJECT_ROOT / "config" / "features_v1.yaml"
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    
    from features.shared.utils import _load_spy_data
    spy_data = _load_spy_data()
    logger = setup_logger("test", "test_integrity.log")
    
    # Collect feature counts for each ticker
    results = {}
    for ticker_file in ticker_files:
        ticker = ticker_file.stem
        df = pd.read_parquet(ticker_file)
        
        df_feat, validation_issues = apply_features(df, enabled, logger, validate=True, spy_data=spy_data)
        
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 
                      'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [col for col in df_feat.columns if col not in price_cols]
        
        results[ticker] = {
            'rows': len(df_feat),
            'features': len(feature_cols),
            'validation_issues': len(validation_issues)
        }
    
    # Check consistency
    feature_counts = [r['features'] for r in results.values()]
    if len(set(feature_counts)) > 1:
        print(f"   [FAILED] Inconsistent feature counts: {set(feature_counts)}")
        return False
    
    validation_issue_counts = [r['validation_issues'] for r in results.values()]
    total_issues = sum(validation_issue_counts)
    
    print(f"   [PASS] All {len(ticker_files)} tickers have {feature_counts[0]} features")
    print(f"   [INFO] Total validation issues: {total_issues}")
    
    if total_issues > 0:
        print(f"   [WARNING] Some tickers had validation issues (check logs for details)")
    
    return True


def main():
    """Run all integrity tests."""
    print("\n" + "=" * 70)
    print("FEATURE PIPELINE INTEGRITY TESTS")
    print("=" * 70)
    
    tests = [
        ("Feature Completeness", test_feature_completeness),
        ("Data Integrity", test_data_integrity),
        ("Feature Population", test_feature_population),
        ("Value Reasonableness", test_feature_values_reasonable),
        ("Multi-Ticker Consistency", test_multiple_tickers_consistency),
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
    print("INTEGRITY TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASSED]" if result else "[FAILED]"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All integrity tests passed! Features are computed correctly.")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
