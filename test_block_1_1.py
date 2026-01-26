#!/usr/bin/env python3
"""
Test script for Block 1.1 features (v3_New_Dawn).

Tests the 4 foundation features:
- price
- price_log
- price_vs_ma200
- close_position_in_range
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

print("=" * 80)
print("BLOCK 1.1 TESTING: v3_New_Dawn Foundation Features")
print("=" * 80)

# Test 1: Import Verification
print("\n[TEST 1] Import Verification")
print("-" * 80)
try:
    from features.sets.v3_New_Dawn.registry import FEATURE_REGISTRY, load_enabled_features
    from features.sets.v3_New_Dawn.technical import (
        feature_price,
        feature_price_log,
        feature_price_vs_ma200,
        feature_close_position_in_range,
    )
    print("[OK] Successfully imported all Block 1.1 features")
    print(f"[OK] Registry contains {len(FEATURE_REGISTRY)} features")
    print(f"[OK] Features in registry: {list(FEATURE_REGISTRY.keys())}")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Config File Verification
print("\n[TEST 2] Config File Verification")
print("-" * 80)
config_path = PROJECT_ROOT / "config" / "features_v3_New_Dawn.yaml"
if config_path.exists():
    print(f"[OK] Config file exists: {config_path}")
    try:
        enabled = load_enabled_features(str(config_path))
        print(f"[OK] Loaded {len(enabled)} enabled features from config")
        print(f"[OK] Enabled features: {list(enabled.keys())}")
        if len(enabled) != 4:
            print(f"[WARN] Expected 4 features, got {len(enabled)}")
    except Exception as e:
        print(f"[FAIL] Error loading config: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print(f"[FAIL] Config file not found: {config_path}")
    sys.exit(1)

# Test 3: Feature Computation Test
print("\n[TEST 3] Feature Computation Test")
print("-" * 80)

# Create sample data (250 days to test MA200)
dates = pd.date_range('2023-01-01', periods=250, freq='D')
np.random.seed(42)

# Generate realistic price data
base_price = 100.0
returns = np.random.normal(0.001, 0.02, 250)  # Daily returns ~0.1% mean, 2% std
prices = base_price * np.exp(np.cumsum(returns))

# Create OHLCV DataFrame
df = pd.DataFrame({
    'close': prices,
    'adj close': prices,  # Use adjusted close
    'open': prices * (1 + np.random.normal(0, 0.005, 250)),
    'high': prices * (1 + np.abs(np.random.normal(0.01, 0.01, 250))),
    'low': prices * (1 - np.abs(np.random.normal(0.01, 0.01, 250))),
    'volume': np.random.randint(1000000, 10000000, 250),
}, index=dates)

# Ensure high >= close >= low
df['high'] = df[['close', 'high']].max(axis=1)
df['low'] = df[['close', 'low']].min(axis=1)

print(f"[OK] Created sample DataFrame with {len(df)} rows")
print(f"[OK] Columns: {list(df.columns)}")

# Test each feature
test_results = {}
for feature_name, feature_func in FEATURE_REGISTRY.items():
    try:
        result = feature_func(df)
        test_results[feature_name] = {
            'success': True,
            'length': len(result),
            'nan_count': result.isna().sum(),
            'inf_count': np.isinf(result).sum() if hasattr(result, 'values') else 0,
            'min': result.min() if not result.isna().all() else np.nan,
            'max': result.max() if not result.isna().all() else np.nan,
            'mean': result.mean() if not result.isna().all() else np.nan,
        }
        print(f"[OK] {feature_name}: computed successfully")
        print(f"      Length: {test_results[feature_name]['length']}, "
              f"NaN: {test_results[feature_name]['nan_count']}, "
              f"Inf: {test_results[feature_name]['inf_count']}")
    except Exception as e:
        test_results[feature_name] = {'success': False, 'error': str(e)}
        print(f"[FAIL] {feature_name}: {e}")
        import traceback
        traceback.print_exc()

# Test 4: Validation Checks
print("\n[TEST 4] Validation Checks")
print("-" * 80)

# Check price feature
if 'price' in test_results and test_results['price']['success']:
    price_result = feature_price(df)
    if (price_result > 0).all() or price_result.isna().all():
        print("[OK] price: All values are positive (or NaN)")
    else:
        print("[FAIL] price: Some values are non-positive")

# Check price_log feature
if 'price_log' in test_results and test_results['price_log']['success']:
    price_log_result = feature_price_log(df)
    if np.isfinite(price_log_result).all() or price_log_result.isna().all():
        print("[OK] price_log: All values are finite (or NaN)")
    else:
        print("[FAIL] price_log: Some values are infinite or NaN")

# Check price_vs_ma200 feature
if 'price_vs_ma200' in test_results and test_results['price_vs_ma200']['success']:
    price_vs_ma200_result = feature_price_vs_ma200(df)
    # First 199 should be NaN (insufficient data for 200-day MA)
    first_199_nan = price_vs_ma200_result.iloc[:199].isna().all()
    # After day 200, should have values
    after_200_values = price_vs_ma200_result.iloc[199:].notna().any()
    if first_199_nan and after_200_values:
        print("[OK] price_vs_ma200: First 199 days are NaN, values after day 200")
    else:
        print(f"[WARN] price_vs_ma200: First 199 NaN check: {first_199_nan}, After 200 values: {after_200_values}")

# Check close_position_in_range feature
if 'close_position_in_range' in test_results and test_results['close_position_in_range']['success']:
    position_result = feature_close_position_in_range(df)
    # Should be in [0, 1] range
    in_range = ((position_result >= 0) & (position_result <= 1)).all() or position_result.isna().all()
    if in_range:
        print("[OK] close_position_in_range: All values in [0, 1] range (or NaN)")
    else:
        print("[FAIL] close_position_in_range: Some values outside [0, 1] range")

# Test 5: Feature Pipeline Integration
print("\n[TEST 5] Feature Pipeline Integration")
print("-" * 80)
try:
    from src.feature_pipeline import load_enabled_features_for_set
    enabled = load_enabled_features_for_set(str(config_path), "v3_New_Dawn")
    print(f"[OK] Feature pipeline can load features: {len(enabled)} features")
    print(f"[OK] Features: {list(enabled.keys())}")
except Exception as e:
    print(f"[FAIL] Feature pipeline integration error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
all_passed = all(r.get('success', False) for r in test_results.values())
if all_passed:
    print("[SUCCESS] All features computed successfully!")
    print(f"\nFeature Statistics:")
    for name, stats in test_results.items():
        if stats['success']:
            print(f"  {name}:")
            print(f"    NaN count: {stats['nan_count']}/{stats['length']}")
            print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"    Mean: {stats['mean']:.4f}")
else:
    print("[FAIL] Some features failed to compute")
    for name, stats in test_results.items():
        if not stats.get('success', False):
            print(f"  {name}: {stats.get('error', 'Unknown error')}")

print("\n" + "=" * 80)
print("Next Steps:")
print("1. Test with real data: python src/swing_trade_app.py features --feature-set v3_New_Dawn")
print("2. Verify features in output Parquet files")
print("3. Check for any validation warnings")
print("=" * 80)
