#!/usr/bin/env python3
"""Comprehensive Phase 3 Testing Script"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 80)
print("PHASE 3 TESTING: Feature Set Isolation")
print("=" * 80)

# Test 1: Import Verification
print("\n[TEST 1] Import Verification")
print("-" * 80)
try:
    from features.sets.v1.registry import load_enabled_features
    from features.sets.v1.technical import feature_price
    print("[OK] Successfully imported from features.sets.v1.*")
except ImportError as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Config File Verification
print("\n[TEST 2] Config File Verification")
print("-" * 80)
config_path = Path("config/features_v1.yaml")
train_config_path = Path("config/train_features_v1.yaml")
metadata_path = Path("config/feature_sets_metadata.yaml")

if config_path.exists():
    print(f"[OK] Config file exists: {config_path}")
    features = load_enabled_features(str(config_path))
    print(f"[OK] Loaded {len(features)} features from config")
else:
    print(f"[FAIL] Config file not found: {config_path}")
    sys.exit(1)

if train_config_path.exists():
    print(f"[OK] Train config file exists: {train_config_path}")
else:
    print(f"[WARN] Train config file not found: {train_config_path}")

if metadata_path.exists():
    print(f"[OK] Metadata file exists: {metadata_path}")
else:
    print(f"[WARN] Metadata file not found: {metadata_path}")

# Test 3: Feature Set Manager
print("\n[TEST 3] Feature Set Manager")
print("-" * 80)
try:
    from feature_set_manager import (
        get_feature_set_config_path,
        get_feature_set_data_path,
        feature_set_exists,
        DEFAULT_FEATURE_SET
    )
    
    if feature_set_exists("v1"):
        print(f"[OK] Feature set 'v1' exists")
        config_path_actual = get_feature_set_config_path("v1")
        data_path_actual = get_feature_set_data_path("v1")
        print(f"[OK] Config path: {config_path_actual}")
        print(f"[OK] Data path: {data_path_actual}")
        
        if config_path_actual.exists():
            print(f"[OK] Config path is valid")
        else:
            print(f"[FAIL] Config path does not exist: {config_path_actual}")
            sys.exit(1)
    else:
        print(f"[FAIL] Feature set 'v1' does not exist")
        sys.exit(1)
        
except ImportError as e:
    print(f"[FAIL] Cannot import feature_set_manager: {e}")
    sys.exit(1)

# Test 4: Feature Pipeline Import
print("\n[TEST 4] Feature Pipeline Integration")
print("-" * 80)
try:
    from feature_pipeline import load_enabled_features_for_set
    enabled = load_enabled_features_for_set(str(config_path), "v1")
    print(f"[OK] Feature pipeline can load features for v1: {len(enabled)} features")
except Exception as e:
    print(f"[FAIL] Feature pipeline error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Data Directory Check
print("\n[TEST 5] Data Directory Structure")
print("-" * 80)
data_dir = get_feature_set_data_path("v1")
if data_dir.exists():
    parquet_files = list(data_dir.glob("*.parquet"))
    print(f"[OK] Data directory exists: {data_dir}")
    print(f"[INFO] Found {len(parquet_files)} feature files")
    if len(parquet_files) > 0:
        print(f"[INFO] Sample files: {[f.name for f in parquet_files[:5]]}")
else:
    print(f"[WARN] Data directory does not exist yet: {data_dir}")
    print("[INFO] This is OK - features will be created when pipeline runs")

# Test 6: Check for old imports in active code
print("\n[TEST 6] Old Import Check")
print("-" * 80)
import os
old_imports_found = []
for root, dirs, files in os.walk("src"):
    if "__pycache__" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            filepath = Path(root) / file
            try:
                content = filepath.read_text(encoding="utf-8")
                if "from features.technical import" in content or "from features.registry import" in content:
                    old_imports_found.append(str(filepath))
            except:
                pass

if old_imports_found:
    print(f"[FAIL] Found old imports in: {old_imports_found}")
    sys.exit(1)
else:
    print("[OK] No old imports found in src/ directory")

# Test 7: Verify feature count matches expected
print("\n[TEST 7] Feature Count Verification")
print("-" * 80)
expected_features = 57  # According to PIPELINE_STEPS.md
actual_features = len(enabled)
print(f"[INFO] Expected features: {expected_features}")
print(f"[INFO] Actual features: {actual_features}")
if actual_features >= expected_features:
    print(f"[OK] Feature count is acceptable (>= {expected_features})")
else:
    print(f"[WARN] Feature count is lower than expected")

# Summary
print("\n" + "=" * 80)
print("PHASE 3 TEST SUMMARY")
print("=" * 80)
print("[SUCCESS] All critical tests passed!")
print("\nNext steps:")
print("  1. Run feature pipeline: python src/swing_trade_app.py features --feature-set v1")
print("  2. Train model: python src/swing_trade_app.py train --feature-set v1")
print("  3. Run backtest: python src/swing_trade_app.py backtest --feature-set v1")
print("=" * 80)
