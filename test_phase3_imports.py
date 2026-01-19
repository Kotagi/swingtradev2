#!/usr/bin/env python3
"""Quick test script for Phase 3 - Import Verification"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    print("Testing imports...")
    from features.sets.v1.registry import load_enabled_features
    print("[OK] Successfully imported from features.sets.v1.registry")
    
    from features.sets.v1.technical import feature_price
    print("[OK] Successfully imported from features.sets.v1.technical")
    
    config_path = Path("config/features_v1.yaml")
    if config_path.exists():
        print(f"[OK] Config file exists: {config_path}")
        features = load_enabled_features(str(config_path))
        print(f"[OK] Loaded {len(features)} features from config")
        print(f"  Feature names: {list(features.keys())[:10]}...")  # Show first 10
    else:
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    print("\n[SUCCESS] All import tests passed!")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
