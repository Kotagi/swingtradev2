#!/usr/bin/env python3
"""
Test script to validate that optimized feature calculations match original calculations.

This script:
1. Loads a sample ticker's data
2. Computes features with and without intermediates
3. Compares outputs to ensure they match (within floating point tolerance)
4. Reports any differences
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_set_manager import get_feature_set_config_path, DEFAULT_FEATURE_SET
from features.sets.v3_New_Dawn.registry import load_enabled_features
from features.shared.utils import compute_shared_intermediates
import inspect


def compute_feature_without_intermediates(func, df):
    """Compute feature without intermediates (original method)."""
    sig = inspect.signature(func)
    # Remove all cached parameters to test original behavior
    params_to_remove = ['intermediates', 'cached_historical_prob', 'cached_gain_prob_score', 
                       'cached_gain_regime', 'cached_gain_consistency', 'cached_result']
    return func(df)


def compute_feature_with_intermediates(func, df, intermediates, gain_probability_cache=None):
    """Compute feature with intermediates (optimized method)."""
    sig = inspect.signature(func)
    params = sig.parameters
    kwargs = {}
    
    if 'intermediates' in params:
        kwargs['intermediates'] = intermediates
    
    # Add gain_probability cache if available
    if gain_probability_cache:
        if 'cached_historical_prob' in params and 'historical_gain_probability' in gain_probability_cache:
            kwargs['cached_historical_prob'] = gain_probability_cache['historical_gain_probability']
        if 'cached_gain_prob_score' in params and 'gain_probability_score' in gain_probability_cache:
            kwargs['cached_gain_prob_score'] = gain_probability_cache['gain_probability_score']
        if 'cached_gain_regime' in params and 'gain_regime' in gain_probability_cache:
            kwargs['cached_gain_regime'] = gain_probability_cache['gain_regime']
        if 'cached_gain_consistency' in params and 'gain_consistency' in gain_probability_cache:
            kwargs['cached_gain_consistency'] = gain_probability_cache['gain_consistency']
        if 'cached_result' in params:
            # For features that cache themselves
            feature_name = func.__name__.replace('feature_', '')
            if feature_name in gain_probability_cache:
                kwargs['cached_result'] = gain_probability_cache[feature_name]
    
    if kwargs:
        return func(df, **kwargs)
    else:
        return func(df)


def test_feature_integrity(ticker_file: Path, config_path: Path, max_features: int = None):
    """
    Test that features computed with intermediates match original calculations.
    
    Args:
        ticker_file: Path to cleaned Parquet file for a ticker
        config_path: Path to feature config YAML
        max_features: Maximum number of features to test (None = all)
    """
    print(f"\n{'='*80}")
    print(f"Testing Feature Integrity: {ticker_file.name}")
    print(f"{'='*80}\n")
    
    # Load data
    try:
        df = pd.read_parquet(ticker_file, engine='pyarrow')
    except:
        df = pd.read_parquet(ticker_file)
    
    print(f"Loaded {len(df)} rows of data")
    
    # Load enabled features
    enabled_features = load_enabled_features(config_path)
    feature_names = list(enabled_features.keys())
    
    if max_features:
        feature_names = feature_names[:max_features]
        print(f"Testing first {max_features} features (out of {len(enabled_features)} total)")
    else:
        print(f"Testing {len(feature_names)} features")
    
    # Compute intermediates once
    intermediates = compute_shared_intermediates(df)
    print(f"Computed {len(intermediates)} shared intermediates")
    
    # Compute gain_probability cache if any gain_probability features are enabled
    gain_probability_feature_names = [
        'historical_gain_probability', 'gain_probability_score', 'gain_regime', 'gain_consistency',
        'gain_probability_rank', 'gain_probability_trend', 'gain_probability_momentum',
        'gain_probability_volatility_adjusted', 'gain_probability_consistency_rank',
        'gain_regime_transition_probability', 'volatility_gain_probability_interaction',
        'gain_probability_volatility_regime_interaction', 'top_features_ensemble',
    ]
    has_gain_probability = any(name in feature_names for name in gain_probability_feature_names)
    
    gain_probability_cache = {}
    if has_gain_probability:
        from features.sets.v3_New_Dawn.technical import (
            feature_historical_gain_probability,
            feature_gain_probability_score,
        )
        print("Computing gain_probability cache...")
        gain_probability_cache['historical_gain_probability'] = feature_historical_gain_probability(df, target_gain=0.15, horizon=20)
        gain_probability_cache['gain_probability_score'] = feature_gain_probability_score(
            df, target_gain=0.15, horizon=20, 
            cached_historical_prob=gain_probability_cache['historical_gain_probability']
        )
        print(f"Computed {len(gain_probability_cache)} gain_probability cached features\n")
    else:
        print()
    
    # Test each feature
    results = {
        'passed': [],
        'failed': [],
        'skipped': [],
        'errors': []
    }
    
    for i, feature_name in enumerate(feature_names, 1):
        func = enabled_features[feature_name]
        
        try:
            # Check if feature accepts optimization parameters
            sig = inspect.signature(func)
            params = sig.parameters
            
            # Skip if feature doesn't use any optimizations
            has_optimizations = any(param in params for param in [
                'intermediates', 'cached_historical_prob', 'cached_gain_prob_score',
                'cached_gain_regime', 'cached_gain_consistency', 'cached_result'
            ])
            
            if not has_optimizations:
                results['skipped'].append(feature_name)
                if i % 50 == 0 or i == len(feature_names):
                    print(f"[{i}/{len(feature_names)}] {feature_name}: SKIPPED (doesn't use optimizations)")
                continue
            
            # Compute without intermediates/cache (original)
            original = compute_feature_without_intermediates(func, df)
            
            # Compute with intermediates/cache (optimized)
            optimized = compute_feature_with_intermediates(func, df, intermediates, gain_probability_cache)
            
            # Compare results
            # Handle NaN values specially
            both_nan = original.isna() & optimized.isna()
            both_finite = original.notna() & optimized.notna()
            
            # Check finite values match
            if both_finite.any():
                finite_original = original[both_finite]
                finite_optimized = optimized[both_finite]
                
                # Use numpy allclose for floating point comparison
                # rtol=1e-10, atol=1e-10 for very tight tolerance
                matches = np.allclose(
                    finite_original.values,
                    finite_optimized.values,
                    rtol=1e-10,
                    atol=1e-10,
                    equal_nan=True
                )
                
                if not matches:
                    # Find differences
                    diff = np.abs(finite_original.values - finite_optimized.values)
                    max_diff = np.nanmax(diff)
                    max_diff_idx = np.nanargmax(diff)
                    
                    results['failed'].append(feature_name)
                    print(f"[{i}/{len(feature_names)}] {feature_name}: FAILED")
                    print(f"  Max difference: {max_diff:.2e} at index {max_diff_idx}")
                    print(f"  Original value: {finite_original.iloc[max_diff_idx]:.10f}")
                    print(f"  Optimized value: {finite_optimized.iloc[max_diff_idx]:.10f}")
                    continue
            
            # Check NaN positions match
            original_nan = original.isna()
            optimized_nan = optimized.isna()
            nan_matches = (original_nan == optimized_nan).all()
            
            if not nan_matches:
                results['failed'].append(feature_name)
                print(f"[{i}/{len(feature_names)}] {feature_name}: FAILED (NaN positions don't match)")
                continue
            
            # All checks passed
            results['passed'].append(feature_name)
            if (i % 50 == 0) or (i == len(feature_names)):
                print(f"[{i}/{len(feature_names)}] {feature_name}: PASSED")
        
        except Exception as e:
            results['errors'].append((feature_name, str(e)))
            print(f"[{i}/{len(feature_names)}] {feature_name}: ERROR - {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total features tested: {len(feature_names)}")
    print(f"Passed: {len(results['passed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])} (don't use intermediates)")
    print(f"Errors: {len(results['errors'])}")
    
    if results['failed']:
        print(f"\n[FAILED] FAILED FEATURES ({len(results['failed'])}):")
        for name in results['failed']:
            print(f"  - {name}")
    
    if results['errors']:
        print(f"\n[ERROR] ERRORS ({len(results['errors'])}):")
        for name, error in results['errors']:
            print(f"  - {name}: {error}")
    
    if results['failed'] or results['errors']:
        print(f"\n[FAILED] INTEGRITY CHECK FAILED")
        return False
    else:
        print(f"\n[PASSED] ALL TESTS PASSED - Data integrity maintained!")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test feature calculation integrity")
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Ticker symbol to test (default: AAPL)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to test (default: all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to feature config YAML (default: uses default config)"
    )
    
    args = parser.parse_args()
    
    # Find ticker file
    clean_data_dir = PROJECT_ROOT / "data" / "clean"
    ticker_file = clean_data_dir / f"{args.ticker}.parquet"
    
    if not ticker_file.exists():
        print(f"Error: Ticker file not found: {ticker_file}")
        print(f"Available tickers in {clean_data_dir}:")
        for f in sorted(clean_data_dir.glob("*.parquet"))[:10]:
            print(f"  - {f.stem}")
        sys.exit(1)
    
    # Get config path
    if args.config:
        config_path = Path(args.config)
    else:
        # Use v3_New_Dawn instead of default v1
        config_path = get_feature_set_config_path("v3_New_Dawn")
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Run test
    success = test_feature_integrity(ticker_file, config_path, max_features=args.max_features)
    
    sys.exit(0 if success else 1)
