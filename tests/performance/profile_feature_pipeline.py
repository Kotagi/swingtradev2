#!/usr/bin/env python3
"""
Profile the feature pipeline to identify performance bottlenecks.

This script adds timing instrumentation to identify where time is being spent.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
from feature_set_manager import get_feature_set_config_path, DEFAULT_FEATURE_SET
from features.sets.v3_New_Dawn.registry import load_enabled_features
from features.shared.utils import (
    compute_shared_intermediates,
    _get_close_series,
    _get_high_series,
    _get_low_series,
    _get_volume_series,
    _get_open_series,
)
import inspect


def profile_intermediates_computation(df):
    """Profile how long each intermediate takes to compute."""
    print("\n" + "="*80)
    print("PROFILING INTERMEDIATES COMPUTATION")
    print("="*80)
    
    times = {}
    intermediates = {}
    
    # Base series (use utility functions to handle column name variations)
    start = time.perf_counter()
    close = intermediates['_close'] = _get_close_series(df)
    times['_close'] = time.perf_counter() - start
    
    start = time.perf_counter()
    high = intermediates['_high'] = _get_high_series(df)
    times['_high'] = time.perf_counter() - start
    
    start = time.perf_counter()
    low = intermediates['_low'] = _get_low_series(df)
    times['_low'] = time.perf_counter() - start
    
    start = time.perf_counter()
    volume = intermediates['_volume'] = _get_volume_series(df)
    times['_volume'] = time.perf_counter() - start
    
    # Returns
    start = time.perf_counter()
    intermediates['_returns_1d'] = close.pct_change()
    times['_returns_1d'] = time.perf_counter() - start
    
    start = time.perf_counter()
    intermediates['_log_returns_1d'] = np.log(close / close.shift(1))
    times['_log_returns_1d'] = time.perf_counter() - start
    
    # Moving Averages
    for window in [20, 50, 200]:
        start = time.perf_counter()
        intermediates[f'_sma{window}'] = close.rolling(window=window, min_periods=1).mean()
        times[f'_sma{window}'] = time.perf_counter() - start
    
    for span in [20, 50, 200, 12, 26]:
        start = time.perf_counter()
        intermediates[f'_ema{span}'] = close.ewm(span=span, adjust=False).mean()
        times[f'_ema{span}'] = time.perf_counter() - start
    
    # Volatility
    returns = intermediates['_returns_1d']
    for window in [5, 21]:
        start = time.perf_counter()
        intermediates[f'_volatility_{window}d'] = returns.rolling(window=window, min_periods=1).std()
        times[f'_volatility_{window}d'] = time.perf_counter() - start
    
    # True Range & ATR
    start = time.perf_counter()
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    intermediates['_tr'] = tr
    times['_tr'] = time.perf_counter() - start
    
    start = time.perf_counter()
    intermediates['_atr14'] = tr.rolling(window=14, min_periods=1).mean()
    times['_atr14'] = time.perf_counter() - start
    
    # RSI
    for period in [7, 14, 21]:
        start = time.perf_counter()
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean().replace(0, 1e-10)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        intermediates[f'_rsi{period}'] = rsi
        times[f'_rsi{period}'] = time.perf_counter() - start
    
    # 52-week
    start = time.perf_counter()
    intermediates['_high_52w'] = close.rolling(window=252, min_periods=1).max()
    times['_high_52w'] = time.perf_counter() - start
    
    start = time.perf_counter()
    intermediates['_low_52w'] = close.rolling(window=252, min_periods=1).min()
    times['_low_52w'] = time.perf_counter() - start
    
    # Volume
    start = time.perf_counter()
    intermediates['_volume_avg_20d'] = volume.rolling(window=20, min_periods=1).mean()
    times['_volume_avg_20d'] = time.perf_counter() - start
    
    # Resampled (expensive!)
    if isinstance(close.index, pd.DatetimeIndex):
        start = time.perf_counter()
        intermediates['_weekly_close'] = close.resample('W-FRI').last()
        times['_weekly_close'] = time.perf_counter() - start
        
        start = time.perf_counter()
        intermediates['_monthly_close'] = close.resample('ME').last()
        times['_monthly_close'] = time.perf_counter() - start
    
    # Report
    total_time = sum(times.values())
    print(f"\nTotal intermediates computation time: {total_time:.3f}s\n")
    
    # Sort by time
    sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)
    print("Slowest intermediates:")
    for key, elapsed in sorted_times[:15]:
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        print(f"  {key:20s}: {elapsed:8.3f}s ({pct:5.1f}%)")
    
    return intermediates, times, total_time


def profile_feature_computation(df, enabled_features, intermediates):
    """Profile how long each feature takes to compute."""
    print("\n" + "="*80)
    print("PROFILING FEATURE COMPUTATION")
    print("="*80)
    
    # Set up gain_probability cache if any gain_probability features are enabled
    gain_probability_feature_names = [
        'historical_gain_probability',
        'gain_probability_score',
        'gain_regime',
        'gain_consistency',
        'gain_probability_rank',
        'gain_probability_trend',
        'gain_probability_momentum',
        'gain_probability_volatility_adjusted',
        'gain_probability_consistency_rank',
        'gain_regime_transition_probability',
        'volatility_gain_probability_interaction',
        'gain_probability_volatility_regime_interaction',
        'top_features_ensemble',
    ]
    
    gain_probability_cache = {}
    has_gain_probability_features = any(name in enabled_features for name in gain_probability_feature_names)
    
    if has_gain_probability_features:
        from features.sets.v3_New_Dawn.technical import (
            feature_historical_gain_probability,
            feature_gain_probability_score,
            feature_gain_regime,
            feature_gain_consistency,
        )
        
        # Compute cache upfront (this will be timed as part of individual feature calls)
        # But we need it available for dependent features
        print("Setting up gain_probability cache...")
        cache_start = time.perf_counter()
        gain_probability_cache['historical_gain_probability'] = feature_historical_gain_probability(df, target_gain=0.15, horizon=20)
        gain_probability_cache['gain_probability_score'] = feature_gain_probability_score(
            df, target_gain=0.15, horizon=20,
            cached_historical_prob=gain_probability_cache['historical_gain_probability']
        )
        cache_time = time.perf_counter() - cache_start
        print(f"Gain probability cache setup: {cache_time:.3f}s\n")
    
    feature_times = {}
    signature_times = {}
    
    for name, func in enabled_features.items():
        # Time signature inspection
        sig_start = time.perf_counter()
        sig = inspect.signature(func)
        params = sig.parameters
        sig_time = time.perf_counter() - sig_start
        signature_times[name] = sig_time
        
        # Time feature computation
        start = time.perf_counter()
        try:
            kwargs = {}
            if 'spy_data' in params:
                kwargs['spy_data'] = None
            if 'intermediates' in params:
                kwargs['intermediates'] = intermediates
            
            # Pass cached gain_probability features if available
            if has_gain_probability_features:
                # Pass cached historical_gain_probability
                if 'cached_historical_prob' in params and 'historical_gain_probability' in gain_probability_cache:
                    kwargs['cached_historical_prob'] = gain_probability_cache['historical_gain_probability']
                if 'cached_result' in params:
                    # For historical_gain_probability itself, pass cached_result
                    if name == 'historical_gain_probability' and 'historical_gain_probability' in gain_probability_cache:
                        kwargs['cached_result'] = gain_probability_cache['historical_gain_probability']
                
                # Pass cached gain_probability_score
                if 'cached_gain_prob_score' in params and 'gain_probability_score' in gain_probability_cache:
                    kwargs['cached_gain_prob_score'] = gain_probability_cache['gain_probability_score']
                if 'cached_result' in params:
                    # For gain_probability_score itself, pass cached_result
                    if name == 'gain_probability_score' and 'gain_probability_score' in gain_probability_cache:
                        kwargs['cached_result'] = gain_probability_cache['gain_probability_score']
                
                # Pass cached gain_regime (needed for gain_regime_transition_probability)
                if 'cached_gain_regime' in params:
                    # Compute gain_regime if not already cached
                    if 'gain_regime' not in gain_probability_cache:
                        from features.sets.v3_New_Dawn.technical import feature_gain_regime
                        gain_probability_cache['gain_regime'] = feature_gain_regime(
                            df, target_gain=0.15, horizon=20,
                            cached_gain_prob_score=gain_probability_cache.get('gain_probability_score')
                        )
                    kwargs['cached_gain_regime'] = gain_probability_cache['gain_regime']
                
                # Pass cached gain_consistency (needed for gain_probability_consistency_rank)
                if 'cached_gain_consistency' in params:
                    # Compute gain_consistency if not already cached
                    if 'gain_consistency' not in gain_probability_cache:
                        from features.sets.v3_New_Dawn.technical import feature_gain_consistency
                        gain_probability_cache['gain_consistency'] = feature_gain_consistency(
                            df, target_gain=0.15, horizon=20,
                            cached_historical_prob=gain_probability_cache.get('historical_gain_probability')
                        )
                    kwargs['cached_gain_consistency'] = gain_probability_cache['gain_consistency']
            
            if kwargs:
                result = func(df, **kwargs)
            else:
                result = func(df)
            
            elapsed = time.perf_counter() - start
            feature_times[name] = elapsed
        except Exception as e:
            print(f"  ERROR computing {name}: {e}")
            feature_times[name] = -1
    
    # Report
    total_time = sum(t for t in feature_times.values() if t > 0)
    total_sig_time = sum(signature_times.values())
    
    print(f"\nTotal feature computation time: {total_time:.3f}s")
    print(f"Total signature inspection time: {total_sig_time:.3f}s ({total_sig_time/total_time*100:.1f}%)\n")
    
    # Sort by time
    sorted_times = sorted([(k, v) for k, v in feature_times.items() if v > 0], 
                         key=lambda x: x[1], reverse=True)
    print("Slowest 20 features:")
    for name, elapsed in sorted_times[:20]:
        pct = (elapsed / total_time * 100) if total_time > 0 else 0
        uses_intermediates = 'intermediates' in inspect.signature(enabled_features[name]).parameters
        marker = "[I]" if uses_intermediates else "   "
        print(f"  {marker} {name:30s}: {elapsed:8.3f}s ({pct:5.1f}%)")
    
    return feature_times, total_time, total_sig_time


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile feature pipeline performance")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker to profile")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    
    args = parser.parse_args()
    
    # Load data
    clean_data_dir = PROJECT_ROOT / "data" / "clean"
    ticker_file = clean_data_dir / f"{args.ticker}.parquet"
    
    if not ticker_file.exists():
        print(f"Error: {ticker_file} not found")
        sys.exit(1)
    
    print(f"Loading {ticker_file.name}...")
    start = time.perf_counter()
    try:
        df = pd.read_parquet(ticker_file, engine='pyarrow')
    except:
        df = pd.read_parquet(ticker_file)
    load_time = time.perf_counter() - start
    print(f"Loaded {len(df)} rows in {load_time:.3f}s\n")
    
    # Load features
    if args.config:
        config_path = Path(args.config)
    else:
        # Use v3_New_Dawn instead of default v1
        config_path = get_feature_set_config_path("v3_New_Dawn")
    
    enabled_features = load_enabled_features(config_path)
    print(f"Loaded {len(enabled_features)} enabled features\n")
    
    # Profile intermediates
    intermediates, intermediate_times, intermediate_total = profile_intermediates_computation(df)
    
    # Profile features
    feature_times, feature_total, sig_total = profile_feature_computation(df, enabled_features, intermediates)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Data loading:        {load_time:8.3f}s")
    print(f"Intermediates:       {intermediate_total:8.3f}s ({intermediate_total/(load_time+intermediate_total+feature_total)*100:.1f}%)")
    print(f"Feature computation: {feature_total:8.3f}s ({feature_total/(load_time+intermediate_total+feature_total)*100:.1f}%)")
    print(f"Signature inspection: {sig_total:8.3f}s ({sig_total/feature_total*100:.1f}% of feature time)")
    print(f"Total:               {load_time + intermediate_total + feature_total:8.3f}s")
    
    # Count features using intermediates
    uses_intermediates = sum(1 for name, func in enabled_features.items() 
                           if 'intermediates' in inspect.signature(func).parameters)
    print(f"\nFeatures using intermediates: {uses_intermediates}/{len(enabled_features)} ({uses_intermediates/len(enabled_features)*100:.1f}%)")


if __name__ == "__main__":
    main()
