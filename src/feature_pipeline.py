#!/usr/bin/env python3
"""
feature_pipeline.py

Parallelized feature pipeline with optional full-refresh mode:
  - Reads cleaned Parquet files
  - By default, skips any ticker whose output is up-to-date (caching)
  - With --full, forces recomputation of all tickers
  - Computes enabled features (labels are now calculated during training)
  - Writes the resulting feature Parquet files
Uses Joblib to distribute work across all CPU cores on Windows.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root and src to Python path
# Resolve to absolute path to ensure correct resolution when run as subprocess
_script_file = Path(__file__).resolve()
PROJECT_ROOT = _script_file.parent.parent
SRC_DIR = _script_file.parent
_project_root_str = str(PROJECT_ROOT)
_src_dir_str = str(SRC_DIR)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)
if _src_dir_str not in sys.path:
    sys.path.insert(0, _src_dir_str)

import inspect
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
try:
    from tqdm.auto import tqdm
    from tqdm.contrib.concurrent import process_map
except ImportError:
    from tqdm import tqdm
    process_map = None

from utils.logger import setup_logger
# Labels are now calculated during training, not during feature engineering
from feature_set_manager import (
    get_feature_set_config_path,
    get_feature_set_data_path,
    feature_set_exists,
    DEFAULT_FEATURE_SET
)
from features.shared.utils import _load_spy_data, compute_shared_intermediates


def load_enabled_features_for_set(config_path: str, feature_set: str = None) -> dict:
    """
    Load enabled features for a feature set.
    
    Dynamically imports the registry module for the specified feature set
    and calls load_enabled_features on it.
    
    Args:
        config_path: Path to the feature config YAML file.
        feature_set: Name of the feature set (e.g., "v1"). If None, uses DEFAULT_FEATURE_SET.
    
    Returns:
        Dict mapping feature names to feature functions.
    """
    if feature_set is None:
        feature_set = DEFAULT_FEATURE_SET
    
    # Import the registry module for this feature set
    # Dynamically import based on feature set name
    import importlib
    # Convert feature set name to valid Python module name (spaces/dashes to underscores)
    module_name = feature_set.replace(" ", "_").replace("-", "_")
    registry_module = importlib.import_module(f"features.sets.{module_name}.registry")
    load_enabled_features = registry_module.load_enabled_features
    
    return load_enabled_features(config_path)


def validate_feature(series: pd.Series, feature_name: str) -> Tuple[bool, List[str]]:
    """
    Validate a feature series for common issues.
    
    Args:
        series: Feature series to validate.
        feature_name: Name of the feature for error messages.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for infinities
    n_inf = np.isinf(series).sum()
    if n_inf > 0:
        issues.append(f"{n_inf} infinite values")
    
    # Check for excessive NaNs (>50% missing)
    n_nan = series.isna().sum()
    pct_nan = (n_nan / len(series)) * 100 if len(series) > 0 else 0
    if pct_nan > 50:
        issues.append(f"{pct_nan:.1f}% NaN values ({n_nan}/{len(series)})")
    
    # Check for constant values (no variance)
    if series.notna().sum() > 1:
        if series.nunique() == 1:
            issues.append("constant value (no variance)")
    
    return len(issues) == 0, issues


def apply_features(
    df: pd.DataFrame,
    enabled_features: dict,
    logger,
    validate: bool = True,
    spy_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Apply each enabled feature function to the DataFrame with validation.
    
    Now uses shared intermediate calculations to avoid redundant computations.

    Args:
        df: DataFrame of cleaned OHLCV data.
        enabled_features: Dict[name -> feature_fn].
        logger: Logger for status.
        validate: If True, validate features for NaNs/infinities.
        spy_data: Optional SPY DataFrame to pass to features that need it.

    Returns:
        (df_feat, validation_issues) where:
        - df_feat: original df + feature columns
        - validation_issues: dict mapping feature_name -> list of issues
    """
    validation_issues = {}
    feature_dict = {}  # Collect all features first
    
    # Pre-compute shared intermediates once (Strategy 1: Shared Intermediate Cache)
    intermediates = compute_shared_intermediates(df)
    
    # Strategy 1 (Gain Probability): Cache expensive gain_probability features
    # Check if any gain_probability features are enabled
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
        # Import here to avoid circular imports
        from features.sets.v3_New_Dawn.technical import (
            feature_historical_gain_probability,
            feature_gain_probability_score,
        )
        
        # Compute historical_gain_probability once (expensive - 58 seconds)
        logger.debug("Computing historical_gain_probability (cached for reuse)")
        gain_probability_cache['historical_gain_probability'] = feature_historical_gain_probability(df, target_gain=0.15, horizon=20)
        
        # Compute gain_probability_score once (depends on historical_gain_probability)
        # Use cached historical_gain_probability to avoid recomputing
        logger.debug("Computing gain_probability_score (cached for reuse)")
        gain_probability_cache['gain_probability_score'] = feature_gain_probability_score(
            df, target_gain=0.15, horizon=20, 
            cached_historical_prob=gain_probability_cache['historical_gain_probability']
        )
    
    # Cache signal features for signal_ensemble_score and related features
    signal_cache = {}
    signal_feature_names = [
        'signal_ensemble_score',
        'signal_quality_score',
        'false_positive_risk',
    ]
    has_signal_features = any(name in enabled_features for name in signal_feature_names)
    
    if has_signal_features:
        # Import here to avoid circular imports
        from features.sets.v3_New_Dawn.technical import (
            feature_signal_consistency,
            feature_signal_confirmation,
            feature_signal_strength,
            feature_signal_timing,
            feature_signal_risk_reward,
            feature_signal_historical_success,
            feature_signal_quality_score,
        )
        
        # Compute signal features once (they're called multiple times)
        logger.debug("Computing signal features (cached for reuse)")
        signal_cache['signal_consistency'] = feature_signal_consistency(df)
        signal_cache['signal_confirmation'] = feature_signal_confirmation(df)
        signal_cache['signal_strength'] = feature_signal_strength(df)
        signal_cache['signal_timing'] = feature_signal_timing(df)
        signal_cache['signal_risk_reward'] = feature_signal_risk_reward(df)
        signal_cache['signal_historical_success'] = feature_signal_historical_success(df)
        
        # Compute signal_quality_score (depends on the above)
        signal_cache['signal_quality_score'] = feature_signal_quality_score(
            df,
            cached_signal_consistency=signal_cache['signal_consistency'],
            cached_signal_confirmation=signal_cache['signal_confirmation'],
            cached_signal_strength=signal_cache['signal_strength'],
            cached_signal_timing=signal_cache['signal_timing'],
            cached_signal_risk_reward=signal_cache['signal_risk_reward']
        )
    
    # Pre-compute signal_ensemble_score if needed (depends on cached signal features)
    if has_signal_features and 'signal_ensemble_score' in enabled_features:
        from features.sets.v3_New_Dawn.technical import feature_signal_ensemble_score
        logger.debug("Computing signal_ensemble_score (cached for reuse)")
        signal_cache['signal_ensemble_score'] = feature_signal_ensemble_score(
            df,
            cached_signal_quality_score=signal_cache.get('signal_quality_score'),
            cached_signal_strength=signal_cache.get('signal_strength'),
            cached_signal_confirmation=signal_cache.get('signal_confirmation'),
            cached_signal_timing=signal_cache.get('signal_timing'),
            cached_signal_historical_success=signal_cache.get('signal_historical_success')
        )
    
    for name, func in enabled_features.items():
        try:
            # Check function signature to determine which parameters to pass
            sig = inspect.signature(func)
            params = sig.parameters
            
            # Build arguments based on what the function accepts
            kwargs = {}
            if 'spy_data' in params:
                kwargs['spy_data'] = spy_data
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
                        logger.debug("Computing gain_regime (cached for reuse)")
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
                        logger.debug("Computing gain_consistency (cached for reuse)")
                        gain_probability_cache['gain_consistency'] = feature_gain_consistency(
                            df, target_gain=0.15, horizon=20,
                            cached_historical_prob=gain_probability_cache.get('historical_gain_probability')
                        )
                    kwargs['cached_gain_consistency'] = gain_probability_cache['gain_consistency']
            
            # Pass cached signal features if available
            if has_signal_features:
                # Pass cached signal_quality_score
                if 'cached_signal_quality_score' in params and 'signal_quality_score' in signal_cache:
                    kwargs['cached_signal_quality_score'] = signal_cache['signal_quality_score']
                if 'cached_result' in params:
                    # For signal_quality_score itself, pass cached_result
                    if name == 'signal_quality_score' and 'signal_quality_score' in signal_cache:
                        kwargs['cached_result'] = signal_cache['signal_quality_score']
                
                # Pass cached signal components for signal_quality_score
                if 'cached_signal_consistency' in params and 'signal_consistency' in signal_cache:
                    kwargs['cached_signal_consistency'] = signal_cache['signal_consistency']
                if 'cached_signal_confirmation' in params and 'signal_confirmation' in signal_cache:
                    kwargs['cached_signal_confirmation'] = signal_cache['signal_confirmation']
                if 'cached_signal_strength' in params and 'signal_strength' in signal_cache:
                    kwargs['cached_signal_strength'] = signal_cache['signal_strength']
                if 'cached_signal_timing' in params and 'signal_timing' in signal_cache:
                    kwargs['cached_signal_timing'] = signal_cache['signal_timing']
                if 'cached_signal_risk_reward' in params and 'signal_risk_reward' in signal_cache:
                    kwargs['cached_signal_risk_reward'] = signal_cache['signal_risk_reward']
                if 'cached_signal_historical_success' in params and 'signal_historical_success' in signal_cache:
                    kwargs['cached_signal_historical_success'] = signal_cache['signal_historical_success']
                
                # Pass cached signal_ensemble_score components
                if 'cached_signal_ensemble_score' in params and 'signal_ensemble_score' in signal_cache:
                    kwargs['cached_signal_ensemble_score'] = signal_cache['signal_ensemble_score']
                
                # Pass cached_result for signal_ensemble_score itself
                if 'cached_result' in params:
                    if name == 'signal_ensemble_score' and 'signal_ensemble_score' in signal_cache:
                        kwargs['cached_result'] = signal_cache['signal_ensemble_score']
            
            # Call feature function with appropriate arguments
            if kwargs:
                feature_series = func(df, **kwargs)
            else:
                feature_series = func(df)
            
            # Validate feature output
            if validate:
                is_valid, issues = validate_feature(feature_series, name)
                if not is_valid:
                    validation_issues[name] = issues
                    # Log warnings for validation issues
                    for issue in issues:
                        logger.warning(f"{name}: {issue}")
                    # Replace infinities with NaN for downstream handling
                    feature_series = feature_series.replace([np.inf, -np.inf], np.nan)
            
            feature_dict[name] = feature_series
            logger.debug(f"{name} computed")
            
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
            validation_issues[name] = [f"Computation error: {str(e)}"]
            # Add NaN column on failure to maintain DataFrame structure
            feature_dict[name] = pd.Series(np.nan, index=df.index, name=name)
    
    # Join all features at once using pd.concat (much faster than adding one by one)
    if feature_dict:
        feature_df = pd.DataFrame(feature_dict, index=df.index)
        df_feat = pd.concat([df, feature_df], axis=1)
    else:
        df_feat = df.copy()
    
    return df_feat, validation_issues


def process_file(
    file_path: Path,
    output_path: Path,
    enabled: dict,
    log_file: str,
    full_refresh: bool,
    spy_data: Optional[pd.DataFrame] = None
) -> Tuple[str, Optional[str], Dict]:
    """
    Worker: process one ticker end-to-end with optional caching and incremental updates.
    
    Supports incremental feature engineering:
    - Adds new features for existing dates without rebuilding everything
    - Adds new dates for all enabled features when new trading days are available
    - Only rebuilds what's necessary when full_refresh=False

    Args:
        file_path: Path to cleaned input Parquet.
        output_path: Directory for feature Parquets.
        enabled: Mapping of feature names to functions.
        log_file: Shared log file path.
        full_refresh: If True, ignore existing outputs and recompute everything.
        spy_data: Optional SPY DataFrame to pass to features that need it.

    Returns:
        (filename, error_message_or_None, stats_dict)
    """
    ticker = file_path.stem
    logger = setup_logger(ticker, log_file)
    out_file = output_path / f"{ticker}.parquet"
    
    stats = {
        'status': 'success',
        'features_computed': 0,
        'features_failed': 0,
        'validation_issues': 0,
        'rows_before': 0,
        'rows_after': 0,
        'new_features_added': 0,
        'new_dates_added': 0,
        'incremental_mode': False
    }

    # 1) Load cleaned input data
    try:
        try:
            df_input = pd.read_parquet(file_path, engine='pyarrow')
        except (ImportError, ValueError):
            df_input = pd.read_parquet(file_path)
        
        if df_input.empty:
            raise ValueError(f"{ticker}: Empty DataFrame")
        
        # Ensure index is datetime
        if not isinstance(df_input.index, pd.DatetimeIndex):
            df_input.index = pd.to_datetime(df_input.index)
        
        stats['rows_before'] = len(df_input)
    except Exception as e:
        stats['status'] = 'failed'
        logger.error(f"Error loading input data for {ticker}: {e}", exc_info=True)
        return (file_path.name, str(e), stats)

    # 2) Check for incremental update opportunities
    df_existing = None
    needs_full_rebuild = full_refresh
    
    if not full_refresh and out_file.exists():
        try:
            # Load existing feature file
            try:
                df_existing = pd.read_parquet(out_file, engine='pyarrow')
            except (ImportError, ValueError):
                df_existing = pd.read_parquet(out_file)
            
            # Ensure index is datetime
            if not isinstance(df_existing.index, pd.DatetimeIndex):
                df_existing.index = pd.to_datetime(df_existing.index)
            
            # Check if input file was modified (simple cache check)
            input_mtime = file_path.stat().st_mtime
            output_mtime = out_file.stat().st_mtime
            
            # Identify what needs to be computed
            price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            existing_feature_cols = [col for col in df_existing.columns if col not in price_cols]
            enabled_feature_names = set(enabled.keys())
            existing_feature_names = set(existing_feature_cols)
            
            # Find new features (enabled but not in existing)
            new_features = enabled_feature_names - existing_feature_names
            
            # Find new dates (in input but not in existing)
            existing_dates = set(df_existing.index)
            input_dates = set(df_input.index)
            new_dates = input_dates - existing_dates
            
            # Check if we can skip entirely (no new features, no new dates, and output is newer)
            if not new_features and not new_dates and output_mtime >= input_mtime:
                logger.debug(f"Skipping {ticker}, fully up-to-date")
                stats['status'] = 'skipped'
                return (file_path.name, None, stats)
            
            # Determine if we need incremental update or full rebuild
            if new_features or new_dates:
                stats['incremental_mode'] = True
                stats['new_features_added'] = len(new_features)
                stats['new_dates_added'] = len(new_dates)
                
                if new_features:
                    logger.info(f"{ticker}: Found {len(new_features)} new feature(s) to compute: {sorted(new_features)[:5]}{'...' if len(new_features) > 5 else ''}")
                if new_dates:
                    logger.info(f"{ticker}: Found {len(new_dates)} new date(s) to compute: {min(new_dates) if new_dates else 'N/A'} to {max(new_dates) if new_dates else 'N/A'}")
                
                # If input file was modified, we might need to recompute features that depend on historical data
                # For now, we'll do incremental updates. Full rebuild only if explicitly requested.
                needs_full_rebuild = False
            else:
                # No new features or dates, but input was modified - might need to check for data corrections
                # For safety, if input is significantly newer, do full rebuild
                if input_mtime > output_mtime:
                    time_diff = input_mtime - output_mtime
                    # If input is more than 1 hour newer, assume data corrections and rebuild
                    if time_diff > 3600:
                        logger.info(f"{ticker}: Input file significantly newer ({time_diff:.0f}s), doing full rebuild")
                        needs_full_rebuild = True
                    else:
                        logger.debug(f"{ticker}: Input file slightly newer, but no new features/dates - skipping")
                        stats['status'] = 'skipped'
                        return (file_path.name, None, stats)
        except Exception as e:
            logger.warning(f"{ticker}: Error reading existing feature file, will rebuild: {e}")
            df_existing = None
            needs_full_rebuild = True

    # 3) Compute features (incremental or full)
    try:
        if needs_full_rebuild or df_existing is None:
            # Full rebuild: compute all enabled features for all dates
            logger.debug(f"Full rebuild for {ticker}")
            df_feat, validation_issues = apply_features(df_input, enabled, logger, validate=True, spy_data=spy_data)
            stats['features_computed'] = len(enabled) - len(validation_issues)
        else:
            # Incremental update: compute only what's needed
            logger.debug(f"Incremental update for {ticker}")
            
            # Identify what to compute
            new_features = set(enabled.keys()) - set([col for col in df_existing.columns if col not in price_cols])
            new_dates = set(df_input.index) - set(df_existing.index)
            
            validation_issues = {}
            features_to_compute = {}
            
            if new_features and new_dates:
                # Case 1: Both new features and new dates
                # Compute new features for existing dates only, and all features for new dates
                logger.debug(f"Computing {len(new_features)} new features for existing dates, and all features for {len(new_dates)} new dates")
                
                # Get existing dates that are also in input
                existing_dates_set = set(df_existing.index) & set(df_input.index)
                df_existing_dates = df_input.loc[list(existing_dates_set)]
                
                # Compute new features for existing dates only
                new_features_dict = {name: enabled[name] for name in new_features}
                df_new_feat_existing, new_validation = apply_features(df_existing_dates, new_features_dict, logger, validate=True, spy_data=spy_data)
                
                # Compute all enabled features for new dates only
                df_new_dates = df_input.loc[list(new_dates)]
                df_new_dates_feat, new_dates_validation = apply_features(df_new_dates, enabled, logger, validate=True, spy_data=spy_data)
                
                validation_issues.update(new_validation)
                validation_issues.update(new_dates_validation)
                
                # Merge: start with existing, add new features for existing dates, then add new dates
                # First, ensure price columns come from input (source of truth)
                price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                existing_feature_cols = [col for col in df_existing.columns if col not in price_cols]
                
                # Start with existing features (excluding price columns)
                df_feat = df_existing[existing_feature_cols].copy()
                
                # Add new feature columns for existing dates only
                for feat_name in new_features:
                    if feat_name in df_new_feat_existing.columns:
                        df_feat[feat_name] = df_new_feat_existing[feat_name]
                
                # Add/update rows for new dates (all features including new ones)
                # Extract only feature columns from new_dates_feat (price columns will come from input)
                new_dates_feat_cols = [col for col in df_new_dates_feat.columns if col not in price_cols]
                df_feat = pd.concat([df_feat, df_new_dates_feat[new_dates_feat_cols]])
                df_feat = df_feat[~df_feat.index.duplicated(keep='last')]  # Keep new dates version
                df_feat = df_feat.sort_index()
                
                # Add price columns from input (source of truth)
                for price_col in price_cols:
                    if price_col in df_input.columns:
                        df_feat[price_col] = df_input[price_col]
                
                stats['features_computed'] = len(new_features) + len(enabled)
                
            elif new_features:
                # Case 2: Only new features (no new dates)
                # Compute new features for existing dates only
                logger.debug(f"Computing {len(new_features)} new features for existing dates")
                
                # Use existing dates that are also in input (in case input was trimmed)
                common_dates = set(df_existing.index) & set(df_input.index)
                df_existing_dates = df_input.loc[list(common_dates)]
                
                new_features_dict = {name: enabled[name] for name in new_features}
                df_new_feat, new_validation = apply_features(df_existing_dates, new_features_dict, logger, validate=True, spy_data=spy_data)
                
                validation_issues.update(new_validation)
                
                # Merge: add new feature columns to existing
                # Ensure price columns come from input (source of truth)
                price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                existing_feature_cols = [col for col in df_existing.columns if col not in price_cols]
                
                # Start with existing features (excluding price columns)
                df_feat = df_existing[existing_feature_cols].copy()
                
                # Add price columns from input (source of truth)
                for price_col in price_cols:
                    if price_col in df_input.columns:
                        df_feat[price_col] = df_input[price_col]
                
                # Add new feature columns
                for feat_name in new_features:
                    if feat_name in df_new_feat.columns:
                        df_feat[feat_name] = df_new_feat[feat_name]
                
                stats['features_computed'] = len(new_features)
                
            elif new_dates:
                # Case 3: Only new dates (no new features)
                # Compute all enabled features for new dates only
                logger.debug(f"Computing all features for {len(new_dates)} new dates")
                
                df_new_dates = df_input.loc[list(new_dates)]
                df_new_dates_feat, new_dates_validation = apply_features(df_new_dates, enabled, logger, validate=True, spy_data=spy_data)
                
                validation_issues.update(new_dates_validation)
                
                # Merge: concatenate new dates to existing
                # Ensure price columns come from input (source of truth)
                price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                existing_feature_cols = [col for col in df_existing.columns if col not in price_cols]
                
                # Start with existing features (excluding price columns)
                df_feat = df_existing[existing_feature_cols].copy()
                
                # Extract only feature columns from new_dates_feat (price columns will come from input)
                new_dates_feat_cols = [col for col in df_new_dates_feat.columns if col not in price_cols]
                df_feat = pd.concat([df_feat, df_new_dates_feat[new_dates_feat_cols]])
                df_feat = df_feat[~df_feat.index.duplicated(keep='last')]  # Remove duplicates, keep new
                df_feat = df_feat.sort_index()
                
                # Add price columns from input (source of truth)
                for price_col in price_cols:
                    if price_col in df_input.columns:
                        df_feat[price_col] = df_input[price_col]
                
                stats['features_computed'] = len(enabled)
            else:
                # No new features or dates - should have been caught earlier, but handle gracefully
                logger.debug(f"No new features or dates for {ticker}")
                df_feat = df_existing.copy()
                stats['features_computed'] = 0
        
        stats['features_failed'] = len([v for v in validation_issues.values() if 'Computation error' in str(v)])
        stats['validation_issues'] = sum(len(issues) for issues in validation_issues.values())
        stats['rows_after'] = len(df_feat)

        # 4) Final validation: check for excessive NaNs in final output
        price_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        feature_cols = [col for col in df_feat.columns if col not in price_cols]
        if feature_cols:
            nan_counts = df_feat[feature_cols].isna().sum(axis=1)
            high_nan_rows = (nan_counts > len(feature_cols) * 0.5).sum()
            if high_nan_rows > 0:
                logger.warning(f"{ticker}: {high_nan_rows} rows with >50% missing features")

        # 5) Write output Parquet (features only, no labels, use PyArrow engine for faster I/O)
        try:
            df_feat.to_parquet(out_file, index=True, engine='pyarrow')
        except (ImportError, ValueError):
            df_feat.to_parquet(out_file, index=True)
        
        if stats['incremental_mode']:
            logger.debug(f"Finished incremental update for {ticker} -> {out_file} (added {stats['new_features_added']} features, {stats['new_dates_added']} dates)")
        else:
            logger.debug(f"Finished {ticker} -> {out_file}")
        
        return (file_path.name, None, stats)

    except Exception as e:
        stats['status'] = 'failed'
        logger.error(f"Error processing {ticker}: {e}", exc_info=True)
        return (file_path.name, str(e), stats)


def main(
    input_dir: str,
    output_dir: str,
    config_path: str,
    full_refresh: bool
) -> None:
    """
    Entry point: parallelize processing with optional full-refresh.
    
    Note: Labels are now calculated during training, not during feature engineering.

    Args:
        input_dir: Directory of cleaned Parquet files.
        output_dir: Directory for feature Parquets.
        config_path: Path to features.yaml toggle file.
        full_refresh: If True, recompute all tickers regardless of cache.
    """
    start_time = time.perf_counter()

    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    master_log = "outputs/logs/feature_pipeline.log"
    logger = setup_logger("pipeline", master_log)

    # Determine feature set from config path or use default
    # Extract feature set from config path if it matches pattern
    feature_set = DEFAULT_FEATURE_SET
    config_path_obj = Path(config_path)
    if "features_v" in config_path_obj.name:
        # Extract feature set from filename (e.g., "features_v1.yaml" -> "v1")
        feature_set = config_path_obj.stem.replace("features_", "")
    elif config_path_obj.name == "features.yaml":
        # Old default config, use v3_New_Dawn
        feature_set = "v3_New_Dawn"
    
    enabled = load_enabled_features_for_set(config_path, feature_set)
    logger.info(f"Using feature set: {feature_set}")
    logger.info(f"Enabled features: {list(enabled.keys())}")
    if full_refresh:
        logger.info("Full-refresh mode: recomputing all tickers")

    # Pre-load SPY data once to share across all worker processes
    # This eliminates redundant I/O and parsing for market context features
    logger.info("Loading SPY data for market context features...")
    spy_data = _load_spy_data()
    if spy_data is not None:
        logger.info(f"SPY data loaded: {len(spy_data)} rows from {spy_data.index.min()} to {spy_data.index.max()}")
    else:
        logger.warning("SPY data not available - market context features will return NaN")

    files = sorted(input_path.glob("*.parquet"))
    logger.info(f"{len(files)} tickers to process")
    
    if len(files) == 0:
        logger.warning("No Parquet files found to process")
        return

    # Collect all file paths for progress tracking
    file_list = list(files)
    
    # Parallel dispatch with full_refresh flag and progress bar
    logger.info("Starting parallel feature computation...")
    
    # Process in batches for more real-time progress updates
    # CPU-aware batch size: Use available CPU cores for optimal parallelization
    # Formula: min(2x CPU cores, 10% of files, max 50) with minimum of 10 for better CPU utilization
    num_cores = os.cpu_count() or 8  # Default to 8 if detection fails
    optimal_batch_size = num_cores * 2  # 2x cores for good parallelism
    # Ensure minimum of 10 workers (when enough files available) for better parallelization
    min_batch = min(10, len(file_list))  # Don't exceed available files
    batch_size = max(min_batch, min(optimal_batch_size, len(file_list) // 10, 50))
    logger.info(f"Using batch size: {batch_size} (CPU cores: {num_cores})")
    results = []
    stats_tracker = {'success': 0, 'skipped': 0, 'failed': 0}
    
    # Create progress bar
    with tqdm(total=len(file_list), desc="Building features", unit="ticker", ncols=100,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
        
        # Process files in batches for real-time progress updates
        for i in range(0, len(file_list), batch_size):
            batch = file_list[i:i + batch_size]
            
            # Process this batch in parallel (pass spy_data to each worker)
            batch_results = Parallel(
                n_jobs=-1,
                backend="multiprocessing",
                verbose=0
            )(
                delayed(process_file)(
                    f, output_path, enabled,
                    master_log, full_refresh, spy_data
                )
                for f in batch
            )
            
            # Update progress bar as we process batch results
            for result in batch_results:
                fn, err, stats = result
                status = stats.get('status', 'success')
                
                if status == 'skipped':
                    stats_tracker['skipped'] += 1
                elif err:
                    stats_tracker['failed'] += 1
                else:
                    stats_tracker['success'] += 1
                
                # Update progress bar with current stats
                pbar.set_postfix_str(f"S:{stats_tracker['success']} F:{stats_tracker['failed']} X:{stats_tracker['skipped']}")
                pbar.update(1)
                results.append(result)

    # Process results and collect statistics
    failures = []
    skipped = []
    all_stats = {}
    
    for result in results:
        fn, err, stats = result
        all_stats[fn] = stats
        if stats['status'] == 'skipped':
            skipped.append(fn)
        elif err:
            failures.append((fn, err))
    
    # Enhanced summary statistics
    num_processed = len(files) - len(skipped)
    num_success = num_processed - len(failures)
    total_features_computed = sum(s.get('features_computed', 0) for s in all_stats.values())
    total_features_failed = sum(s.get('features_failed', 0) for s in all_stats.values())
    total_validation_issues = sum(s.get('validation_issues', 0) for s in all_stats.values())
    total_rows_before = sum(s.get('rows_before', 0) for s in all_stats.values())
    total_rows_after = sum(s.get('rows_after', 0) for s in all_stats.values())
    
    # Incremental update statistics
    incremental_updates = sum(1 for s in all_stats.values() if s.get('incremental_mode', False))
    total_new_features_added = sum(s.get('new_features_added', 0) for s in all_stats.values())
    total_new_dates_added = sum(s.get('new_dates_added', 0) for s in all_stats.values())
    
    elapsed = time.perf_counter() - start_time
    
    # Print comprehensive summary
    logger.info("=" * 70)
    logger.info("=== FEATURE PIPELINE SUMMARY ===")
    logger.info(f"Total tickers: {len(files)}")
    logger.info(f"Processed: {num_processed}")
    logger.info(f"Skipped (cached): {len(skipped)}")
    logger.info(f"Successfully completed: {num_success}")
    logger.info(f"Failed: {len(failures)}")
    if incremental_updates > 0:
        logger.info(f"Incremental updates: {incremental_updates} ticker(s)")
        logger.info(f"  - New features added: {total_new_features_added:,}")
        logger.info(f"  - New dates added: {total_new_dates_added:,}")
    logger.info(f"Total features computed: {total_features_computed:,}")
    logger.info(f"Total feature failures: {total_features_failed:,}")
    logger.info(f"Total validation issues: {total_validation_issues:,}")
    logger.info(f"Total rows processed: {total_rows_before:,} -> {total_rows_after:,}")
    logger.info(f"Processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    if num_processed > 0:
        logger.info(f"Average time per ticker: {elapsed/num_processed:.3f} seconds")
    
    if failures:
        logger.warning(f"\nFailed tickers ({len(failures)}):")
        # Also print to stdout so GUI can capture it
        print(f"\nFailed tickers ({len(failures)}):", flush=True)
        for fn, err in failures[:10]:  # Show first 10
            ticker_name = Path(fn).stem if fn else "unknown"
            logger.warning(f"  {ticker_name}: {err}")
            print(f"  {ticker_name}: {err}", flush=True)
        if len(failures) > 10:
            logger.warning(f"  ... and {len(failures) - 10} more")
            print(f"  ... and {len(failures) - 10} more", flush=True)
    
    if skipped:
        logger.info(f"\nSkipped (already up-to-date): {len(skipped)} tickers")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature pipeline with optional full-refresh and feature set support."
    )
    parser.add_argument("--input-dir",  help="Cleaned Parquet directory")
    parser.add_argument("--output-dir", help="Features Parquet directory")
    parser.add_argument("--config",     help="features.yaml config file path")
    parser.add_argument(
        "--feature-set",
        type=str,
        help=f"Feature set name (e.g., 'v1', 'v2'). If specified, automatically sets --config and --output-dir. Default: '{DEFAULT_FEATURE_SET}'"
    )
    # Labels are now calculated during training, not during feature engineering
    parser.add_argument(
        "--full", "--force-full",
        action="store_true",
        dest="full_refresh",
        help="Recompute all tickers, ignoring cache"
    )
    args = parser.parse_args()

    # Handle feature set vs explicit paths
    if args.feature_set:
        # Use feature set
        if not feature_set_exists(args.feature_set):
            print(f"Error: Feature set '{args.feature_set}' does not exist.")
            print(f"Available feature sets: {', '.join([DEFAULT_FEATURE_SET])}")
            sys.exit(1)
        
        config_path = str(get_feature_set_config_path(args.feature_set))
        output_dir = str(get_feature_set_data_path(args.feature_set))
        
        # input_dir defaults to data/clean if not specified
        input_dir = args.input_dir or str(Path(__file__).parent.parent / "data" / "clean")
        
        print(f"Using feature set: {args.feature_set}")
        print(f"  Config: {config_path}")
        print(f"  Output: {output_dir}")
    else:
        # Use explicit paths (backward compatibility)
        if not args.input_dir or not args.output_dir or not args.config:
            print("Error: Either --feature-set must be specified, or all of --input-dir, --output-dir, and --config must be provided.")
            sys.exit(1)
        
        input_dir = args.input_dir
        output_dir = args.output_dir
        config_path = args.config

    main(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config_path,
        full_refresh=args.full_refresh
    )

