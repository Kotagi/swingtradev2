#!/usr/bin/env python3
"""
feature_pipeline.py

Parallelized feature pipeline with optional full-refresh mode:
  - Reads cleaned Parquet files
  - By default, skips any ticker whose output is up-to-date (caching)
  - With --full, forces recomputation of all tickers
  - Computes enabled features & labels future returns
  - Writes the resulting feature/label Parquet files
Uses Joblib to distribute work across all CPU cores on Windows.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root and src to Python path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utils.logger import setup_logger
from features.registry import load_enabled_features
from utils.labeling import label_future_return
from feature_set_manager import (
    get_feature_set_config_path,
    get_feature_set_data_path,
    feature_set_exists,
    DEFAULT_FEATURE_SET
)


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
    validate: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Apply each enabled feature function to the DataFrame with validation.

    Args:
        df: DataFrame of cleaned OHLCV data.
        enabled_features: Dict[name -> feature_fn].
        logger: Logger for status.
        validate: If True, validate features for NaNs/infinities.

    Returns:
        (df_feat, validation_issues) where:
        - df_feat: original df + feature columns
        - validation_issues: dict mapping feature_name -> list of issues
    """
    validation_issues = {}
    feature_dict = {}  # Collect all features first
    
    for name, func in enabled_features.items():
        try:
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
    label_horizon: int,
    label_threshold: float,
    log_file: str,
    full_refresh: bool
) -> Tuple[str, Optional[str], Dict]:
    """
    Worker: process one ticker end-to-end with optional caching.

    Args:
        file_path: Path to cleaned input Parquet.
        output_path: Directory for feature/label Parquets.
        enabled: Mapping of feature names to functions.
        label_horizon: Days ahead for label generation.
        label_threshold: Threshold for positive label.
        log_file: Shared log file path.
        full_refresh: If True, ignore existing outputs and recompute everything.

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
        'rows_after': 0
    }

    # Caching: skip if up-to-date and not full_refresh
    if not full_refresh and out_file.exists():
        input_mtime = file_path.stat().st_mtime
        output_mtime = out_file.stat().st_mtime
        if output_mtime >= input_mtime:
            logger.debug(f"Skipping {ticker}, up-to-date")
            stats['status'] = 'skipped'
            return (file_path.name, None, stats)

    logger.debug(f"Start processing {ticker}")

    try:
        # 1) Load cleaned data
        df = pd.read_parquet(file_path)
        stats['rows_before'] = len(df)
        
        if df.empty:
            raise ValueError(f"{ticker}: Empty DataFrame")

        # 2) Compute features with validation
        df_feat, validation_issues = apply_features(df, enabled, logger, validate=True)
        stats['features_computed'] = len(enabled) - len(validation_issues)
        stats['features_failed'] = len([v for v in validation_issues.values() if 'Computation error' in str(v)])
        stats['validation_issues'] = sum(len(issues) for issues in validation_issues.values())

        # 3) Generate binary labels
        close_col = 'Close' if 'Close' in df_feat.columns else 'close'
        high_col = 'High' if 'High' in df_feat.columns else 'high'
        df_labeled = label_future_return(
            df_feat,
            close_col=close_col,
            high_col=high_col,
            horizon=label_horizon,
            threshold=label_threshold,
            label_name=f"label_{label_horizon}d"
        )
        stats['rows_after'] = len(df_labeled)

        # 4) Final validation: check for excessive NaNs in final output
        feature_cols = [col for col in df_labeled.columns if col not in ['open', 'high', 'low', 'close', 'volume', f'label_{label_horizon}d']]
        if feature_cols:
            nan_counts = df_labeled[feature_cols].isna().sum(axis=1)
            high_nan_rows = (nan_counts > len(feature_cols) * 0.5).sum()
            if high_nan_rows > 0:
                logger.warning(f"{ticker}: {high_nan_rows} rows with >50% missing features")

        # 5) Write output Parquet
        df_labeled.to_parquet(out_file, index=True)
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
    label_horizon: int,
    label_threshold: float,
    full_refresh: bool
) -> None:
    """
    Entry point: parallelize processing with optional full-refresh.

    Args:
        input_dir: Directory of cleaned Parquet files.
        output_dir: Directory for feature-labeled Parquets.
        config_path: Path to features.yaml toggle file.
        label_horizon: Days ahead for labels.
        label_threshold: Threshold for positive labels.
        full_refresh: If True, recompute all tickers regardless of cache.
    """
    start_time = time.perf_counter()

    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    master_log = "feature_pipeline.log"
    logger = setup_logger("pipeline", master_log)

    enabled = load_enabled_features(config_path)
    logger.info(f"Enabled features: {list(enabled.keys())}")
    if full_refresh:
        logger.info("Full-refresh mode: recomputing all tickers")

    files = sorted(input_path.glob("*.parquet"))
    logger.info(f"{len(files)} tickers to process")
    
    if len(files) == 0:
        logger.warning("No Parquet files found to process")
        return

    # Collect all file paths for progress tracking
    file_list = list(files)
    
    # Parallel dispatch with full_refresh flag
    # Note: tqdm doesn't work well with joblib Parallel, so we'll track progress differently
    logger.info("Starting parallel feature computation...")
    results = Parallel(
        n_jobs=-1,
        backend="multiprocessing",
        verbose=0  # Reduce verbosity, we'll use our own progress tracking
    )(
        delayed(process_file)(
            f, output_path, enabled,
            label_horizon, label_threshold,
            master_log, full_refresh
        )
        for f in file_list
    )

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
    
    elapsed = time.perf_counter() - start_time
    
    # Print comprehensive summary
    logger.info("=" * 70)
    logger.info("=== FEATURE PIPELINE SUMMARY ===")
    logger.info(f"Total tickers: {len(files)}")
    logger.info(f"Processed: {num_processed}")
    logger.info(f"Skipped (cached): {len(skipped)}")
    logger.info(f"Successfully completed: {num_success}")
    logger.info(f"Failed: {len(failures)}")
    logger.info(f"Total features computed: {total_features_computed:,}")
    logger.info(f"Total feature failures: {total_features_failed:,}")
    logger.info(f"Total validation issues: {total_validation_issues:,}")
    logger.info(f"Total rows processed: {total_rows_before:,} -> {total_rows_after:,}")
    logger.info(f"Processing time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    if num_processed > 0:
        logger.info(f"Average time per ticker: {elapsed/num_processed:.3f} seconds")
    
    if failures:
        logger.warning(f"\nFailed tickers ({len(failures)}):")
        for fn, err in failures[:10]:  # Show first 10
            logger.warning(f"  {fn}: {err}")
        if len(failures) > 10:
            logger.warning(f"  ... and {len(failures) - 10} more")
    
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
    parser.add_argument("--horizon",    type=int,   default=5,   help="Label days")
    parser.add_argument("--threshold",  type=float, default=0.0, help="Return thresh")
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
        label_horizon=args.horizon,
        label_threshold=args.threshold,
        full_refresh=args.full_refresh
    )

