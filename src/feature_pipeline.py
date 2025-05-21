#!/usr/bin/env python3
"""
feature_pipeline.py

Parallelized feature pipeline with file-based caching:
  - Reads cleaned Parquet files
  - Skips any ticker whose output is already up-to-date
  - Computes enabled features & labels future returns
  - Writes the resulting feature/label Parquet files
Uses Joblib to distribute work across all CPU cores on Windows.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from utils.logger import setup_logger
from features.registry import load_enabled_features
from utils.labeling import label_future_return


def apply_features(df: pd.DataFrame, enabled_features: dict, logger) -> pd.DataFrame:
    """
    Apply each enabled feature function to the DataFrame.

    Args:
        df: DataFrame of cleaned OHLCV data.
        enabled_features: Dict[name -> feature_fn].
        logger: Logger for status.
    Returns:
        df_feat: original df + feature columns.
    """
    df_feat = df.copy()
    for name, func in enabled_features.items():
        try:
            df_feat[name] = func(df_feat)
            logger.info(f"{name} computed")
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
    return df_feat


def process_file(
    file_path: Path,
    output_path: Path,
    enabled: dict,
    label_horizon: int,
    label_threshold: float,
    log_file: str
) -> tuple:
    """
    Worker: process one ticker end-to-end with caching.

    Steps:
      A) If output exists and is newer than input, skip.
      1) Read cleaned input Parquet.
      2) Compute features.
      3) Label future returns.
      4) Write output Parquet.

    Returns:
      (filename, error or None)
    """
    ticker = file_path.stem
    logger = setup_logger(ticker, log_file)
    out_file = output_path / f"{ticker}.parquet"

    # —— A) File‐based caching check —— #
    if out_file.exists():
        input_mtime  = file_path.stat().st_mtime
        output_mtime = out_file.stat().st_mtime
        if output_mtime >= input_mtime:
            logger.info(f"Skipping {ticker}, up-to-date")
            return (file_path.name, None)

    logger.info(f"Start processing {ticker}")

    try:
        # 1) Load cleaned data
        df = pd.read_parquet(file_path)

        # 2) Compute features
        df_feat = apply_features(df, enabled, logger)

        # 3) Generate binary labels
        df_labeled = label_future_return(
            df_feat,
            close_col='Close' if 'Close' in df_feat.columns else 'close',
            horizon=label_horizon,
            threshold=label_threshold,
            label_name=f"label_{label_horizon}d"
        )

        # 4) Write output Parquet
        df_labeled.to_parquet(out_file, index=True)
        logger.info(f"Finished {ticker} -> {out_file}")
        return (file_path.name, None)

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}", exc_info=True)
        return (file_path.name, str(e))


def main(
    input_dir: str,
    output_dir: str,
    config_path: str,
    label_horizon: int,
    label_threshold: float
) -> None:
    """
    Entry point: parallelize processing with caching.
    """
    start_time = time.perf_counter()

    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up the master logger
    master_log = "feature_pipeline.log"
    logger = setup_logger("pipeline", master_log)

    # Load which features to compute
    enabled = load_enabled_features(config_path)
    logger.info(f"Enabled features: {list(enabled.keys())}")

    # List all cleaned input Parquets
    files = sorted(input_path.glob("*.parquet"))
    logger.info(f"{len(files)} tickers to process")

    # Parallel dispatch
    results = Parallel(
        n_jobs=-1,
        backend="multiprocessing",
        verbose=5
    )(
        delayed(process_file)(
            f, output_path, enabled, label_horizon, label_threshold, master_log
        )
        for f in files
    )

    # Summarize failures
    failures = [(fn, err) for fn, err in results if err]
    if failures:
        logger.warning(f"{len(failures)}/{len(files)} tickers failed:")
        for fn, err in failures:
            logger.warning(f" - {fn}: {err}")
    else:
        logger.info("All tickers processed or skipped successfully")

    # Final timing
    elapsed = time.perf_counter() - start_time
    logger.info(f"=== Completed in {elapsed:.2f} seconds ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel feature pipeline with caching."
    )
    parser.add_argument("--input-dir",  required=True, help="Cleaned Parquets")
    parser.add_argument("--output-dir", required=True, help="Features_labeled Parquets")
    parser.add_argument("--config",     required=True, help="features.yaml")
    parser.add_argument("--horizon",    type=int,   default=5,   help="Label days ahead")
    parser.add_argument("--threshold",  type=float, default=0.0, help="Return thresh.")
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        label_horizon=args.horizon,
        label_threshold=args.threshold
    )
