#!/usr/bin/env python3
"""
feature_pipeline.py

Parallelized feature pipeline: reads cleaned CSVs, computes a configurable set of features,
labels future returns, and writes the resulting feature/label CSVs to disk.  Uses Joblib to
distribute work across all CPU cores on a Windows machine (with appropriate __main__ guard).

Usage:
    python src/feature_pipeline.py \
        --input-dir data/clean \
        --output-dir data/features_labeled \
        --config config/features.yaml \
        --horizon 5 \
        --threshold 0.0
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from utils.logger import setup_logger
from features.registry import load_enabled_features
from utils.labeling import label_future_return


def apply_features(
    df: pd.DataFrame,
    enabled_features: dict,
    logger
) -> pd.DataFrame:
    """
    Apply all enabled feature functions to a single DataFrame.

    Args:
        df: Cleaned OHLCV DataFrame indexed by date.
        enabled_features: Mapping from feature name (str) to feature function.
        logger: Logger to record success or failure of each feature.

    Returns:
        A new DataFrame containing the original columns plus one column per feature.
    """
    df_feat = df.copy()
    for name, func in enabled_features.items():
        try:
            # Compute feature and assign as new column
            df_feat[name] = func(df_feat)
            logger.info(f"{name} computed")
        except Exception as e:
            # Log full traceback on failure, but continue with others
            logger.error(f"{name} failed: {e}", exc_info=True)
    return df_feat


def process_file(
    csv_file: Path,
    output_path: Path,
    enabled: dict,
    label_horizon: int,
    label_threshold: float,
    log_file: str
) -> tuple:
    """
    Worker function to process one ticker's CSV end-to-end.

    Steps:
      1. Read cleaned CSV into DataFrame
      2. Compute features via apply_features()
      3. Generate labels with label_future_return()
      4. Write labeled features to output_path

    Args:
        csv_file: Path to the cleaned input CSV for a single ticker.
        output_path: Directory where the output CSV will be saved.
        enabled: Mapping of enabled feature names to functions.
        label_horizon: Number of days ahead to compute the return label.
        label_threshold: Minimum return to flag a positive label.
        log_file: Path to the master log file for this run.

    Returns:
        Tuple of (filename, error_message_or_None).  error_message is None on success.
    """
    # Each worker configures its own logger to the shared log file
    logger = setup_logger(csv_file.stem, log_file)
    ticker = csv_file.stem
    logger.info(f"Start processing {ticker}")

    try:
        # Load cleaned data (assumes datetime index in column 0)
        df = pd.read_csv(csv_file, parse_dates=True, index_col=0)

        # Compute all enabled features
        df_feat = apply_features(df, enabled, logger)

        # Generate binary labels for future returns
        df_labeled = label_future_return(
            df_feat,
            close_col='Close' if 'Close' in df_feat.columns else 'close',
            horizon=label_horizon,
            threshold=label_threshold,
            label_name=f"label_{label_horizon}d"
        )

        # Save the final features+labels CSV
        df_labeled.to_csv(output_path / csv_file.name)
        logger.info(f"Finished {ticker}")
        return (csv_file.name, None)

    except Exception as e:
        # On any error, log and return the exception message
        logger.error(f"Error processing {ticker}: {e}", exc_info=True)
        return (csv_file.name, str(e))


def main(
    input_dir: str,
    output_dir: str,
    config_path: str,
    label_horizon: int,
    label_threshold: float
) -> None:
    """
    Entry point for the parallelized feature pipeline.

    Args:
        input_dir: Directory containing cleaned CSV files.
        output_dir: Directory where feature-labeled CSVs will be written.
        config_path: Path to YAML file listing which features to compute.
        label_horizon: Days ahead for label generation.
        label_threshold: Return threshold to classify positive labels.
    """
    # Start overall timer
    start_time = time.perf_counter()

    # Resolve input and output paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up master logger
    master_log = "feature_pipeline.log"
    logger = setup_logger("pipeline", master_log)

    # Load which features are enabled from the YAML config
    enabled = load_enabled_features(config_path)
    logger.info(f"Enabled features: {list(enabled.keys())}")

    # Gather all cleaned CSVs to process
    csvs = sorted(input_path.glob("*.csv"))
    logger.info(f"{len(csvs)} tickers to process in parallel")

    # Dispatch each ticker to a separate process
    results = Parallel(
        n_jobs=-1,                   # use all available cores
        backend="multiprocessing",   # safe on Windows
        verbose=5                    # show progress
    )(
        delayed(process_file)(
            csv_file,
            output_path,
            enabled,
            label_horizon,
            label_threshold,
            master_log
        )
        for csv_file in csvs
    )

    # Summarize any ticker-level failures
    failures = [(fn, err) for fn, err in results if err]
    if failures:
        logger.warning(f"{len(failures)}/{len(csvs)} tickers failed:")
        for fn, err in failures:
            logger.warning(f" - {fn}: {err}")
    else:
        logger.info("All tickers processed successfully")

    # Log total elapsed time
    elapsed = time.perf_counter() - start_time
    logger.info(f"=== Feature pipeline completed in {elapsed:.2f} seconds ===")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Parallel feature pipeline: compute features & labels for each ticker."
    )
    parser.add_argument("--input-dir",  required=True, help="Clean data folder")
    parser.add_argument("--output-dir", required=True, help="Features_labeled output folder")
    parser.add_argument("--config",     required=True, help="YAML of feature toggles")
    parser.add_argument("--horizon",    type=int,   default=5,   help="Label lookahead days")
    parser.add_argument("--threshold",  type=float, default=0.0, help="Positive return threshold")
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        label_horizon=args.horizon,
        label_threshold=args.threshold
    )
