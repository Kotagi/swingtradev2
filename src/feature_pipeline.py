# src/feature_pipeline.py

#!/usr/bin/env python3
"""
feature_pipeline.py

Reads cleaned ticker CSVs, computes a configurable set of features, labels future returns,
and writes the resulting feature/label CSVs. Measures total runtime and logs progress.

Usage:
    python src/feature_pipeline.py --input-dir data/clean --output-dir data/features_labeled \
        --config config/features.yaml --horizon 5 --threshold 0.0
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from utils.logger import setup_logger
from features.registry import load_enabled_features
from utils.labeling import label_future_return


def apply_features(
    df: pd.DataFrame,
    enabled_features: dict,
    logger
) -> pd.DataFrame:
    """
    Apply each enabled feature function to the DataFrame.

    Args:
        df: DataFrame with cleaned OHLCV data indexed by date.
        enabled_features: Mapping from feature name to feature-function.
        logger: Logger instance to record successes or failures.

    Returns:
        DataFrame extended with new feature columns.
    """
    df_feat = df.copy()
    for name, func in enabled_features.items():
        try:
            df_feat[name] = func(df_feat)
            logger.info(f"{name} computed")
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
    return df_feat


def main(
    input_dir: str,
    output_dir: str,
    config_path: str,
    label_horizon: int,
    label_threshold: float
) -> None:
    """
    Main pipeline function: loads cleaned CSVs, applies features, labels returns,
    and writes output CSVs.

    Args:
        input_dir: Directory containing cleaned CSVs of raw data.
        output_dir: Directory to write feature-labeled CSVs.
        config_path: Path to YAML config file with feature toggles.
        label_horizon: Number of days to look ahead for label generation.
        label_threshold: Minimum return threshold to assign a positive label.
    """
    # Start timing the pipeline
    start_time = time.perf_counter()

    # Prepare input and output paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger to record feature pipeline operations
    logger = setup_logger("feature_pipeline.log")

    # Load the dictionary of enabled features from config
    enabled = load_enabled_features(config_path)
    logger.info(f"Enabled features: {list(enabled.keys())}")

    # Iterate through each ticker CSV in the input directory
    for csv_file in sorted(input_path.glob("*.csv")):
        ticker = csv_file.stem
        logger.info(f"Processing {ticker}")
        # Read cleaned OHLCV data
        df = pd.read_csv(csv_file, parse_dates=True, index_col=0)

        # Compute feature columns
        df_feat = apply_features(df, enabled, logger)

        # Generate labels based on future returns
        df_labeled = label_future_return(
            df_feat,
            close_col='Close' if 'Close' in df_feat.columns else 'close',
            horizon=label_horizon,
            threshold=label_threshold,
            label_name=f"label_{label_horizon}d"
        )

        # Write the final feature+label CSV
        df_labeled.to_csv(output_path / csv_file.name)
        logger.info(f"Features + labels written to {csv_file.name}")

    # Log total elapsed time for the pipeline
    elapsed = time.perf_counter() - start_time
    logger.info(f"=== Feature pipeline completed in {elapsed:.2f} seconds ===")


if __name__ == "__main__":
    # Parse command-line arguments for pipeline parameters
    parser = argparse.ArgumentParser(description="Compute features and labels for tickers.")
    parser.add_argument("--input-dir", required=True, help="Directory of cleaned CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write feature CSVs")
    parser.add_argument("--config", required=True, help="Path to feature toggles YAML")
    parser.add_argument("--horizon", type=int, default=5, help="Label horizon in days")
    parser.add_argument("--threshold", type=float, default=0.0, help="Return threshold for labeling")
    args = parser.parse_args()

    # Execute the pipeline with provided arguments
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        label_horizon=args.horizon,
        label_threshold=args.threshold
    )

