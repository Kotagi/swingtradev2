#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

from utils.logger import setup_logger
from features.registry import load_enabled_features

def apply_features(df: pd.DataFrame, features: dict, logger) -> pd.DataFrame:
    """
    Apply each feature function to the DataFrame.
    Features: dict of feature_name -> function.
    Returns the DataFrame with new feature columns.
    """
    for name, func in features.items():
        try:
            df[name] = func(df)
            logger.info(f"✅ {name} computed")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}", exc_info=True)
    return df

def main(input_dir: str, output_dir: str, config_path: str) -> None:
    # Setup logger
    logger = setup_logger(__name__, log_file="feature_pipeline.log")

    # Paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load enabled feature functions
    enabled = load_enabled_features(config_path)
    if not enabled:
        logger.warning("No features enabled in config; exiting.")
        return

    # Process each CSV
    for csv_file in sorted(input_path.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file, parse_dates=True, index_col=0)
            logger.info(f"Loaded {csv_file.name} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load {csv_file.name}: {e}", exc_info=True)
            continue

        # Apply features
        df_feat = apply_features(df, enabled, logger)

        # Save output
        out_file = output_path / csv_file.name
        try:
            df_feat.to_csv(out_file)
            logger.info(f"Features written to {out_file.name}")
        except Exception as e:
            logger.error(f"Failed to write {out_file.name}: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply feature functions to cleaned CSVs.")
    parser.add_argument("--input-dir",  required=True, help="Directory of cleaned CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write feature CSVs")
    parser.add_argument("--config",     required=True, help="Path to features config YAML")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.config)
