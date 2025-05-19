#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

from utils.logger import setup_logger
from features.registry import load_enabled_features
from utils.labeling import label_future_return

def apply_features(df: pd.DataFrame, enabled_features: dict, logger) -> pd.DataFrame:
    """
    Apply each enabled feature function to the DataFrame.
    """
    df_feat = df.copy()
    for name, func in enabled_features.items():
        try:
            df_feat[name] = func(df_feat)
            logger.info(f"{name} computed")
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
    return df_feat

def main(input_dir: str, output_dir: str, config: str, 
         label_horizon: int, label_threshold: float) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("feature_pipeline.log")
    enabled = load_enabled_features(config)
    logger.info(f"Enabled features: {list(enabled.keys())}")

    for csv_file in sorted(input_path.glob("*.csv")):
        ticker = csv_file.stem
        logger.info(f"Processing {ticker}")
        df = pd.read_csv(csv_file, parse_dates=True, index_col=0)
        df_feat = apply_features(df, enabled, logger)
        # Labeling integration:
        df_labeled = label_future_return(
            df_feat,
            close_col='Close' if 'Close' in df_feat.columns else 'close',
            horizon=label_horizon,
            threshold=label_threshold,
            label_name=f"label_{label_horizon}d"
        )
        df_labeled.to_csv(output_path / csv_file.name)
        logger.info(f"Features + labels written to {csv_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute features and labels.")
    parser.add_argument("--input-dir",  required=True, help="Directory of cleaned CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write feature CSVs")
    parser.add_argument("--config",     required=True, help="Path to feature toggles YAML")
    parser.add_argument("--horizon",    type=int, default=5, help="Label horizon in days")
    parser.add_argument("--threshold",  type=float, default=0.0, help="Return threshold for labeling")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.config, args.horizon, args.threshold)
