#!/usr/bin/env python3
"""
swing_trade_app.py

Main application for the Swing Trading ML System.

This is a unified interface that orchestrates:
  1. Data downloading from yfinance
  2. Feature engineering
  3. Model training with user-defined parameters
  4. Backtesting with configurable trade windows and return thresholds
  5. Current trade identification

Usage:
    python src/swing_trade_app.py [command] [options]

Commands:
    download      - Download stock data from yfinance
    clean         - Clean raw CSV files into Parquet format
    features      - Build feature set from cleaned data
    train         - Train ML model with user-defined parameters
    backtest      - Run backtests with configurable parameters
    identify      - Identify current potential trades
    full-pipeline - Run complete pipeline: download -> clean -> features -> train -> backtest -> identify
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
SCRIPTS_DIR = PROJECT_ROOT / "src"


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: {description} failed: {e}")
        return False


def download_data(
    tickers_file: str = "data/tickers/sp500_tickers.csv",
    start_date: str = "2008-01-01",
    raw_folder: str = "data/raw",
    sectors_file: str = "data/tickers/sectors.csv",
    full: bool = False,
    resume: bool = False,
    max_retries: int = 3
) -> bool:
    """Download stock data from yfinance with enhanced features."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "download_data.py"),
        "--tickers-file", tickers_file,
        "--start-date", start_date,
        "--raw-folder", raw_folder,
        "--sectors-file", sectors_file,
        "--chunk-size", "100",
        "--pause", "1.0",
        "--max-retries", str(max_retries)
    ]
    
    if full:
        cmd.append("--full")
    
    if resume:
        cmd.append("--resume")
    
    return run_command(cmd, "Downloading Stock Data")


def clean_data(
    raw_dir: str = "data/raw",
    clean_dir: str = "data/clean",
    resume: bool = False,
    workers: int = 4,
    verbose: bool = False
) -> bool:
    """Clean raw CSV files into Parquet format."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "clean_data.py"),
        "--raw-dir", raw_dir,
        "--clean-dir", clean_dir,
        "--workers", str(workers)
    ]
    
    if resume:
        cmd.append("--resume")
    
    if verbose:
        cmd.append("--verbose")
    
    return run_command(cmd, "Cleaning Raw Data")


def build_features(
    input_dir: str = "data/clean",
    output_dir: str = "data/features_labeled",
    config: str = "config/features.yaml",
    horizon: int = 5,
    threshold: float = 0.0,
    full: bool = False
) -> bool:
    """Build feature set from cleaned data."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "feature_pipeline.py"),
        "--input-dir", input_dir,
        "--output-dir", output_dir,
        "--config", config,
        "--horizon", str(horizon),
        "--threshold", str(threshold)
    ]
    
    if full:
        cmd.append("--full")
    
    return run_command(cmd, f"Building Features (horizon={horizon}d, threshold={threshold:.2%})")


def train_model(
    diagnostics: bool = False
) -> bool:
    """Train ML model."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_model.py")
    ]
    
    if diagnostics:
        cmd.append("--diagnostics")
    
    return run_command(cmd, "Training ML Model")


def run_backtest(
    horizon: int,
    return_threshold: float,
    position_size: float = 1000.0,
    strategy: str = "model",
    model_path: str = "models/xgb_classifier_selected_features.pkl",
    model_threshold: float = 0.5,
    output: Optional[str] = None
) -> bool:
    """Run backtest with configurable parameters."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "enhanced_backtest.py"),
        "--horizon", str(horizon),
        "--return-threshold", str(return_threshold),
        "--position-size", str(position_size),
        "--strategy", strategy,
        "--model", model_path,
        "--model-threshold", str(model_threshold)
    ]
    
    if output:
        cmd.extend(["--output", output])
    
    return run_command(
        cmd,
        f"Running Backtest (horizon={horizon}d, threshold={return_threshold:.2%}, strategy={strategy})"
    )


def identify_trades(
    model_path: str = "models/xgb_classifier_selected_features.pkl",
    min_probability: float = 0.5,
    top_n: int = 20,
    output: Optional[str] = None
) -> bool:
    """Identify current potential trades."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "identify_trades.py"),
        "--model", model_path,
        "--min-probability", str(min_probability),
        "--top-n", str(top_n)
    ]
    
    if output:
        cmd.extend(["--output", output])
    
    return run_command(cmd, "Identifying Current Trading Opportunities")


def main():
    """Main entry point for the swing trading app."""
    parser = argparse.ArgumentParser(
        description="Swing Trading ML Application - Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data
  python src/swing_trade_app.py download --full

  # Clean raw data
  python src/swing_trade_app.py clean --resume --workers 8

  # Build features with 10-day horizon and 5% return threshold
  python src/swing_trade_app.py features --horizon 10 --threshold 0.05

  # Train model
  python src/swing_trade_app.py train

  # Run backtest with 7-day window and 3% return threshold
  python src/swing_trade_app.py backtest --horizon 7 --return-threshold 0.03

  # Identify current trades
  python src/swing_trade_app.py identify --min-probability 0.6 --top-n 10

  # Run full pipeline
  python src/swing_trade_app.py full-pipeline --horizon 5 --return-threshold 0.05
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download stock data")
    dl_parser.add_argument("--tickers-file", default="data/tickers/sp500_tickers.csv")
    dl_parser.add_argument("--start-date", default="2008-01-01")
    dl_parser.add_argument("--raw-folder", default="data/raw")
    dl_parser.add_argument("--sectors-file", default="data/tickers/sectors.csv")
    dl_parser.add_argument("--full", action="store_true", help="Force full redownload")
    dl_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    dl_parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean raw CSV files")
    clean_parser.add_argument("--raw-dir", default="data/raw")
    clean_parser.add_argument("--clean-dir", default="data/clean")
    clean_parser.add_argument("--resume", action="store_true", help="Skip already-cleaned files")
    clean_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    clean_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Features command
    feat_parser = subparsers.add_parser("features", help="Build feature set")
    feat_parser.add_argument("--input-dir", default="data/clean")
    feat_parser.add_argument("--output-dir", default="data/features_labeled")
    feat_parser.add_argument("--config", default="config/features.yaml")
    feat_parser.add_argument("--horizon", type=int, default=5, help="Trade window in days")
    feat_parser.add_argument("--threshold", type=float, default=0.0, help="Return threshold (e.g., 0.05 for 5%%)")
    feat_parser.add_argument("--full", action="store_true", help="Force full recompute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument("--diagnostics", action="store_true", help="Include feature importance diagnostics")
    
    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--horizon", type=int, required=True, help="Trade window in days")
    bt_parser.add_argument("--return-threshold", type=float, required=True, help="Return threshold (e.g., 0.05 for 5%%)")
    bt_parser.add_argument("--position-size", type=float, default=1000.0, help="Position size per trade")
    bt_parser.add_argument("--strategy", choices=["model", "oracle", "rsi"], default="model")
    bt_parser.add_argument("--model", default="models/xgb_classifier_selected_features.pkl")
    bt_parser.add_argument("--model-threshold", type=float, default=0.5, help="Model probability threshold")
    bt_parser.add_argument("--output", help="Output CSV file for trades")
    
    # Identify command
    id_parser = subparsers.add_parser("identify", help="Identify current trades")
    id_parser.add_argument("--model", default="models/xgb_classifier_selected_features.pkl")
    id_parser.add_argument("--min-probability", type=float, default=0.5, help="Minimum prediction probability")
    id_parser.add_argument("--top-n", type=int, default=20, help="Maximum number of opportunities")
    id_parser.add_argument("--output", help="Output CSV file")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("full-pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("--horizon", type=int, required=True, help="Trade window in days")
    pipeline_parser.add_argument("--return-threshold", type=float, required=True, help="Return threshold (e.g., 0.05 for 5%%)")
    pipeline_parser.add_argument("--position-size", type=float, default=1000.0)
    pipeline_parser.add_argument("--model-threshold", type=float, default=0.5)
    pipeline_parser.add_argument("--min-probability", type=float, default=0.5)
    pipeline_parser.add_argument("--top-n", type=int, default=20)
    pipeline_parser.add_argument("--full-download", action="store_true", help="Force full data download")
    pipeline_parser.add_argument("--full-features", action="store_true", help="Force full feature recompute")
    pipeline_parser.add_argument("--skip-download", action="store_true", help="Skip data download step")
    pipeline_parser.add_argument("--skip-train", action="store_true", help="Skip model training step")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    success = True
    
    if args.command == "download":
        success = download_data(
            tickers_file=args.tickers_file,
            start_date=args.start_date,
            raw_folder=args.raw_folder,
            sectors_file=args.sectors_file,
            full=args.full,
            resume=args.resume,
            max_retries=args.max_retries
        )
    
    elif args.command == "clean":
        success = clean_data(
            raw_dir=args.raw_dir,
            clean_dir=args.clean_dir,
            resume=args.resume,
            workers=args.workers,
            verbose=args.verbose
        )
    
    elif args.command == "features":
        success = build_features(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=args.config,
            horizon=args.horizon,
            threshold=args.threshold,
            full=args.full
        )
    
    elif args.command == "train":
        success = train_model(diagnostics=args.diagnostics)
    
    elif args.command == "backtest":
        success = run_backtest(
            horizon=args.horizon,
            return_threshold=args.return_threshold,
            position_size=args.position_size,
            strategy=args.strategy,
            model_path=args.model,
            model_threshold=args.model_threshold,
            output=args.output
        )
    
    elif args.command == "identify":
        success = identify_trades(
            model_path=args.model,
            min_probability=args.min_probability,
            top_n=args.top_n,
            output=args.output
        )
    
    elif args.command == "full-pipeline":
        print("\n" + "="*80)
        print("RUNNING FULL PIPELINE")
        print("="*80)
        
        # Step 1: Download data
        if not args.skip_download:
            success = download_data(full=args.full_download, resume=False)
            if not success:
                print("\nPipeline stopped: Data download failed")
                return
        
        # Step 2: Clean data
        success = clean_data(resume=True)
        if not success:
            print("\nPipeline stopped: Data cleaning failed")
            return
        
        # Step 3: Build features
        success = build_features(
            horizon=args.horizon,
            threshold=args.return_threshold,
            full=args.full_features
        )
        if not success:
            print("\nPipeline stopped: Feature building failed")
            return
        
        # Step 4: Train model
        if not args.skip_train:
            success = train_model()
            if not success:
                print("\nPipeline stopped: Model training failed")
                return
        
        # Step 5: Run backtest
        success = run_backtest(
            horizon=args.horizon,
            return_threshold=args.return_threshold,
            position_size=args.position_size,
            model_threshold=args.model_threshold
        )
        if not success:
            print("\nPipeline stopped: Backtest failed")
            return
        
        # Step 6: Identify current trades
        success = identify_trades(
            min_probability=args.min_probability,
            top_n=args.top_n
        )
        
        if success:
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

