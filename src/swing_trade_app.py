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
    full: bool = False,
    feature_set: Optional[str] = None
) -> bool:
    """Build feature set from cleaned data."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "feature_pipeline.py"),
        "--horizon", str(horizon),
        "--threshold", str(threshold)
    ]
    
    if feature_set:
        # Use feature set (automatically sets config and output_dir)
        cmd.extend(["--feature-set", feature_set])
        if input_dir != "data/clean":
            cmd.extend(["--input-dir", input_dir])
    else:
        # Use explicit paths (backward compatibility)
        cmd.extend([
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--config", config
        ])
    
    if full:
        cmd.append("--full")
    
    feature_set_desc = f" (feature set: {feature_set})" if feature_set else ""
    return run_command(cmd, f"Building Features (horizon={horizon}d, threshold={threshold:.2%}){feature_set_desc}")


def train_model(
    diagnostics: bool = False,
    tune: bool = False,
    n_iter: int = 20,
    cv: bool = False,
    no_early_stop: bool = False,
    plots: bool = False,
    fast: bool = False,
    cv_folds: Optional[int] = None,
    imbalance_multiplier: float = 1.0,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    val_end: Optional[str] = None,
    horizon: Optional[int] = None,
    label_col: Optional[str] = None,
    feature_set: Optional[str] = None,
    model_output: Optional[str] = None
) -> bool:
    """Train ML model with optional hyperparameter tuning and visualization."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_model.py")
    ]
    
    if tune:
        cmd.append("--tune")
        cmd.extend(["--n-iter", str(n_iter)])
    
    if cv:
        cmd.append("--cv")
    
    if no_early_stop:
        cmd.append("--no-early-stop")
    
    if plots:
        cmd.append("--plots")
    
    if fast:
        cmd.append("--fast")
    
    if cv_folds is not None:
        cmd.extend(["--cv-folds", str(cv_folds)])
    
    if diagnostics:
        cmd.append("--diagnostics")
    
    if imbalance_multiplier != 1.0:
        cmd.extend(["--imbalance-multiplier", str(imbalance_multiplier)])
    
    if train_end is not None:
        cmd.extend(["--train-end", train_end])
    
    if val_end is not None:
        cmd.extend(["--val-end", val_end])
    
    if horizon is not None:
        cmd.extend(["--horizon", str(horizon)])
    
    if label_col is not None:
        cmd.extend(["--label-col", label_col])
    
    if feature_set is not None:
        cmd.extend(["--feature-set", feature_set])
    
    if model_output is not None:
        cmd.extend(["--model-output", model_output])
    
    feature_set_desc = f" (feature set: {feature_set})" if feature_set else ""
    return run_command(cmd, f"Training ML Model{feature_set_desc}")


def run_backtest(
    horizon: int,
    return_threshold: float,
    position_size: float = 1000.0,
    strategy: str = "model",
    model_path: str = "models/xgb_classifier_selected_features.pkl",
    model_threshold: float = 0.5,
    stop_loss: Optional[float] = None,
    stop_loss_mode: Optional[str] = None,
    atr_stop_k: float = 1.8,
    atr_stop_min_pct: float = 0.04,
    atr_stop_max_pct: float = 0.10,
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
    
    if stop_loss is not None:
        cmd.extend(["--stop-loss", str(stop_loss)])
    if stop_loss_mode is not None:
        cmd.extend(["--stop-loss-mode", stop_loss_mode])
        cmd.extend(["--atr-stop-k", str(atr_stop_k)])
        cmd.extend(["--atr-stop-min-pct", str(atr_stop_min_pct)])
        cmd.extend(["--atr-stop-max-pct", str(atr_stop_max_pct)])
        if stop_loss_mode == "swing_atr":
            cmd.extend(["--swing-lookback-days", str(swing_lookback_days)])
            cmd.extend(["--swing-atr-buffer-k", str(swing_atr_buffer_k)])
    
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
    output: Optional[str] = None,
    stop_loss_mode: Optional[str] = None,
    atr_stop_k: float = 1.8,
    atr_stop_min_pct: float = 0.04,
    atr_stop_max_pct: float = 0.10
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
    
    if stop_loss_mode is not None:
        cmd.extend(["--stop-loss-mode", stop_loss_mode])
        cmd.extend(["--atr-stop-k", str(atr_stop_k)])
        cmd.extend(["--atr-stop-min-pct", str(atr_stop_min_pct)])
        cmd.extend(["--atr-stop-max-pct", str(atr_stop_max_pct)])
    
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
    feat_parser.add_argument("--feature-set", type=str, default=None, help="Feature set name (e.g., 'v1', 'v2'). If specified, automatically sets config and output-dir. Default: uses explicit paths.")
    feat_parser.add_argument("--input-dir", default="data/clean")
    feat_parser.add_argument("--output-dir", default="data/features_labeled")
    feat_parser.add_argument("--config", default="config/features.yaml")
    feat_parser.add_argument("--horizon", type=int, default=5, help="Trade window in days")
    feat_parser.add_argument("--threshold", type=float, default=0.0, help="Return threshold (e.g., 0.05 for 5%%)")
    feat_parser.add_argument("--full", action="store_true", help="Force full recompute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning (RandomizedSearchCV)")
    train_parser.add_argument("--n-iter", type=int, default=20, help="Number of hyperparameter search iterations (default: 20)")
    train_parser.add_argument("--cv", action="store_true", help="Use cross-validation for hyperparameter tuning")
    train_parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    train_parser.add_argument("--plots", action="store_true", help="Generate training curves and feature importance plots")
    train_parser.add_argument("--fast", action="store_true", help="Use faster hyperparameter tuning (~3-5x faster, slightly less optimal)")
    train_parser.add_argument("--cv-folds", type=int, default=None, help="Number of CV folds (default: 3, or 2 if --fast)")
    train_parser.add_argument("--diagnostics", action="store_true", help="Include SHAP diagnostics")
    train_parser.add_argument("--imbalance-multiplier", type=float, default=1.0, help="Class imbalance multiplier (default: 1.0, try 2.0-3.0 for more trades)")
    train_parser.add_argument("--train-start", type=str, default=None, help="Training data start date (YYYY-MM-DD). Default: None (use all available data from the beginning)")
    train_parser.add_argument("--train-end", type=str, default=None, help="Training data end date (YYYY-MM-DD, default: 2022-12-31)")
    train_parser.add_argument("--val-end", type=str, default=None, help="Validation data end date (YYYY-MM-DD, default: 2023-12-31)")
    train_parser.add_argument("--horizon", type=int, default=None, help="Trade horizon in days (e.g., 5, 30). Used to auto-detect label column.")
    train_parser.add_argument("--label-col", type=str, default=None, help="Label column name (e.g., 'label_5d', 'label_30d'). Auto-detected if not specified.")
    train_parser.add_argument("--feature-set", type=str, default=None, help="Feature set name (e.g., 'v1', 'v2'). If specified, automatically sets data directory and train config.")
    train_parser.add_argument("--model-output", type=str, default=None, help="Custom model output file path (e.g., 'models/my_custom_model.pkl'). If not specified, uses default naming.")
    
    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--horizon", type=int, required=True, help="Trade window in days")
    bt_parser.add_argument("--return-threshold", type=float, required=True, help="Return threshold (e.g., 0.05 for 5%%)")
    bt_parser.add_argument("--position-size", type=float, default=1000.0, help="Position size per trade")
    bt_parser.add_argument("--strategy", choices=["model", "oracle", "rsi"], default="model")
    bt_parser.add_argument("--model", default="models/xgb_classifier_selected_features.pkl")
    bt_parser.add_argument("--model-threshold", type=float, default=0.5, help="Model probability threshold")
    bt_parser.add_argument("--stop-loss", type=float, default=None, help="Stop-loss threshold as decimal (e.g., -0.05 for -5%%). If not specified and return-threshold is provided, uses 2:1 risk-reward. DEPRECATED: Use --stop-loss-mode and related args for adaptive stops.")
    bt_parser.add_argument("--stop-loss-mode", type=str, choices=["constant", "adaptive_atr", "swing_atr"], default=None, help="Stop-loss mode: 'constant' (fixed), 'adaptive_atr' (ATR-based), or 'swing_atr' (swing low + ATR buffer)")
    bt_parser.add_argument("--atr-stop-k", type=float, default=1.8, help="ATR multiplier for adaptive stops (default: 1.8). Used for adaptive_atr and swing_atr fallback.")
    bt_parser.add_argument("--atr-stop-min-pct", type=float, default=0.04, help="Minimum stop distance for adaptive stops (default: 0.04 = 4%%)")
    bt_parser.add_argument("--atr-stop-max-pct", type=float, default=0.10, help="Maximum stop distance for adaptive stops (default: 0.10 = 10%%)")
    bt_parser.add_argument("--swing-lookback-days", type=int, default=10, help="Days to look back for swing low (default: 10). Only used when --stop-loss-mode=swing_atr.")
    bt_parser.add_argument("--swing-atr-buffer-k", type=float, default=0.75, help="ATR multiplier for swing_atr buffer (default: 0.75). Only used when --stop-loss-mode=swing_atr.")
    bt_parser.add_argument("--output", help="Output CSV file for trades")
    
    # Identify command
    id_parser = subparsers.add_parser("identify", help="Identify current trades")
    id_parser.add_argument("--model", default="models/xgb_classifier_selected_features.pkl")
    id_parser.add_argument("--min-probability", type=float, default=0.5, help="Minimum prediction probability")
    id_parser.add_argument("--top-n", type=int, default=20, help="Maximum number of opportunities")
    id_parser.add_argument("--output", help="Output CSV file")
    id_parser.add_argument("--stop-loss-mode", type=str, choices=["constant", "adaptive_atr", "swing_atr"], default=None, help="Stop-loss mode: 'constant' (fixed), 'adaptive_atr' (ATR-based), or 'swing_atr' (swing low + ATR buffer). If specified, calculates and displays recommended stop-loss for each trade.")
    id_parser.add_argument("--atr-stop-k", type=float, default=1.8, help="ATR multiplier for adaptive stops (default: 1.8). Used for adaptive_atr and swing_atr fallback.")
    id_parser.add_argument("--atr-stop-min-pct", type=float, default=0.04, help="Minimum stop distance for adaptive stops (default: 0.04 = 4%%)")
    id_parser.add_argument("--atr-stop-max-pct", type=float, default=0.10, help="Maximum stop distance for adaptive stops (default: 0.10 = 10%%)")
    id_parser.add_argument("--swing-lookback-days", type=int, default=10, help="Days to look back for swing low (default: 10). Only used when --stop-loss-mode=swing_atr.")
    id_parser.add_argument("--swing-atr-buffer-k", type=float, default=0.75, help="ATR multiplier for swing_atr buffer (default: 0.75). Only used when --stop-loss-mode=swing_atr.")
    
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
            full=args.full,
            feature_set=getattr(args, 'feature_set', None)
        )
    
    elif args.command == "train":
        success = train_model(
            diagnostics=args.diagnostics,
            tune=args.tune,
            n_iter=args.n_iter,
            cv=args.cv,
            no_early_stop=args.no_early_stop,
            plots=args.plots,
            fast=args.fast,
            cv_folds=args.cv_folds,
            imbalance_multiplier=getattr(args, 'imbalance_multiplier', 1.0),
            train_start=getattr(args, 'train_start', None),
            train_end=getattr(args, 'train_end', None),
            val_end=getattr(args, 'val_end', None),
            horizon=getattr(args, 'horizon', None),
            label_col=getattr(args, 'label_col', None),
            feature_set=getattr(args, 'feature_set', None),
            model_output=getattr(args, 'model_output', None)
        )
    
    elif args.command == "backtest":
        # Build command with new stop-loss parameters
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "enhanced_backtest.py"),
            "--horizon", str(args.horizon),
            "--return-threshold", str(args.return_threshold),
            "--position-size", str(args.position_size),
            "--strategy", args.strategy,
            "--model", args.model,
            "--model-threshold", str(args.model_threshold)
        ]
        
        # Add stop-loss parameters
        if getattr(args, 'stop_loss', None) is not None:
            cmd.extend(["--stop-loss", str(args.stop_loss)])
        if getattr(args, 'stop_loss_mode', None) is not None:
            cmd.extend(["--stop-loss-mode", args.stop_loss_mode])
            cmd.extend(["--atr-stop-k", str(getattr(args, 'atr_stop_k', 1.8))])
            cmd.extend(["--atr-stop-min-pct", str(getattr(args, 'atr_stop_min_pct', 0.04))])
            cmd.extend(["--atr-stop-max-pct", str(getattr(args, 'atr_stop_max_pct', 0.10))])
            if args.stop_loss_mode == "swing_atr":
                cmd.extend(["--swing-lookback-days", str(getattr(args, 'swing_lookback_days', 10))])
                cmd.extend(["--swing-atr-buffer-k", str(getattr(args, 'swing_atr_buffer_k', 0.75))])
        
        if args.output:
            cmd.extend(["--output", args.output])
        
        success = run_command(
            cmd,
            f"Running Backtest (horizon={args.horizon}d, threshold={args.return_threshold:.2%}, strategy={args.strategy})"
        )
    
    elif args.command == "identify":
        success = identify_trades(
            model_path=args.model,
            min_probability=args.min_probability,
            top_n=args.top_n,
            output=args.output,
            stop_loss_mode=getattr(args, 'stop_loss_mode', None),
            atr_stop_k=getattr(args, 'atr_stop_k', 1.8),
            atr_stop_min_pct=getattr(args, 'atr_stop_min_pct', 0.04),
            atr_stop_max_pct=getattr(args, 'atr_stop_max_pct', 0.10)
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

