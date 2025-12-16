#!/usr/bin/env python3
"""
compare_filters.py

Compare backtest performance with and without entry filters to identify
which filters are helping or hurting.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.enhanced_backtest import run_backtest, load_model, calculate_metrics, TICKERS_CSV, DATA_DIR, DEFAULT_MODEL
from utils.stop_loss_policy import create_stop_loss_config_from_args
from src.apply_entry_filters import get_default_filters


def main():
    parser = argparse.ArgumentParser(
        description="Compare backtest with and without filters"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Trade window (holding period in days)"
    )
    parser.add_argument(
        "--return-threshold",
        type=float,
        required=True,
        help="Return threshold (e.g., 0.15 for 15%)"
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=0.80,
        help="Model probability threshold"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop-loss threshold (e.g., -0.075 for -7.5%%). DEPRECATED: Use --stop-loss-mode and related args for adaptive stops."
    )
    parser.add_argument(
        "--stop-loss-mode",
        type=str,
        choices=["constant", "adaptive_atr", "swing_atr"],
        default=None,
        help="Stop-loss mode: 'constant' (fixed), 'adaptive_atr' (ATR-based), or 'swing_atr' (swing low + ATR buffer)"
    )
    parser.add_argument(
        "--atr-stop-k",
        type=float,
        default=1.8,
        help="ATR multiplier for adaptive stops (default: 1.8). Used for adaptive_atr and swing_atr fallback."
    )
    parser.add_argument(
        "--atr-stop-min-pct",
        type=float,
        default=0.04,
        help="Minimum stop distance for adaptive stops (default: 0.04 = 4%%). Used for adaptive_atr and swing_atr."
    )
    parser.add_argument(
        "--atr-stop-max-pct",
        type=float,
        default=0.10,
        help="Maximum stop distance for adaptive stops (default: 0.10 = 10%%). Used for adaptive_atr and swing_atr."
    )
    parser.add_argument(
        "--swing-lookback-days",
        type=int,
        default=10,
        help="Days to look back for swing low (default: 10). Only used when --stop-loss-mode=swing_atr."
    )
    parser.add_argument(
        "--swing-atr-buffer-k",
        type=float,
        default=0.75,
        help="ATR multiplier for swing_atr buffer (default: 0.75). Only used when --stop-loss-mode=swing_atr."
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=1000.0,
        help="Position size per trade"
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default="2024-01-01",
        help="Test data start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test-end-date",
        type=str,
        default=None,
        help="Test data end date (YYYY-MM-DD)"
    )
    
    args = parser.parse_args()
    
    # Load model
    model_path = Path(DEFAULT_MODEL)
    model, features, scaler, features_to_scale, _ = load_model(model_path)
    
    if model is None:
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load tickers
    tickers_df = pd.read_csv(TICKERS_CSV, header=None)
    tickers = tickers_df.iloc[:, 0].astype(str).tolist()
    
    # Create stop-loss configuration
    stop_loss_config = create_stop_loss_config_from_args(
        stop_loss_mode=getattr(args, 'stop_loss_mode', None),
        constant_stop_loss_pct=None,
        atr_stop_k=getattr(args, 'atr_stop_k', 1.8),
        atr_stop_min_pct=getattr(args, 'atr_stop_min_pct', 0.04),
        atr_stop_max_pct=getattr(args, 'atr_stop_max_pct', 0.10),
        swing_lookback_days=getattr(args, 'swing_lookback_days', 10),
        swing_atr_buffer_k=getattr(args, 'swing_atr_buffer_k', 0.75),
        return_threshold=args.return_threshold,
        legacy_stop_loss=getattr(args, 'stop_loss', None)
    )
    
    # Legacy stop_loss_value for backward compatibility
    stop_loss_value = args.stop_loss
    if stop_loss_value is None and args.return_threshold is not None:
        stop_loss_value = -abs(args.return_threshold) / 2
    
    print(f"\n{'='*80}")
    print("FILTER COMPARISON ANALYSIS")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Return Threshold: {args.return_threshold:.2%}")
    print(f"  Model Threshold: {args.model_threshold:.2f}")
    print(f"  Stop-Loss Mode: {stop_loss_config.mode}")
    if stop_loss_config.mode == "constant":
        print(f"  Stop-Loss: {stop_loss_config.constant_stop_loss_pct:.2%}")
    elif stop_loss_config.mode == "adaptive_atr":
        print(f"  ATR Stop K: {stop_loss_config.atr_stop_k}")
        print(f"  ATR Stop Range: {stop_loss_config.atr_stop_min_pct:.2%} - {stop_loss_config.atr_stop_max_pct:.2%}")
    elif stop_loss_config.mode == "swing_atr":
        print(f"  Swing Lookback Days: {stop_loss_config.swing_lookback_days}")
        print(f"  Swing ATR Buffer K: {stop_loss_config.swing_atr_buffer_k}")
        print(f"  Stop Range: {stop_loss_config.atr_stop_min_pct:.2%} - {stop_loss_config.atr_stop_max_pct:.2%}")
    print(f"  Test Period: {args.test_start_date} to {args.test_end_date or 'end'}")
    
    # Run baseline (no filters)
    print(f"\n{'='*80}")
    print("RUNNING BASELINE (NO FILTERS)")
    print(f"{'='*80}")
    
    trades_baseline = run_backtest(
        tickers=tickers,
        data_dir=DATA_DIR,
        horizon=args.horizon,
        return_threshold=args.return_threshold,
        position_size=args.position_size,
        model=model,
        features=features,
        model_threshold=args.model_threshold,
        strategy="model",
        stop_loss=stop_loss_value,  # Legacy parameter
        stop_loss_config=stop_loss_config,  # New parameter
        scaler=scaler,
        features_to_scale=features_to_scale,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        entry_filters=None
    )
    
    metrics_baseline = calculate_metrics(trades_baseline, args.position_size) if not trades_baseline.empty else {}
    
    # Run with all filters
    print(f"\n{'='*80}")
    print("RUNNING WITH ALL FILTERS")
    print(f"{'='*80}")
    
    all_filters = get_default_filters()
    print(f"\nApplying {len(all_filters)} filters:")
    for feat, (op, val) in sorted(all_filters.items()):
        print(f"  {feat} {op} {val}")
    
    trades_filtered = run_backtest(
        tickers=tickers,
        data_dir=DATA_DIR,
        horizon=args.horizon,
        return_threshold=args.return_threshold,
        position_size=args.position_size,
        model=model,
        features=features,
        model_threshold=args.model_threshold,
        strategy="model",
        stop_loss=stop_loss_value,  # Legacy parameter
        stop_loss_config=stop_loss_config,  # New parameter
        scaler=scaler,
        features_to_scale=features_to_scale,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        entry_filters=all_filters
    )
    
    metrics_filtered = calculate_metrics(trades_filtered, args.position_size) if not trades_filtered.empty else {}
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Baseline':>15} {'With Filters':>15} {'Change':>15}")
    print("-" * 75)
    
    metrics_to_compare = [
        ('n_trades', 'Total Trades', 'int'),
        ('win_rate', 'Win Rate', 'percent'),
        ('avg_return', 'Avg Return', 'percent'),
        ('annual_return', 'Annual Return', 'percent'),
        ('total_pnl', 'Total P&L', 'currency'),
        ('sharpe_ratio', 'Sharpe Ratio', 'float'),
        ('profit_factor', 'Profit Factor', 'float'),
    ]
    
    for key, label, fmt in metrics_to_compare:
        baseline_val = metrics_baseline.get(key, 0)
        filtered_val = metrics_filtered.get(key, 0)
        
        if fmt == 'percent':
            baseline_str = f"{baseline_val:.2%}"
            filtered_str = f"{filtered_val:.2%}"
            change = filtered_val - baseline_val
            change_str = f"{change:+.2%}"
        elif fmt == 'currency':
            baseline_str = f"${baseline_val:,.2f}"
            filtered_str = f"${filtered_val:,.2f}"
            change = filtered_val - baseline_val
            change_str = f"${change:+,.2f}"
        elif fmt == 'int':
            baseline_str = f"{int(baseline_val):,}"
            filtered_str = f"{int(filtered_val):,}"
            change = filtered_val - baseline_val
            change_str = f"{change:+,d}"
        else:  # float
            baseline_str = f"{baseline_val:.2f}"
            filtered_str = f"{filtered_val:.2f}"
            change = filtered_val - baseline_val
            change_str = f"{change:+.2f}"
        
        print(f"{label:<30} {baseline_str:>15} {filtered_str:>15} {change_str:>15}")
    
    # Show stop-loss rates
    if 'exit_reason' in trades_baseline.columns and 'exit_reason' in trades_filtered.columns:
        sl_baseline = (trades_baseline['exit_reason'] == 'stop_loss').sum() / len(trades_baseline) * 100 if len(trades_baseline) > 0 else 0
        sl_filtered = (trades_filtered['exit_reason'] == 'stop_loss').sum() / len(trades_filtered) * 100 if len(trades_filtered) > 0 else 0
        
        print(f"\n{'Stop-Loss Rate':<30} {sl_baseline:>14.1f}% {sl_filtered:>14.1f}% {sl_filtered - sl_baseline:>+14.1f}%")
    
    # Analyze which filters are most restrictive
    if not trades_filtered.empty:
        print(f"\n{'='*80}")
        print("FILTER ANALYSIS")
        print(f"{'='*80}")
        print(f"\nFilters reduced trades from {len(trades_baseline)} to {len(trades_filtered)}")
        print(f"  Reduction: {len(trades_baseline) - len(trades_filtered)} trades ({((len(trades_baseline) - len(trades_filtered)) / len(trades_baseline) * 100):.1f}%)")
        
        # Check which features are available in the data
        if not trades_baseline.empty:
            # Sample a ticker to check feature availability
            sample_ticker = trades_baseline.iloc[0]['ticker'] if 'ticker' in trades_baseline.columns else tickers[0]
            sample_file = DATA_DIR / f"{sample_ticker}.parquet"
            if sample_file.exists():
                sample_df = pd.read_parquet(sample_file)
                available_features = set(sample_df.columns)
                filter_features = set(all_filters.keys())
                
                missing_features = filter_features - available_features
                if missing_features:
                    print(f"\n⚠️  Missing features (filters not applied):")
                    for feat in sorted(missing_features):
                        print(f"    - {feat}")
                
                applied_features = filter_features & available_features
                print(f"\n✓ Applied {len(applied_features)}/{len(all_filters)} filters")


if __name__ == "__main__":
    main()

