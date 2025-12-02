#!/usr/bin/env python3
"""
apply_entry_filters.py

Applies entry filters to reduce stop-loss trades based on analysis findings.

This script runs a backtest with entry filters that prevent trades when certain
conditions are met (e.g., negative momentum, low market correlation, timing issues).

Default filters are based on stop-loss analysis of validation data (2023).
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.enhanced_backtest import run_backtest, load_model, calculate_metrics, TICKERS_CSV, DATA_DIR, DEFAULT_MODEL


def get_default_filters() -> Dict[str, Tuple[str, float]]:
    """
    Get default entry filters based on stop-loss analysis.
    
    These filters are derived from validation data (2023) analysis showing
    which features correlate with stop-loss trades.
    
    Returns:
        Dict of {feature_name: (operator, threshold)}
    """
    return {
        # Momentum filters (stop-loss trades have negative momentum)
        'rsi_slope': ('>', -1.349),
        'log_return_1d': ('>', -0.017),
        'log_return_5d': ('>', -0.057),
        
        # Market context filters (stop-loss trades underperform market)
        'relative_strength_spy_5d': ('>', -4.269),
        'market_correlation_20d': ('>', 0.532),
        
        # Price action filters (stop-loss trades near support/breakdowns)
        'candle_body_pct': ('>', -25.370),
        'close_position_in_range': ('>', 0.368),
        'price_near_support': ('<', 0.443),
        'rolling_min_5d_breakdown': ('<', 1.481),
        'close_vs_ma10': ('>', 0.954),
        
        # Overbought filters (stop-loss trades in overbought conditions)
        'weekly_rsi_14w': ('<', 44.455),
        'trend_consistency': ('<', 35.638),
        'high_vs_close': ('<', 3.000),
        
        # MACD filter (stop-loss trades have negative MACD cross)
        'macd_cross_signal': ('>', -0.074),
        
        # Time-based filters (stop-loss trades cluster in certain months)
        'month_of_year_cos': ('>', 0.172),
        'day_of_month_cos': ('>', 0.039),
    }


def apply_timing_filters(
    df: pd.DataFrame,
    avoid_monday_tuesday: bool = True,
    avoid_october: bool = True
) -> pd.Series:
    """
    Apply timing-based filters to avoid high-risk entry periods.
    
    Args:
        df: DataFrame with DatetimeIndex
        avoid_monday_tuesday: If True, filter out Monday/Tuesday entries
        avoid_october: If True, filter out October entries
    
    Returns:
        Boolean Series indicating which rows pass timing filters
    """
    mask = pd.Series(True, index=df.index)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert if possible
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        else:
            return mask  # Can't apply timing filters without dates
    else:
        dates = df.index
    
    if avoid_monday_tuesday:
        # Monday = 0, Tuesday = 1
        mask = mask & ~dates.dayofweek.isin([0, 1])
    
    if avoid_october:
        # October = 10
        mask = mask & (dates.month != 10)
    
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest with entry filters to reduce stop-losses"
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
        help="Return threshold (e.g., 0.15 for 15%%)"
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
        help="Stop-loss threshold (e.g., -0.075 for -7.5%%)"
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
        help="Test data start date (YYYY-MM-DD). Default: 2024-01-01"
    )
    parser.add_argument(
        "--test-end-date",
        type=str,
        default=None,
        help="Test data end date (YYYY-MM-DD). Default: None (all available)"
    )
    parser.add_argument(
        "--use-default-filters",
        action="store_true",
        default=True,
        help="Use default filters from stop-loss analysis (default: True)"
    )
    parser.add_argument(
        "--no-default-filters",
        action="store_true",
        default=False,
        help="Disable default filters from stop-loss analysis (use only custom filters)"
    )
    parser.add_argument(
        "--no-timing-filters",
        action="store_true",
        default=False,
        help="Disable timing filters (Monday/Tuesday, October)"
    )
    parser.add_argument(
        "--custom-filter",
        action="append",
        nargs=3,
        metavar=("FEATURE", "OPERATOR", "THRESHOLD"),
        help="Add custom filter: --custom-filter feature_name > 0.5 (can be used multiple times)"
    )
    
    args = parser.parse_args()
    
    # Build entry filters
    entry_filters = {}
    
    # Use default filters unless explicitly disabled
    use_defaults = args.use_default_filters and not args.no_default_filters
    
    if use_defaults:
        entry_filters.update(get_default_filters())
        print("\nUsing default filters from stop-loss analysis:")
        for feat, (op, val) in sorted(entry_filters.items()):
            print(f"  {feat} {op} {val}")
    
    # Add custom filters
    if args.custom_filter:
        for feature, operator, threshold_str in args.custom_filter:
            try:
                threshold = float(threshold_str)
                if operator not in ['>', '<', '>=', '<=']:
                    print(f"Warning: Invalid operator '{operator}'. Must be one of: >, <, >=, <=")
                    continue
                entry_filters[feature] = (operator, threshold)
                print(f"  Added custom filter: {feature} {operator} {threshold}")
            except ValueError:
                print(f"Warning: Invalid threshold '{threshold_str}' for {feature}. Skipping.")
    
    # Load model
    model_path = Path(DEFAULT_MODEL)
    model, features, scaler, features_to_scale, _ = load_model(model_path)
    
    if model is None:
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load tickers
    tickers_df = pd.read_csv(TICKERS_CSV, header=None)
    tickers = tickers_df.iloc[:, 0].astype(str).tolist()
    
    # Calculate stop-loss
    stop_loss_value = args.stop_loss
    if stop_loss_value is None and args.return_threshold is not None:
        stop_loss_value = -abs(args.return_threshold) / 2
    
    print(f"\n{'='*80}")
    print("BACKTEST WITH ENTRY FILTERS")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Return Threshold: {args.return_threshold:.2%}")
    print(f"  Model Threshold: {args.model_threshold:.2f}")
    print(f"  Stop-Loss: {stop_loss_value:.2%}" if stop_loss_value else "  Stop-Loss: None")
    print(f"  Position Size: ${args.position_size:,.2f}")
    print(f"  Test Period: {args.test_start_date} to {args.test_end_date or 'end'}")
    print(f"  Timing Filters: {'Disabled' if args.no_timing_filters else 'Enabled (avoid Mon/Tue, Oct)'}")
    
    # We need to modify run_backtest to support timing filters
    # For now, we'll apply timing filters in a wrapper
    # But first, let's run the backtest with feature filters
    
    # Run backtest with filters
    trades = run_backtest(
        tickers=tickers,
        data_dir=DATA_DIR,
        horizon=args.horizon,
        return_threshold=args.return_threshold,
        position_size=args.position_size,
        model=model,
        features=features,
        model_threshold=args.model_threshold,
        strategy="model",
        stop_loss=stop_loss_value,
        scaler=scaler,
        features_to_scale=features_to_scale,
        test_start_date=args.test_start_date,
        test_end_date=args.test_end_date,
        entry_filters=entry_filters if entry_filters else None
    )
    
    if trades.empty:
        print("\nNo trades generated with filters applied.")
        return
    
    # Apply timing filters to trades if enabled
    if not args.no_timing_filters and len(trades) > 0:
        # Filter trades by entry date
        trades_before = len(trades)
        
        if isinstance(trades.index, pd.DatetimeIndex):
            dates = trades.index
        elif 'entry_date' in trades.columns:
            dates = pd.to_datetime(trades['entry_date'])
        else:
            dates = None
        
        if dates is not None:
            # Monday = 0, Tuesday = 1
            # October = 10
            timing_mask = ~dates.dayofweek.isin([0, 1]) & (dates.month != 10)
            
            if isinstance(trades.index, pd.DatetimeIndex):
                trades = trades[timing_mask]
            else:
                trades = trades[timing_mask.values]
            
            trades_after = len(trades)
            if trades_before > trades_after:
                print(f"\nTiming filters removed {trades_before - trades_after} trades")
    
    if trades.empty:
        print("\nNo trades remaining after all filters applied.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(trades, args.position_size)
    
    # Display results
    print(f"\n{'='*80}")
    print("BACKTEST RESULTS (WITH FILTERS)")
    print(f"{'='*80}")
    print(f"\nBacktest Period: {metrics.get('date_range', 'N/A')}")
    print(f"\nPerformance Metrics:")
    print(f"  Total Trades: {metrics['n_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Average Return: {metrics['avg_return']:.2%}")
    print(f"  Annual Return: {metrics.get('annual_return', 0.0):.2%}")
    print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Average P&L per Trade: ${metrics['avg_pnl']:,.2f}")
    print(f"  Maximum Drawdown: ${metrics['max_drawdown']:,.2f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Average Holding Days: {metrics['avg_holding_days']:.1f}")
    print(f"  Max Concurrent Positions: {metrics.get('max_concurrent_positions', 0)}")
    print(f"  Max Capital Invested: ${metrics.get('max_capital_invested', 0.0):,.2f}")
    
    # Show exit reasons
    if 'exit_reason' in trades.columns:
        exit_reasons = trades['exit_reason'].value_counts()
        print(f"\nExit Reasons:")
        for reason, count in exit_reasons.items():
            pct = (count / len(trades)) * 100
            print(f"  {reason}: {count:,} ({pct:.1f}%)")
        
        stop_loss_count = exit_reasons.get('stop_loss', 0)
        stop_loss_pct = (stop_loss_count / len(trades)) * 100 if len(trades) > 0 else 0
        print(f"\nStop-Loss Rate: {stop_loss_pct:.1f}%")
        print(f"  (Baseline without filters: ~44.9%)")


if __name__ == "__main__":
    main()
