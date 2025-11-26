#!/usr/bin/env python3
"""
analyze_stop_losses.py

Analyzes stop-loss trades to identify patterns and reduce stop-loss rate.

This script:
1. Loads backtest trades (or runs a backtest to generate them)
2. Identifies stop-loss trades vs winners
3. Compares features/conditions between stop-loss and winning trades
4. Identifies patterns that predict stop-losses
5. Suggests filters or improvements to reduce stop-loss rate
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
from collections import Counter

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.enhanced_backtest import run_backtest, load_model, TICKERS_CSV, DATA_DIR, DEFAULT_MODEL


def load_trades_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load trades from a CSV file."""
    df = pd.read_csv(csv_path)
    if 'entry_date' in df.columns:
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df = df.set_index('entry_date')
    return df


def get_entry_features(trades: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Get feature values at entry time for each trade.
    
    Returns DataFrame with trade info + feature values at entry.
    """
    all_features = []
    
    for idx, trade in trades.iterrows():
        # Get ticker - trades DataFrame has 'ticker' column
        ticker = trade.get('ticker', 'UNKNOWN') if 'ticker' in trade.index else 'UNKNOWN'
        
        # Get entry_date - trades DataFrame has entry_date as index
        if isinstance(idx, pd.Timestamp):
            entry_date = idx
        else:
            entry_date = idx  # Use index as entry_date
        
        if pd.isna(entry_date):
            continue
        
        # Load feature data for this ticker
        ticker_file = data_dir / f"{ticker}.parquet"
        if not ticker_file.exists():
            continue
        
        try:
            df = pd.read_parquet(ticker_file).sort_index()
            
            # Find the row closest to entry date
            if isinstance(df.index, pd.DatetimeIndex):
                # Find exact date or closest before
                entry_date_dt = pd.to_datetime(entry_date)
                if entry_date_dt in df.index:
                    entry_row = df.loc[entry_date_dt]
                else:
                    # Get closest date before entry
                    before_dates = df.index[df.index <= entry_date_dt]
                    if len(before_dates) > 0:
                        entry_row = df.loc[before_dates[-1]]
                    else:
                        continue
            else:
                continue
            
            # Extract features (exclude OHLCV, volume, labels)
            exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume'}
            exclude_cols.update([col for col in df.columns if col.startswith('label_')])
            
            features = {col: entry_row[col] for col in df.columns if col not in exclude_cols}
            features['ticker'] = ticker
            features['entry_date'] = entry_date
            
            # Get trade info - trade is a Series, access columns directly
            features['exit_reason'] = trade.get('exit_reason', 'unknown')
            features['return'] = trade.get('return', 0)
            features['pnl'] = trade.get('pnl', 0)
            features['holding_days'] = trade.get('holding_days', 0)
            
            all_features.append(features)
        
        except Exception as e:
            print(f"Error processing {ticker} at {entry_date}: {e}")
            continue
    
    if not all_features:
        return pd.DataFrame()
    
    return pd.DataFrame(all_features)


def analyze_stop_loss_patterns(trades: pd.DataFrame, features_df: pd.DataFrame) -> Dict:
    """
    Analyze patterns in stop-loss trades vs winners.
    
    Returns dictionary with analysis results.
    """
    results = {
        'stop_loss_trades': [],
        'winning_trades': [],
        'feature_comparisons': {},
        'recommendations': []
    }
    
    # Separate stop-loss trades from winners
    stop_loss_trades = trades[trades['exit_reason'] == 'stop_loss'].copy()
    winning_trades = trades[trades['return'] > 0].copy()
    target_trades = trades[trades['exit_reason'] == 'target_reached'].copy()
    
    results['stop_loss_count'] = len(stop_loss_trades)
    results['winning_count'] = len(winning_trades)
    results['target_count'] = len(target_trades)
    
    print(f"\n{'='*80}")
    print("STOP-LOSS ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal Trades: {len(trades)}")
    print(f"Stop-Loss Trades: {len(stop_loss_trades)} ({len(stop_loss_trades)/len(trades)*100:.1f}%)")
    print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
    print(f"Target Reached: {len(target_trades)} ({len(target_trades)/len(trades)*100:.1f}%)")
    
    # Analyze features if available
    if not features_df.empty:
        print(f"\n{'='*80}")
        print("FEATURE ANALYSIS: Stop-Loss vs Winners")
        print(f"{'='*80}")
        
        # Use features_df directly (it already has exit_reason, return, pnl from get_entry_features)
        if 'exit_reason' not in features_df.columns:
            print("Warning: exit_reason not found in features_df. Skipping feature analysis.")
            stop_loss_features = pd.DataFrame()
            winner_features = pd.DataFrame()
        else:
            stop_loss_features = features_df[features_df['exit_reason'] == 'stop_loss'].copy()
            winner_features = features_df[features_df['return'] > 0].copy()
        
        if not stop_loss_features.empty and not winner_features.empty:
            # Compare feature distributions
            feature_cols = [col for col in features_df.columns 
                          if col not in ['ticker', 'entry_date', 'exit_reason', 'return', 'pnl', 'holding_days']]
            
            comparisons = []
            for feat in feature_cols:
                try:
                    sl_mean = stop_loss_features[feat].mean()
                    win_mean = winner_features[feat].mean()
                    sl_std = stop_loss_features[feat].std()
                    win_std = winner_features[feat].std()
                    
                    if pd.notna(sl_mean) and pd.notna(win_mean) and sl_std > 0:
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt((sl_std**2 + win_std**2) / 2)
                        if pooled_std > 0:
                            cohens_d = (sl_mean - win_mean) / pooled_std
                            
                            comparisons.append({
                                'feature': feat,
                                'stop_loss_mean': sl_mean,
                                'winner_mean': win_mean,
                                'difference': sl_mean - win_mean,
                                'cohens_d': cohens_d,
                                'abs_effect': abs(cohens_d)
                            })
                except:
                    continue
            
            # Sort by effect size
            comparisons_df = pd.DataFrame(comparisons)
            if not comparisons_df.empty:
                comparisons_df = comparisons_df.sort_values('abs_effect', ascending=False)
                
                print("\nTop Features That Differ Between Stop-Loss and Winners:")
                print("-" * 80)
                print(f"{'Feature':<30} {'Stop-Loss Mean':<18} {'Winner Mean':<18} {'Difference':<15} {'Effect Size':<12}")
                print("-" * 80)
                
                for _, row in comparisons_df.head(20).iterrows():
                    print(f"{row['feature']:<30} {row['stop_loss_mean']:>15.4f}  {row['winner_mean']:>15.4f}  "
                          f"{row['difference']:>15.4f}  {row['cohens_d']:>12.3f}")
                
                results['feature_comparisons'] = comparisons_df.to_dict('records')
                
                # Generate recommendations
                print(f"\n{'='*80}")
                print("RECOMMENDATIONS")
                print(f"{'='*80}")
                
                recommendations = []
                
                # Look for features where stop-loss trades have significantly different values
                significant_features = comparisons_df[comparisons_df['abs_effect'] > 0.3]
                
                for _, row in significant_features.iterrows():
                    feat = row['feature']
                    sl_val = row['stop_loss_mean']
                    win_val = row['winner_mean']
                    
                    if sl_val > win_val:
                        rec = f"Filter: {feat} < {sl_val:.3f} (stop-loss trades have higher {feat})"
                    else:
                        rec = f"Filter: {feat} > {sl_val:.3f} (stop-loss trades have lower {feat})"
                    
                    recommendations.append(rec)
                    print(f"  • {rec}")
                
                results['recommendations'] = recommendations
    
    # Analyze timing patterns
    print(f"\n{'='*80}")
    print("TIMING ANALYSIS")
    print(f"{'='*80}")
    
    if len(stop_loss_trades) > 0:
        # Get entry dates for stop-loss trades
        if isinstance(stop_loss_trades.index, pd.DatetimeIndex):
            entry_dates_sl = stop_loss_trades.index
        elif 'entry_date' in stop_loss_trades.columns:
            entry_dates_sl = pd.to_datetime(stop_loss_trades['entry_date'])
        else:
            entry_dates_sl = None
        
        if entry_dates_sl is not None:
            stop_loss_trades = stop_loss_trades.copy()
            stop_loss_trades['day_of_week'] = entry_dates_sl.dayofweek
            stop_loss_trades['month'] = entry_dates_sl.month
            
            print("\nStop-Loss Trades by Day of Week:")
            if 'day_of_week' in stop_loss_trades.columns:
                dow_counts = stop_loss_trades['day_of_week'].value_counts().sort_index()
                dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                for dow, count in dow_counts.items():
                    pct = count / len(stop_loss_trades) * 100
                    print(f"  {dow_names[int(dow)]}: {count} ({pct:.1f}%)")
            
            print("\nStop-Loss Trades by Month:")
            if 'month' in stop_loss_trades.columns:
                month_counts = stop_loss_trades['month'].value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                for month, count in month_counts.items():
                    pct = count / len(stop_loss_trades) * 100
                    print(f"  {month_names[int(month)-1]}: {count} ({pct:.1f}%)")
    
    # Analyze how quickly stop-losses occur
    print(f"\n{'='*80}")
    print("STOP-LOSS TIMING")
    print(f"{'='*80}")
    
    if 'holding_days' in stop_loss_trades.columns:
        sl_holding = stop_loss_trades['holding_days']
        print(f"\nAverage days to stop-loss: {sl_holding.mean():.1f}")
        print(f"Median days to stop-loss: {sl_holding.median():.1f}")
        print(f"Min days to stop-loss: {sl_holding.min():.1f}")
        print(f"Max days to stop-loss: {sl_holding.max():.1f}")
        
        # Distribution
        print("\nDays to Stop-Loss Distribution:")
        bins = [0, 1, 2, 3, 5, 7, 10, 15, 30]
        for i in range(len(bins)-1):
            count = ((sl_holding >= bins[i]) & (sl_holding < bins[i+1])).sum()
            if count > 0:
                pct = count / len(sl_holding) * 100
                print(f"  {bins[i]}-{bins[i+1]} days: {count} ({pct:.1f}%)")
        
        immediate_stops = (sl_holding <= 1).sum()
        if immediate_stops > 0:
            pct = immediate_stops / len(sl_holding) * 100
            print(f"\n⚠️  {immediate_stops} stop-losses ({pct:.1f}%) occurred within 1 day - possible bad entries!")
            results['recommendations'].append(f"Filter out trades that hit stop-loss within 1 day (bad entry timing)")
    
    # Analyze return distribution
    print(f"\n{'='*80}")
    print("RETURN ANALYSIS")
    print(f"{'='*80}")
    
    if 'return' in stop_loss_trades.columns:
        sl_returns = stop_loss_trades['return'] * 100
        print(f"\nStop-Loss Return Distribution:")
        print(f"  Mean: {sl_returns.mean():.2f}%")
        print(f"  Median: {sl_returns.median():.2f}%")
        print(f"  Min: {sl_returns.min():.2f}%")
        print(f"  Max: {sl_returns.max():.2f}%")
        print(f"  Std Dev: {sl_returns.std():.2f}%")
        
        # Check if many are hitting exactly at stop-loss threshold
        if 'stop_loss' in trades.columns or 'exit_reason' in trades.columns:
            # Most should be at or near the stop-loss threshold
            near_threshold = (sl_returns <= -7.0) & (sl_returns >= -8.0)
            if near_threshold.sum() > 0:
                pct = near_threshold.sum() / len(sl_returns) * 100
                print(f"\n  {near_threshold.sum()} ({pct:.1f}%) hit stop-loss between -7% and -8% (expected range)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stop-loss trades to identify patterns and reduce stop-loss rate"
    )
    parser.add_argument(
        "--trades-csv",
        type=str,
        default=None,
        help="CSV file with backtest trades (if not provided, will run backtest)"
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Trade window (for running backtest if trades-csv not provided)"
    )
    parser.add_argument(
        "--return-threshold",
        type=float,
        default=0.15,
        help="Return threshold (for running backtest if trades-csv not provided)"
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=0.85,
        help="Model threshold (for running backtest if trades-csv not provided)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop-loss threshold (for running backtest if trades-csv not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for analysis results (JSON)"
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        default=True,
        help="Use validation data (2023) for analysis instead of test data (default: True)"
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2022-12-31",
        help="Training data end date (YYYY-MM-DD). Default: 2022-12-31"
    )
    parser.add_argument(
        "--val-end",
        type=str,
        default="2023-12-31",
        help="Validation data end date (YYYY-MM-DD). Default: 2023-12-31"
    )
    
    args = parser.parse_args()
    
    # Load or generate trades
    if args.trades_csv:
        print(f"Loading trades from {args.trades_csv}...")
        trades = load_trades_from_csv(Path(args.trades_csv))
    else:
        print("Running backtest to generate trades...")
        # Load model
        model_path = Path(DEFAULT_MODEL)
        model, features, scaler, features_to_scale, _ = load_model(model_path)
        
        # Load tickers
        tickers_df = pd.read_csv(TICKERS_CSV, header=None)
        tickers = tickers_df.iloc[:, 0].astype(str).tolist()
        
        # Calculate stop-loss
        stop_loss_value = args.stop_loss
        if stop_loss_value is None and args.return_threshold is not None:
            stop_loss_value = -abs(args.return_threshold) / 2
        
        # Determine date range for analysis
        if args.use_validation:
            # Use validation data (between train_end and val_end)
            analysis_start = args.train_end
            analysis_end = args.val_end
            print(f"\nUsing VALIDATION data for analysis: {analysis_start} to {analysis_end}")
        else:
            # Use test data (after val_end)
            analysis_start = args.val_end
            analysis_end = None  # Use all available test data
            print(f"\nUsing TEST data for analysis: {analysis_start} onwards")
        
        # Run backtest on validation/test data
        trades = run_backtest(
            tickers=tickers,
            data_dir=DATA_DIR,
            horizon=args.horizon,
            return_threshold=args.return_threshold,
            position_size=1000.0,
            model=model,
            features=features,
            model_threshold=args.model_threshold,
            strategy="model",
            stop_loss=stop_loss_value,
            scaler=scaler,
            features_to_scale=features_to_scale,
            test_start_date=analysis_start,
            test_end_date=analysis_end if args.use_validation else None
        )
        
        if trades.empty:
            print("No trades generated. Cannot analyze.")
            return
    
    print(f"\nLoaded {len(trades)} trades")
    
    # Get feature values at entry time
    print("\nExtracting feature values at entry time...")
    features_df = get_entry_features(trades, DATA_DIR)
    
    if features_df.empty:
        print("Warning: Could not extract features. Analysis will be limited.")
        features_df = pd.DataFrame()
    
    # Analyze stop-loss patterns
    results = analyze_stop_loss_patterns(trades, features_df)
    
    # Save results if requested
    if args.output:
        import json
        # Convert to JSON-serializable format
        output_data = {
            'stop_loss_count': results.get('stop_loss_count', 0),
            'winning_count': results.get('winning_count', 0),
            'target_count': results.get('target_count', 0),
            'recommendations': results.get('recommendations', [])
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

