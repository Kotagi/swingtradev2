#!/usr/bin/env python3
"""
enhanced_backtest.py

Enhanced backtesting with user-defined trade window and return threshold.

This script:
  1. Loads labeled feature data
  2. Runs backtests with configurable parameters:
     - Trade window (holding period in days)
     - Return threshold (minimum return to consider a win)
     - Position size
     - Entry signal (model prediction or custom strategy)
  3. Generates comprehensive performance metrics
  4. Outputs detailed reports
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "features_labeled"
MODEL_DIR = PROJECT_ROOT / "models"
TICKERS_CSV = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
DEFAULT_MODEL = MODEL_DIR / "xgb_classifier_selected_features.pkl"


def load_model(model_path: Path):
    """Load trained model, features, and scaler."""
    if not model_path.exists():
        return None, [], None, [], []
    
    data = joblib.load(model_path)
    if isinstance(data, dict):
        return (
            data.get("model"),
            data.get("features", []),
            data.get("scaler"),  # May be None if no scaling was used
            data.get("features_to_scale", []),
            data.get("features_to_keep", [])
        )
    # Legacy format (no scaler)
    return data, [], None, [], []


def backtest_strategy(
    df: pd.DataFrame,
    signal_col: str,
    horizon: int,
    position_size: float,
    return_threshold: float = None,
    stop_loss: float = None,
    open_col: str = None,
    close_col: str = None
) -> pd.DataFrame:
    """
    Backtest a boolean entry signal with exit at return threshold, stop-loss, OR time horizon (whichever comes first).

    Args:
        df: DataFrame indexed by date with OHLCV and signal column.
        signal_col: Name of boolean column indicating entry.
        horizon: Maximum holding period in days.
        position_size: Dollar amount per trade.
        return_threshold: Target return threshold (e.g., 0.15 for 15%). If None, only uses time horizon.
        stop_loss: Stop-loss threshold (e.g., -0.075 for -7.5%). If None, no stop-loss is used.
        open_col: Name of open price column (auto-detected if None).
        close_col: Name of close price column (auto-detected if None).

    Returns:
        DataFrame of trades with columns: entry_date, entry_price, exit_date, 
        exit_price, return, pnl, holding_days, exit_reason.
    """
    # Auto-detect column names (case-insensitive)
    if open_col is None:
        open_col = "Open" if "Open" in df.columns else "open"
    if close_col is None:
        close_col = "Close" if "Close" in df.columns else "close"
    
    # Verify columns exist
    if open_col not in df.columns:
        raise KeyError(f"Open column '{open_col}' not found. Available columns: {list(df.columns)}")
    if close_col not in df.columns:
        raise KeyError(f"Close column '{close_col}' not found. Available columns: {list(df.columns)}")
    
    df_sorted = df.sort_index().copy()
    dates = df_sorted.index
    
    trades = []
    last_exit_date = None  # Track when the last position closed
    
    for i, (date, row) in enumerate(df_sorted.iterrows()):
        # Skip if we have an open position (don't enter duplicate trades)
        # Only enter a new trade if the previous one has closed
        if last_exit_date is not None and date <= last_exit_date:
            continue
        
        if not row.get(signal_col):
            continue
        
        entry_price = row[open_col]
        entry_idx = i
        
        # Find exit: check each day until either threshold is hit or horizon is reached
        exit_date = None
        exit_price = None
        exit_reason = None
        
        # Check up to horizon days ahead
        for days_ahead in range(1, horizon + 1):
            exit_idx = entry_idx + days_ahead
            
            # Check if we've run out of data
            if exit_idx >= len(df_sorted):
                # Use last available price
                exit_idx = len(df_sorted) - 1
                exit_date = dates[exit_idx]
                exit_price = df_sorted.iloc[exit_idx][close_col]
                exit_reason = "end_of_data"
                break
            
            exit_date = dates[exit_idx]
            exit_price = df_sorted.iloc[exit_idx][close_col]
            
            # Calculate return so far
            current_return = (exit_price / entry_price) - 1
            
            # Check if stop-loss is hit (exit early on large losses)
            if stop_loss is not None and current_return <= stop_loss:
                exit_reason = "stop_loss"
                break
            
            # Check if return threshold is reached
            if return_threshold is not None and current_return >= return_threshold:
                exit_reason = "target_reached"
                break
            
            # If we've reached the horizon, exit
            if days_ahead == horizon:
                exit_reason = "time_limit"
                break
        
        if exit_price is None or pd.isna(exit_price):
            continue
        
        ret = (exit_price / entry_price) - 1
        pnl = ret * position_size
        holding_days = (exit_date - date).days if exit_date else horizon
        
        # Record this position's exit date to prevent overlapping trades
        if last_exit_date is None or exit_date > last_exit_date:
            last_exit_date = exit_date
        
        trades.append({
            "entry_date": date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "return": ret,
            "pnl": pnl,
            "holding_days": holding_days,
            "exit_reason": exit_reason
        })
    
    if not trades:
        return pd.DataFrame()
    
    return pd.DataFrame(trades).set_index("entry_date")


def apply_entry_filters(
    df: pd.DataFrame,
    filters: Dict[str, Tuple[str, float]]
) -> pd.Series:
    """
    Apply entry filters to a DataFrame.
    
    Args:
        df: DataFrame with feature columns
        filters: Dict of {feature_name: (operator, threshold)} where operator is '>', '<', '>=', '<='
    
    Returns:
        Boolean Series indicating which rows pass all filters
    """
    if not filters:
        return pd.Series(True, index=df.index)
    
    # Start with all True
    mask = pd.Series(True, index=df.index)
    
    for feature, (operator, threshold) in filters.items():
        if feature not in df.columns:
            # Feature not available - skip this filter
            continue
        
        feature_values = df[feature]
        
        if operator == '>':
            mask = mask & (feature_values > threshold)
        elif operator == '<':
            mask = mask & (feature_values < threshold)
        elif operator == '>=':
            mask = mask & (feature_values >= threshold)
        elif operator == '<=':
            mask = mask & (feature_values <= threshold)
        else:
            print(f"Warning: Unknown operator '{operator}' for feature '{feature}'")
    
    return mask


def generate_model_signals(
    df: pd.DataFrame,
    model,
    features: List[str],
    threshold: float = 0.5,
    scaler=None,
    features_to_scale: List[str] = None
) -> pd.Series:
    """
    Generate entry signals using a trained model.

    Args:
        df: DataFrame with feature columns.
        model: Trained model.
        features: List of feature names.
        threshold: Probability threshold for entry signal.
        scaler: StandardScaler fitted on training data (optional).
        features_to_scale: List of feature names that should be scaled (optional).

    Returns:
        Boolean Series indicating entry signals.
    """
    # Extract features
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features) * 0.8:
        return pd.Series(False, index=df.index)
    
    X = df[available_features].copy()
    
    # Fill missing features
    for f in features:
        if f not in X.columns:
            X[f] = 0.0
    
    X = X[features]
    
    # Apply scaling if scaler is provided
    if scaler is not None and features_to_scale:
        X_scaled = X.copy()
        # Only scale features that were scaled during training
        scale_cols = [f for f in features_to_scale if f in X.columns]
        if scale_cols:
            X_scaled[scale_cols] = scaler.transform(X[scale_cols])
        X = X_scaled
    
    # Make predictions
    try:
        proba = model.predict_proba(X)[:, 1]
        signals = proba >= threshold
        return pd.Series(signals, index=df.index)
    except Exception as e:
        print(f"Error generating signals: {e}")
        return pd.Series(False, index=df.index)


def calculate_metrics(trades: pd.DataFrame, position_size: float = 1000.0) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        trades: DataFrame of trades from backtest_strategy.

    Returns:
        Dictionary of performance metrics.
    """
    if trades.empty:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_holding_days": 0.0
        }
    
    n_trades = len(trades)
    wins = (trades["return"] > 0).sum()
    win_rate = wins / n_trades if n_trades > 0 else 0.0
    
    avg_return = trades["return"].mean()
    total_pnl = trades["pnl"].sum()
    avg_pnl = trades["pnl"].mean()
    
    # Maximum drawdown
    equity = trades["pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = running_max - equity
    max_drawdown = drawdown.max()
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    returns = trades["return"]
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / trades["holding_days"].mean()) if trades["holding_days"].mean() > 0 else 0.0
    else:
        sharpe = 0.0
    
    # Profit factor
    gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_holding_days = trades["holding_days"].mean()
    
    # Calculate maximum concurrent positions (max capital invested)
    max_concurrent_positions = 0
    max_capital_invested = 0.0
    
    if n_trades > 0:
        # Get entry and exit dates
        if isinstance(trades.index, pd.DatetimeIndex):
            entry_dates = trades.index
        elif "entry_date" in trades.columns:
            entry_dates = pd.to_datetime(trades["entry_date"])
        else:
            entry_dates = pd.DatetimeIndex([])
        
        if "exit_date" in trades.columns:
            exit_dates = pd.to_datetime(trades["exit_date"])
        else:
            exit_dates = entry_dates + pd.Timedelta(days=avg_holding_days)
        
        if len(entry_dates) > 0:
            # Create a timeline of all entry and exit events
            events = []
            for entry, exit_date in zip(entry_dates, exit_dates):
                events.append((entry, 1))  # +1 for entry
                events.append((exit_date, -1))  # -1 for exit
            
            # Sort events by date
            events.sort(key=lambda x: x[0])
            
            # Track concurrent positions over time
            current_positions = 0
            max_concurrent_positions = 0
            
            for date, change in events:
                current_positions += change
                max_concurrent_positions = max(max_concurrent_positions, current_positions)
            
            # Calculate max capital invested
            max_capital_invested = max_concurrent_positions * position_size
    
    # Calculate annual return and date range
    annual_return = 0.0
    date_range = "N/A"
    
    if n_trades > 0:
        # Get date range
        # Trades DataFrame has entry_date as index and exit_date as column
        if isinstance(trades.index, pd.DatetimeIndex):
            first_entry = trades.index.min()
        elif "entry_date" in trades.columns:
            first_entry = trades["entry_date"].min()
        else:
            first_entry = pd.Timestamp.now()  # Fallback
        
        if "exit_date" in trades.columns:
            last_exit = trades["exit_date"].max()
        else:
            # Estimate from last entry + average holding days
            if isinstance(trades.index, pd.DatetimeIndex):
                last_entry = trades.index.max()
                last_exit = last_entry + pd.Timedelta(days=avg_holding_days)
            else:
                last_exit = first_entry + pd.Timedelta(days=365)  # Fallback
        
        # Convert to datetime if needed
        if not isinstance(first_entry, pd.Timestamp):
            first_entry = pd.to_datetime(first_entry)
        if not isinstance(last_exit, pd.Timestamp):
            last_exit = pd.to_datetime(last_exit)
        
        date_range = f"{first_entry.strftime('%Y-%m-%d')} to {last_exit.strftime('%Y-%m-%d')}"
        
        # Calculate annual return
        # Annual return = (1 + total_return) ^ (365 / days_in_backtest) - 1
        # Where total_return = total_pnl / initial_capital
        # Use max_capital_invested as initial capital (accounts for concurrent positions)
        days_in_backtest = (last_exit - first_entry).days
        if days_in_backtest > 0 and max_capital_invested > 0:
            # Calculate total return based on max capital invested
            # If we start with max_capital_invested and end with max_capital_invested + total_pnl
            total_return = total_pnl / max_capital_invested
            # Annualize using compound return
            annual_return = ((1 + total_return) ** (365.0 / days_in_backtest)) - 1
        elif days_in_backtest > 0 and position_size > 0:
            # Fallback: if no concurrent positions, use position_size
            total_return = total_pnl / position_size
            annual_return = ((1 + total_return) ** (365.0 / days_in_backtest)) - 1
        else:
            annual_return = 0.0
    
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_holding_days,
        "annual_return": annual_return,
        "date_range": date_range,
        "max_concurrent_positions": max_concurrent_positions,
        "max_capital_invested": max_capital_invested
    }


def run_backtest(
    tickers: List[str],
    data_dir: Path,
    horizon: int,
    return_threshold: float,
    position_size: float,
    model=None,
    features: List[str] = None,
    model_threshold: float = 0.5,
    strategy: str = "model",
    stop_loss: float = None,
    scaler=None,
    features_to_scale: List[str] = None,
    test_start_date: str = None,
    test_end_date: str = None,
    entry_filters: Dict[str, tuple] = None
) -> pd.DataFrame:
    """
    Run backtest across multiple tickers.

    Args:
        tickers: List of ticker symbols.
        data_dir: Directory containing feature Parquet files.
        horizon: Holding period in days.
        return_threshold: Minimum return to consider a win (for labeling).
        position_size: Dollar amount per trade.
        model: Trained model (optional).
        features: List of feature names (optional).
        model_threshold: Probability threshold for model signals.
        strategy: Strategy type ("model", "oracle", "rsi").

    Returns:
        DataFrame with all trades across all tickers.
    """
    all_trades = []
    error_count = 0
    max_errors_to_show = 5
    
    label_col = f"label_{horizon}d"
    
    for ticker in tickers:
        file_path = data_dir / f"{ticker}.parquet"
        if not file_path.exists():
            continue
        
        try:
            df = pd.read_parquet(file_path).sort_index()
            
            # Filter by date range (for validation or test data)
            if test_start_date is not None:
                test_start = pd.to_datetime(test_start_date)
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[df.index >= test_start]
                elif "entry_date" in df.columns:
                    df = df[pd.to_datetime(df["entry_date"]) >= test_start]
            
            if test_end_date is not None:
                test_end = pd.to_datetime(test_end_date)
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[df.index <= test_end]
                elif "entry_date" in df.columns:
                    df = df[pd.to_datetime(df["entry_date"]) <= test_end]
            
            if df.empty:
                continue
            
            # Determine entry signal based on strategy
            if strategy == "model" and model is not None and features:
                df["entry_signal"] = generate_model_signals(
                    df, model, features, model_threshold, scaler, features_to_scale
                )
            elif strategy == "oracle":
                if label_col in df.columns:
                    df["entry_signal"] = df[label_col] == 1
                else:
                    continue
            elif strategy == "rsi":
                # Try different possible RSI column names
                rsi_col = None
                for col in ["rsi_14", "rsi"]:
                    if col in df.columns:
                        rsi_col = col
                        break
                
                if rsi_col:
                    df["entry_signal"] = df[rsi_col] < 30
                else:
                    continue
            else:
                continue
            
            # Apply entry filters if provided
            if entry_filters:
                filter_mask = apply_entry_filters(df, entry_filters)
                # Only enter when both model signal AND filters pass
                df["entry_signal"] = df["entry_signal"] & filter_mask
            
            # Calculate 2:1 risk-reward stop-loss if return_threshold is provided
            calculated_stop_loss = None
            if return_threshold is not None and stop_loss is None:
                calculated_stop_loss = -abs(return_threshold) / 2  # 2:1 risk-reward ratio
            elif stop_loss is not None:
                calculated_stop_loss = stop_loss
            
            # Run backtest for this ticker
            trades = backtest_strategy(
                df,
                signal_col="entry_signal",
                horizon=horizon,
                position_size=position_size,
                return_threshold=return_threshold,
                stop_loss=calculated_stop_loss
            )
            
            if not trades.empty:
                trades["ticker"] = ticker
                all_trades.append(trades)
        
        except Exception as e:
            # Only print first few errors to avoid spam
            error_count += 1
            if error_count <= max_errors_to_show:
                print(f"Error processing {ticker}: {e}")
            elif error_count == max_errors_to_show + 1:
                print(f"... (suppressing further error messages)")
            continue
    
    if error_count > 0:
        print(f"\nNote: {error_count} ticker(s) had errors and were skipped")
    
    if not all_trades:
        return pd.DataFrame()
    
    return pd.concat(all_trades)


def main():
    """Main entry point for enhanced backtesting."""
    parser = argparse.ArgumentParser(
        description="Enhanced backtesting with configurable parameters"
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        default=str(TICKERS_CSV),
        help="Path to CSV file with ticker symbols"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing feature Parquet files"
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
        default=0.0,
        help="Minimum return threshold for labeling (as decimal, e.g., 0.05 for 5%%)"
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=1000.0,
        help="Position size per trade in dollars"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["model", "oracle", "rsi"],
        default="model",
        help="Backtest strategy: model (use trained model), oracle (perfect hindsight), rsi (RSI < 30)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained model (required for model strategy)"
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for model signals (0.0-1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for trades (optional)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop-loss threshold as decimal (e.g., -0.075 for -7.5%%). If not specified and return-threshold is provided, uses 2:1 risk-reward (return_threshold / 2)."
    )
    parser.add_argument(
        "--test-start-date",
        type=str,
        default="2024-01-01",
        help="Start date for test data (YYYY-MM-DD). Only data after this date will be backtested. Default: 2024-01-01 (to avoid data leakage from training/validation)."
    )
    
    args = parser.parse_args()
    
    # Load model if needed
    model = None
    features = []
    scaler = None
    features_to_scale = []
    if args.strategy == "model":
        print(f"Loading model from {args.model}...")
        model, features, scaler, features_to_scale, _ = load_model(Path(args.model))
        if model is None:
            print("ERROR: Model not found. Cannot run model strategy.")
            return
        print(f"Model loaded. Using {len(features)} features.")
        if scaler is not None:
            print(f"Scaler loaded. Will scale {len(features_to_scale)} features during prediction.")
    
    # Load tickers
    tickers_df = pd.read_csv(args.tickers_file, header=None)
    tickers = tickers_df.iloc[:, 0].astype(str).tolist()
    print(f"Loaded {len(tickers)} tickers")
    
    # Run backtest
    # Calculate 2:1 risk-reward stop-loss if return_threshold is provided
    stop_loss_value = getattr(args, 'stop_loss', None)
    if stop_loss_value is None and args.return_threshold is not None:
        stop_loss_value = -abs(args.return_threshold) / 2  # 2:1 risk-reward ratio
    
    print(f"\nRunning backtest with:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Return threshold: {args.return_threshold:.2%}")
    if stop_loss_value is not None:
        print(f"  Stop-Loss: {stop_loss_value:.2%} (2:1 risk-reward)")
    print(f"  Position size: ${args.position_size:,.2f}")
    if args.strategy == "model":
        print(f"  Model threshold: {args.model_threshold:.2f}")
    print(f"  Test data start date: {args.test_start_date} (only data after this date will be used)")
    
    trades = run_backtest(
        tickers=tickers,
        data_dir=Path(args.data_dir),
        horizon=args.horizon,
        return_threshold=args.return_threshold,
        position_size=args.position_size,
        model=model,
        features=features,
        model_threshold=args.model_threshold,
        strategy=args.strategy,
        stop_loss=stop_loss_value,
        scaler=scaler,
        features_to_scale=features_to_scale,
        test_start_date=args.test_start_date
    )
    
    # Calculate metrics
    metrics = calculate_metrics(trades, args.position_size)
    
    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"\nStrategy: {args.strategy}")
    print(f"Trade Window: {args.horizon} days")
    print(f"Return Threshold: {args.return_threshold:.2%}")
    if stop_loss_value is not None:
        print(f"Stop-Loss: {stop_loss_value:.2%} (2:1 risk-reward)")
    print(f"Position Size: ${args.position_size:,.2f}")
    print(f"\nBacktest Period: {metrics.get('date_range', 'N/A')}")
    print("\nPerformance Metrics:")
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
    
    # Show exit reason breakdown if available
    if not trades.empty and 'exit_reason' in trades.columns:
        exit_reasons = trades['exit_reason'].value_counts()
        print(f"\n  Exit Reasons:")
        for reason, count in exit_reasons.items():
            pct = (count / len(trades)) * 100
            print(f"    {reason}: {count:,} ({pct:.1f}%)")
        
        # Analyze time_limit trades
        if 'time_limit' in exit_reasons.index:
            time_limit_trades = trades[trades['exit_reason'] == 'time_limit']
            print(f"\n  Time Limit Trades Analysis:")
            print(f"    Total: {len(time_limit_trades):,}")
            
            # Win/loss breakdown
            wins = (time_limit_trades['return'] > 0).sum()
            losses = (time_limit_trades['return'] < 0).sum()
            breakeven = (time_limit_trades['return'] == 0).sum()
            
            print(f"    Wins: {wins:,} ({wins/len(time_limit_trades)*100:.1f}%)")
            print(f"    Losses: {losses:,} ({losses/len(time_limit_trades)*100:.1f}%)")
            print(f"    Breakeven: {breakeven:,} ({breakeven/len(time_limit_trades)*100:.1f}%)")
            
            # Return distribution
            if len(time_limit_trades) > 0:
                avg_return = time_limit_trades['return'].mean() * 100
                median_return = time_limit_trades['return'].median() * 100
                print(f"    Average Return: {avg_return:.2f}%")
                print(f"    Median Return: {median_return:.2f}%")
                
                # Return ranges
                positive_returns = time_limit_trades[time_limit_trades['return'] > 0]['return'] * 100
                negative_returns = time_limit_trades[time_limit_trades['return'] < 0]['return'] * 100
                
                if len(positive_returns) > 0:
                    print(f"    Avg Winning Return: {positive_returns.mean():.2f}%")
                    print(f"    Median Winning Return: {positive_returns.median():.2f}%")
                    print(f"    Max Winning Return: {positive_returns.max():.2f}%")
                    print(f"    Min Winning Return: {positive_returns.min():.2f}%")
                
                if len(negative_returns) > 0:
                    print(f"    Avg Losing Return: {negative_returns.mean():.2f}%")
                    print(f"    Median Losing Return: {negative_returns.median():.2f}%")
                    print(f"    Max Losing Return: {negative_returns.max():.2f}%")
                    print(f"    Min Losing Return: {negative_returns.min():.2f}%")
                
                # Return distribution buckets
                print(f"\n    Return Distribution:")
                buckets = [
                    (-float('inf'), -0.10, "< -10%"),
                    (-0.10, -0.05, "-10% to -5%"),
                    (-0.05, 0.0, "-5% to 0%"),
                    (0.0, 0.05, "0% to 5%"),
                    (0.05, 0.10, "5% to 10%"),
                    (0.10, 0.15, "10% to 15%"),
                    (0.15, float('inf'), "> 15%")
                ]
                
                for low, high, label in buckets:
                    count = ((time_limit_trades['return'] > low) & (time_limit_trades['return'] <= high)).sum()
                    if count > 0:
                        pct = (count / len(time_limit_trades)) * 100
                        print(f"      {label}: {count:,} ({pct:.1f}%)")
    
    print("="*80)
    
    # Save trades if requested
    if args.output and not trades.empty:
        trades.to_csv(args.output)
        print(f"\nTrades saved to {args.output}")


if __name__ == "__main__":
    main()

