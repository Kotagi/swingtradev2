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
from typing import List, Dict, Optional
from datetime import datetime

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data" / "features_labeled"
MODEL_DIR = PROJECT_ROOT / "models"
TICKERS_CSV = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
DEFAULT_MODEL = MODEL_DIR / "xgb_classifier_selected_features.pkl"


def load_model(model_path: Path):
    """Load trained model and features."""
    if not model_path.exists():
        return None, []
    
    data = joblib.load(model_path)
    if isinstance(data, dict):
        return data.get("model"), data.get("features", [])
    return data, []


def backtest_strategy(
    df: pd.DataFrame,
    signal_col: str,
    horizon: int,
    position_size: float,
    open_col: str = None,
    close_col: str = None
) -> pd.DataFrame:
    """
    Backtest a boolean entry signal over a fixed holding period.

    Args:
        df: DataFrame indexed by date with OHLCV and signal column.
        signal_col: Name of boolean column indicating entry.
        horizon: Holding period in days.
        position_size: Dollar amount per trade.
        open_col: Name of open price column (auto-detected if None).
        close_col: Name of close price column (auto-detected if None).

    Returns:
        DataFrame of trades with columns: entry_date, entry_price, exit_date, 
        exit_price, return, pnl, holding_days.
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
    
    # Precompute exit prices
    df_sorted["exit_price"] = df_sorted[close_col].shift(-horizon)
    df_sorted["exit_date"] = df_sorted.index.to_series().shift(-horizon)
    
    trades = []
    for date, row in df_sorted.iterrows():
        if row.get(signal_col) and pd.notna(row["exit_price"]):
            entry_price = row[open_col]
            exit_price = row["exit_price"]
            exit_date = row["exit_date"]
            ret = exit_price / entry_price - 1
            pnl = ret * position_size
            holding_days = (exit_date - date).days if pd.notna(exit_date) else horizon
            
            trades.append({
                "entry_date": date,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "return": ret,
                "pnl": pnl,
                "holding_days": holding_days
            })
    
    if not trades:
        return pd.DataFrame()
    
    return pd.DataFrame(trades).set_index("entry_date")


def generate_model_signals(
    df: pd.DataFrame,
    model,
    features: List[str],
    threshold: float = 0.5
) -> pd.Series:
    """
    Generate entry signals using a trained model.

    Args:
        df: DataFrame with feature columns.
        model: Trained model.
        features: List of feature names.
        threshold: Probability threshold for entry signal.

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
    
    # Make predictions
    try:
        proba = model.predict_proba(X)[:, 1]
        signals = proba >= threshold
        return pd.Series(signals, index=df.index)
    except Exception as e:
        print(f"Error generating signals: {e}")
        return pd.Series(False, index=df.index)


def calculate_metrics(trades: pd.DataFrame) -> Dict:
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
    
    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_holding_days
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
    strategy: str = "model"
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
            
            # Determine entry signal based on strategy
            if strategy == "model" and model is not None and features:
                df["entry_signal"] = generate_model_signals(df, model, features, model_threshold)
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
            
            # Run backtest for this ticker
            trades = backtest_strategy(
                df,
                signal_col="entry_signal",
                horizon=horizon,
                position_size=position_size
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
    
    args = parser.parse_args()
    
    # Load model if needed
    model = None
    features = []
    if args.strategy == "model":
        print(f"Loading model from {args.model}...")
        model, features = load_model(Path(args.model))
        if model is None:
            print("ERROR: Model not found. Cannot run model strategy.")
            return
        print(f"Model loaded. Using {len(features)} features.")
    
    # Load tickers
    tickers_df = pd.read_csv(args.tickers_file, header=None)
    tickers = tickers_df.iloc[:, 0].astype(str).tolist()
    print(f"Loaded {len(tickers)} tickers")
    
    # Run backtest
    print(f"\nRunning backtest with:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Return threshold: {args.return_threshold:.2%}")
    print(f"  Position size: ${args.position_size:,.2f}")
    if args.strategy == "model":
        print(f"  Model threshold: {args.model_threshold:.2f}")
    
    trades = run_backtest(
        tickers=tickers,
        data_dir=Path(args.data_dir),
        horizon=args.horizon,
        return_threshold=args.return_threshold,
        position_size=args.position_size,
        model=model,
        features=features,
        model_threshold=args.model_threshold,
        strategy=args.strategy
    )
    
    # Calculate metrics
    metrics = calculate_metrics(trades)
    
    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"\nStrategy: {args.strategy}")
    print(f"Trade Window: {args.horizon} days")
    print(f"Return Threshold: {args.return_threshold:.2%}")
    print(f"Position Size: ${args.position_size:,.2f}")
    print("\nPerformance Metrics:")
    print(f"  Total Trades: {metrics['n_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Average Return: {metrics['avg_return']:.2%}")
    print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
    print(f"  Average P&L per Trade: ${metrics['avg_pnl']:,.2f}")
    print(f"  Maximum Drawdown: ${metrics['max_drawdown']:,.2f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Average Holding Days: {metrics['avg_holding_days']:.1f}")
    print("="*80)
    
    # Save trades if requested
    if args.output and not trades.empty:
        trades.to_csv(args.output)
        print(f"\nTrades saved to {args.output}")


if __name__ == "__main__":
    main()

