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
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add project root and src to Python path
# Resolve to absolute path to ensure correct resolution when run as subprocess
_script_file = Path(__file__).resolve()
PROJECT_ROOT = _script_file.parent.parent
SRC_DIR = _script_file.parent

# Ensure project root is in path (check first to avoid duplicates)
_project_root_str = str(PROJECT_ROOT)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

# Also add src directory
_src_dir_str = str(SRC_DIR)
if _src_dir_str not in sys.path:
    sys.path.insert(0, _src_dir_str)

# Verify utils package exists before importing
_utils_path = PROJECT_ROOT / "utils" / "stop_loss_policy.py"
if not _utils_path.exists():
    raise ImportError(
        f"Cannot find utils.stop_loss_policy module. "
        f"Expected at: {_utils_path}. "
        f"PROJECT_ROOT: {PROJECT_ROOT}. "
        f"Current working directory: {Path.cwd()}"
    )

from utils.stop_loss_policy import (
    StopLossConfig,
    calculate_stop_loss_pct,
    create_stop_loss_config_from_args,
    summarize_adaptive_stops
)

# —— CONFIGURATION —— #
# Use the PROJECT_ROOT already defined above
DATA_DIR = PROJECT_ROOT / "data" / "features_labeled"
MODEL_DIR = PROJECT_ROOT / "models"
TICKERS_CSV = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
DEFAULT_MODEL = MODEL_DIR / "xgb_classifier_selected_features.pkl"


def load_model(model_path: Path):
    """Load trained model and features from pickle file.
    
    Uses the feature list the model was actually trained on: prefers "features_to_keep"
    (trained subset) when non-empty, else "features". The feature set parquet may contain
    more columns than the model uses; we must pass only the trained feature list for prediction.
    
    Note: scaler and features_to_scale are kept for backward compatibility but are not used
    (XGBoost doesn't require feature scaling).
    """
    if not model_path.exists():
        return None, [], None, [], []
    
    data = joblib.load(model_path)
    if isinstance(data, dict):
        # Use trained feature list: subset (features_to_keep) if present, else full "features"
        features_raw = data.get("features", [])
        features_to_keep = data.get("features_to_keep", [])
        features = list(features_to_keep) if features_to_keep else list(features_raw)
        return (
            data.get("model"),
            features,
            data.get("scaler"),  # Always None - not used (XGBoost doesn't require scaling)
            data.get("features_to_scale", []),  # Always empty - not used
            features_to_keep or features_raw
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
    stop_loss_config: StopLossConfig = None,
    open_col: str = None,
    close_col: str = None
) -> pd.DataFrame:
    """
    Backtest a boolean entry signal with exit at return threshold, stop-loss, OR time horizon (whichever comes first).

    Args:
        df: DataFrame indexed by date with OHLCV and signal column.
        signal_col: Name of boolean column indicating entry.
        horizon: Maximum holding period in trading days.
        position_size: Dollar amount per trade.
        return_threshold: Target return threshold (e.g., 0.15 for 15%). If None, only uses time horizon.
        stop_loss: Legacy stop-loss threshold (e.g., -0.075 for -7.5%). If None, no stop-loss is used.
                   DEPRECATED: Use stop_loss_config instead for adaptive stops.
        stop_loss_config: StopLossConfig object for adaptive stop-loss behavior. If None and stop_loss is None,
                         no stop-loss is used. If stop_loss is provided, creates a constant config.
        open_col: Name of open price column (auto-detected if None).
        close_col: Name of close price column (auto-detected if None).

    Returns:
        DataFrame of trades with columns: entry_date, entry_price, exit_date, 
        exit_price, return, pnl, holding_days, exit_reason, stop_loss_pct (if adaptive).
    """
    # Auto-detect column names (case-insensitive)
    if open_col is None:
        open_col = "Open" if "Open" in df.columns else "open"
    if close_col is None:
        close_col = "Close" if "Close" in df.columns else "close"
    
    # Auto-detect low column for stop loss gap handling
    low_col = "Low" if "Low" in df.columns else ("low" if "low" in df.columns else None)
    
    # Verify required columns exist
    if open_col not in df.columns:
        raise KeyError(f"Open column '{open_col}' not found. Available columns: {list(df.columns)}")
    if close_col not in df.columns:
        raise KeyError(f"Close column '{close_col}' not found. Available columns: {list(df.columns)}")
    
    # Determine stop-loss configuration
    # Priority: stop_loss_config > stop_loss (legacy) > None
    if stop_loss_config is None and stop_loss is not None:
        # Create constant config from legacy stop_loss parameter
        stop_loss_config = StopLossConfig(
            mode="constant",
            constant_stop_loss_pct=abs(stop_loss)
        )
    
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
        
        # Calculate stop-loss for this trade (per-trade calculation)
        trade_stop_loss = None
        trade_stop_loss_pct = None
        trade_stop_loss_price = None  # Absolute stop loss price
        trade_stop_metadata = {}
        if stop_loss_config is not None:
            # Calculate stop percentage using centralized policy
            # Pass entry_price for swing_atr mode
            trade_stop_loss_pct, trade_stop_metadata = calculate_stop_loss_pct(
                row, stop_loss_config, entry_price=entry_price
            )
            # Convert to negative value for comparison with returns
            trade_stop_loss = -trade_stop_loss_pct
            # Calculate absolute stop loss price
            trade_stop_loss_price = entry_price * (1 + trade_stop_loss)
        
        # Find exit: check each trading day until either threshold is hit or horizon (trading days) is reached
        exit_date = None
        exit_price = None
        exit_reason = None
        
        # Count trading days instead of calendar days
        # We'll exit after 'horizon' trading days have passed
        trading_days_count = 0
        
        # Check each trading day until we reach the horizon (trading days)
        days_ahead = 1
        while True:
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
            exit_row = df_sorted.iloc[exit_idx]
            day_open = exit_row[open_col]
            day_close = exit_row[close_col]
            
            # Increment trading day counter (each iteration is one trading day)
            trading_days_count += 1
            
            # Check if we've reached the horizon (trading days)
            # If so, exit at open price if stop loss and return threshold haven't been reached
            if trading_days_count >= horizon:
                # We've reached the horizon (trading days)
                # Check stop loss at open first
                if trade_stop_loss_price is not None:
                    day_low = exit_row[low_col] if low_col and low_col in exit_row.index else day_open
                    if day_open < trade_stop_loss_price:
                        # Stop loss hit at open
                        exit_price = day_open
                        exit_reason = "stop_loss"
                        break
                    elif day_low < trade_stop_loss_price:
                        # Stop loss hit during day
                        exit_price = trade_stop_loss_price
                        exit_reason = "stop_loss"
                        break
                    elif day_close < trade_stop_loss_price:
                        # Stop loss hit at close
                        exit_price = trade_stop_loss_price
                        exit_reason = "stop_loss"
                        break
                
                # Check return threshold at open
                open_return = (day_open / entry_price) - 1
                if return_threshold is not None and open_return >= return_threshold:
                    exit_price = day_open
                    exit_reason = "target_reached"
                    break
                
                # If neither stop loss nor target reached, exit at open (time limit)
                exit_price = day_open
                exit_reason = "time_limit"
                break
            
            # Before target exit date: check stop loss and return threshold normally
            # Check if stop-loss is hit (handle gap-downs correctly)
            if trade_stop_loss_price is not None:
                # Get the day's open and low prices
                day_low = exit_row[low_col] if low_col and low_col in exit_row.index else day_open
                
                # Priority 1: Check open price first (gap-down scenario)
                if day_open < trade_stop_loss_price:
                    # Stock gapped down below stop loss
                    # In reality, stop order triggers at market open and gets filled at open price (slippage)
                    exit_price = day_open
                    exit_reason = "stop_loss"
                    break
                
                # Priority 2: Check low price (intraday stop loss)
                elif day_low < trade_stop_loss_price:
                    # Stock traded below stop loss during the day
                    # Stop order gets filled at the stop loss price
                    exit_price = trade_stop_loss_price
                    exit_reason = "stop_loss"
                    break
                
                # Priority 3: Sanity check on close price
                elif day_close < trade_stop_loss_price:
                    # Close is below stop loss (shouldn't happen if low wasn't, but handle edge cases)
                    exit_price = trade_stop_loss_price
                    exit_reason = "stop_loss"
                    break
            
            # If no stop loss hit, use close price for return threshold check
            exit_price = day_close
            
            # Calculate return so far
            current_return = (exit_price / entry_price) - 1
            
            # Check if return threshold is reached
            if return_threshold is not None and current_return >= return_threshold:
                exit_reason = "target_reached"
                break
            
            # Move to next trading day
            days_ahead += 1
        
        if exit_price is None or pd.isna(exit_price):
            continue
        
        ret = (exit_price / entry_price) - 1
        pnl = ret * position_size
        holding_days = (exit_date - date).days if exit_date else horizon
        
        # Record this position's exit date to prevent overlapping trades
        if last_exit_date is None or exit_date > last_exit_date:
            last_exit_date = exit_date
        
        trade_dict = {
            "entry_date": date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "return": ret,
            "pnl": pnl,
            "holding_days": holding_days,
            "exit_reason": exit_reason
        }
        
        # Add stop_loss_pct if using adaptive stops
        if trade_stop_loss_pct is not None:
            trade_dict["stop_loss_pct"] = trade_stop_loss_pct
        
        # Add swing_atr metadata if available
        if trade_stop_metadata.get('used_swing_low') is not None:
            trade_dict["used_swing_low"] = trade_stop_metadata.get('used_swing_low', False)
        if 'swing_distance_pct' in trade_stop_metadata:
            trade_dict["swing_distance_pct"] = trade_stop_metadata.get('swing_distance_pct')
        if 'buffer_pct' in trade_stop_metadata:
            trade_dict["buffer_pct"] = trade_stop_metadata.get('buffer_pct')
        
        trades.append(trade_dict)
    
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
    missing_features = [f for f in features if f not in df.columns]
    
    # Log missing features for debugging
    missing_pct = (len(missing_features) / len(features)) * 100 if features else 0
    if missing_features:
        if missing_pct > 5:  # Log if more than 5% of features are missing
            print(f"Warning: {len(missing_features)} features missing from data ({missing_pct:.1f}%): {', '.join(missing_features[:10])}{'...' if len(missing_features) > 10 else ''}")
        else:
            print(f"Note: {len(missing_features)} features missing from data, filling with 0.0")
    
    # Require at least 50% of model features present (parquet may have only "enabled" subset)
    # Missing features are filled with 0.0; very low overlap may indicate wrong feature set
    min_overlap = 0.5
    if len(available_features) < len(features) * min_overlap:
        print(f"Error: Too many features missing ({len(missing_features)}/{len(features)} = {missing_pct:.1f}%). Need at least {min_overlap*100:.0f}% present. Returning no signals.")
        return pd.Series(False, index=df.index)
    
    if missing_features:
        print(f"Using {len(available_features)}/{len(features)} features; filling {len(missing_features)} missing with 0.0")
    
    X = df[available_features].copy()
    
    # Fill missing features (columns that don't exist) with 0.0
    for f in missing_features:
        X[f] = 0.0
    
    # Ensure we have all features in the correct order
    X = X[features]
    
    # CRITICAL: Handle NaN and infinity values before prediction
    # Many features (like sma200_ratio, sma200_slope) have NaN for first 200 days
    # Replace infinities with NaN first, then fill all NaNs with 0.0
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Count NaN values before filling (for debugging)
    nan_counts = X.isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        # Log warning if significant NaN values found (but still proceed)
        nan_pct = (total_nans / (len(X) * len(X.columns))) * 100
        if nan_pct > 10:  # More than 10% NaN values
            print(f"Warning: {nan_pct:.1f}% NaN values in features before prediction. Filling with 0.0.")
    
    X = X.fillna(0.0)  # Fill NaN values with 0.0 (XGBoost can handle NaN, but filling is safer and consistent)
    
    # Note: XGBoost doesn't require feature scaling, so no scaling is applied
    
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
    
    # Maximum drawdown (peak-to-trough from cumulative P&L curve)
    # Sort trades by exit_date to match the cumulative P&L chart
    trades_sorted = trades.copy()
    if "exit_date" in trades_sorted.columns:
        trades_sorted["exit_date_dt"] = pd.to_datetime(trades_sorted["exit_date"], errors='coerce')
        trades_sorted = trades_sorted.sort_values("exit_date_dt")
    elif "entry_date" in trades_sorted.columns:
        trades_sorted["entry_date_dt"] = pd.to_datetime(trades_sorted["entry_date"], errors='coerce')
        trades_sorted = trades_sorted.sort_values("entry_date_dt")
    # If no dates, use existing order (assume already chronological)
    
    # Calculate cumulative P&L in chronological order (matching the chart)
    equity = trades_sorted["pnl"].cumsum()
    # Calculate running peak (highest cumulative P&L reached so far)
    running_max = equity.cummax()
    # Calculate drawdown at each point (peak - current value)
    drawdown = running_max - equity
    # Maximum drawdown is the largest peak-to-trough decline
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
    stop_loss_config: StopLossConfig = None,
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
            
            # Determine stop-loss configuration
            # Priority: stop_loss_config > stop_loss (legacy) > 2:1 risk-reward default
            ticker_stop_loss_config = stop_loss_config
            ticker_stop_loss = stop_loss
            
            if ticker_stop_loss_config is None:
                # Create config from legacy parameters or defaults
                if ticker_stop_loss is None and return_threshold is not None:
                    # Default 2:1 risk-reward ratio
                    ticker_stop_loss_config = StopLossConfig(
                        mode="constant",
                        constant_stop_loss_pct=abs(return_threshold) / 2
                    )
                elif ticker_stop_loss is not None:
                    # Use legacy stop_loss parameter
                    ticker_stop_loss_config = StopLossConfig(
                        mode="constant",
                        constant_stop_loss_pct=abs(ticker_stop_loss)
                    )
            
            # Run backtest for this ticker
            trades = backtest_strategy(
                df,
                signal_col="entry_signal",
                horizon=horizon,
                position_size=position_size,
                return_threshold=return_threshold,
                stop_loss=ticker_stop_loss,  # Legacy parameter (may be None)
                stop_loss_config=ticker_stop_loss_config  # New parameter
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
        help="Stop-loss threshold as decimal (e.g., -0.075 for -7.5%%). If not specified and return-threshold is provided, uses 2:1 risk-reward (return_threshold / 2). DEPRECATED: Use --stop-loss-mode and related args for adaptive stops."
    )
    parser.add_argument(
        "--stop-loss-mode",
        type=str,
        choices=["constant", "adaptive_atr", "swing_atr"],
        default=None,
        help="Stop-loss mode: 'constant' (fixed), 'adaptive_atr' (ATR-based), or 'swing_atr' (swing low + ATR buffer). Default: 'constant' if --stop-loss is provided, otherwise uses 2:1 risk-reward."
    )
    parser.add_argument(
        "--atr-stop-k",
        type=float,
        default=1.8,
        help="ATR multiplier for adaptive stops (default: 1.8). Only used when --stop-loss-mode=adaptive_atr."
    )
    parser.add_argument(
        "--atr-stop-min-pct",
        type=float,
        default=0.04,
        help="Minimum stop distance for adaptive stops (default: 0.04 = 4%%). Only used when --stop-loss-mode=adaptive_atr."
    )
    parser.add_argument(
        "--atr-stop-max-pct",
        type=float,
        default=0.10,
        help="Maximum stop distance for adaptive stops (default: 0.10 = 10%%). Only used when --stop-loss-mode=adaptive_atr or swing_atr."
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
        # Resolve model path relative to project root if it's a relative path
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        model_path = model_path.resolve()
        
        print(f"Loading model from {model_path}...")
        model, features, scaler, features_to_scale, _ = load_model(model_path)
        if model is None:
            print("ERROR: Model not found. Cannot run model strategy.")
            return
        print(f"Model loaded. Using {len(features)} features (trained subset for prediction).")
        # Note: XGBoost doesn't require feature scaling, so scaler is not used
    
    # Load tickers
    tickers_df = pd.read_csv(args.tickers_file, header=None)
    tickers = tickers_df.iloc[:, 0].astype(str).tolist()
    print(f"Data directory: {args.data_dir}")
    print(f"Loaded {len(tickers)} tickers")
    
    # Create stop-loss configuration
    stop_loss_config = create_stop_loss_config_from_args(
        stop_loss_mode=getattr(args, 'stop_loss_mode', None),
        constant_stop_loss_pct=None,  # Will be determined from stop_loss or return_threshold
        atr_stop_k=getattr(args, 'atr_stop_k', 1.8),
        atr_stop_min_pct=getattr(args, 'atr_stop_min_pct', 0.04),
        atr_stop_max_pct=getattr(args, 'atr_stop_max_pct', 0.10),
        swing_lookback_days=getattr(args, 'swing_lookback_days', 10),
        swing_atr_buffer_k=getattr(args, 'swing_atr_buffer_k', 0.75),
        return_threshold=args.return_threshold,
        legacy_stop_loss=getattr(args, 'stop_loss', None)
    )
    
    # Legacy stop_loss_value for backward compatibility
    stop_loss_value = getattr(args, 'stop_loss', None)
    if stop_loss_value is None and args.return_threshold is not None:
        stop_loss_value = -abs(args.return_threshold) / 2  # 2:1 risk-reward ratio
    
    print(f"\nRunning backtest with:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Horizon: {args.horizon} days")
    print(f"  Return threshold: {args.return_threshold:.2%}")
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
        stop_loss=stop_loss_value,  # Legacy parameter
        stop_loss_config=stop_loss_config,  # New parameter
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
    print(f"Stop-Loss Mode: {stop_loss_config.mode}")
    if stop_loss_config.mode == "constant":
        print(f"Stop-Loss: {stop_loss_config.constant_stop_loss_pct:.2%}")
    elif stop_loss_config.mode == "adaptive_atr":
        print(f"ATR Stop K: {stop_loss_config.atr_stop_k}")
        print(f"ATR Stop Range: {stop_loss_config.atr_stop_min_pct:.2%} - {stop_loss_config.atr_stop_max_pct:.2%}")
    elif stop_loss_config.mode == "swing_atr":
        print(f"Swing Lookback Days: {stop_loss_config.swing_lookback_days}")
        print(f"Swing ATR Buffer K: {stop_loss_config.swing_atr_buffer_k}")
        print(f"Stop Range: {stop_loss_config.atr_stop_min_pct:.2%} - {stop_loss_config.atr_stop_max_pct:.2%}")
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
    
    # Show adaptive stop-loss statistics if applicable
    if stop_loss_config.mode in ["adaptive_atr", "swing_atr"] and not trades.empty and "stop_loss_pct" in trades.columns:
        stop_stats = summarize_adaptive_stops(trades, mode=stop_loss_config.mode)
        mode_name = "Adaptive ATR" if stop_loss_config.mode == "adaptive_atr" else "Swing ATR"
        print(f"\n  {mode_name} Stop-Loss Statistics:")
        if stop_stats["avg_stop_pct"] is not None:
            print(f"    Average Stop Distance: {stop_stats['avg_stop_pct']:.2%}")
            print(f"    Minimum Stop Distance: {stop_stats['min_stop_pct']:.2%}")
            print(f"    Maximum Stop Distance: {stop_stats['max_stop_pct']:.2%}")
            if stop_stats["stop_distribution"]:
                print(f"    Stop Distance Distribution:")
                for label, stats in stop_stats["stop_distribution"].items():
                    print(f"      {label}: {stats['count']:,} trades ({stats['pct']:.1f}%)")
        
        # Show swing_atr-specific statistics
        if stop_loss_config.mode == "swing_atr" and "swing_atr_stats" in stop_stats:
            swing_stats = stop_stats["swing_atr_stats"]
            print(f"\n    Swing ATR Details:")
            print(f"      Trades Using Swing Low: {swing_stats.get('trades_using_swing_low', 0):,} ({swing_stats.get('swing_low_usage_pct', 0):.1f}%)")
            print(f"      Trades Using Fallback: {swing_stats.get('trades_using_fallback', 0):,}")
            if 'avg_swing_distance_pct' in swing_stats:
                print(f"      Average Swing Distance: {swing_stats['avg_swing_distance_pct']:.2%}")
                print(f"      Swing Distance Range: {swing_stats.get('min_swing_distance_pct', 0):.2%} - {swing_stats.get('max_swing_distance_pct', 0):.2%}")
            if 'avg_buffer_pct' in swing_stats:
                print(f"      Average ATR Buffer: {swing_stats['avg_buffer_pct']:.2%}")
    
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
    
    # Save trades if requested (save even if empty to preserve output file)
    if args.output:
        if not trades.empty:
            trades.to_csv(args.output)
            print(f"\nTrades saved to {args.output}")
        else:
            # Save empty DataFrame with expected columns for consistency
            empty_trades = pd.DataFrame(columns=[
                'entry_date', 'entry_price', 'exit_date', 'exit_price',
                'return', 'pnl', 'holding_days', 'exit_reason'
            ])
            empty_trades.to_csv(args.output, index=False)
            print(f"\nEmpty trades file saved to {args.output} (0 trades generated)")


if __name__ == "__main__":
    main()

