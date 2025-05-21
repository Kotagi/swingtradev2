#!/usr/bin/env python3
"""
backtest.py

Run backtests on labeled feature Parquet data for:
  1. Oracle strategy (perfect hindsight on label)
  2. RSI oversold strategy (RSI < 30 entry)

Configurable parameters include position size and holding horizon.
Outputs summary metrics for each strategy across the S&P 500 universe.

Now reads feature files in Parquet format from data/features_labeled/.
"""

import pandas as pd
from pathlib import Path
from typing import List, Tuple

# —— CONFIGURATION —— #
PROJECT_ROOT = Path.cwd()
DATA_DIR      = PROJECT_ROOT / "data" / "features_labeled"
TICKERS_CSV   = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
POSITION_SIZE = 1000.0           # dollars per trade
HORIZON       = 5                # holding period in days
LABEL_COL     = f"label_{HORIZON}d"


def load_universe(tickers_csv: Path = TICKERS_CSV) -> List[str]:
    """
    Load the list of tickers from a CSV file.

    Args:
        tickers_csv: Path to a CSV with one ticker symbol per line.

    Returns:
        List of ticker symbols as strings.
    """
    # Read with no header, first column is the ticker
    df = pd.read_csv(tickers_csv, header=None)
    return df.iloc[:, 0].astype(str).tolist()


def backtest_signals(
    df: pd.DataFrame,
    signal_col: str,
    horizon: int = HORIZON,
    position_size: float = POSITION_SIZE
) -> pd.DataFrame:
    """
    Backtest a boolean entry signal over a fixed holding period.

    For each True in signal_col, enter at next-day open and exit at close after `horizon` days.

    Args:
        df: DataFrame indexed by date, containing 'open', 'close', and signal_col.
        signal_col: Name of boolean column indicating entry.
        horizon: Holding period in days.
        position_size: Dollar amount per trade.

    Returns:
        DataFrame of trades, indexed by entry date, with columns:
          - 'entry': entry price
          - 'exit' : exit price
          - 'ret'  : return fraction (exit/entry - 1)
          - 'pnl'  : profit/loss in dollars
    """
    # Determine open/close column names (case-insensitive)
    open_col  = "Open"  if "Open"  in df.columns else "open"
    close_col = "Close" if "Close" in df.columns else "close"

    # Sort the DataFrame by date to ensure chronological order
    df_sorted = df.sort_index().copy()

    # Precompute exit prices: shift close price backwards by `horizon` rows
    df_sorted["exit_price"] = df_sorted[close_col].shift(-horizon)

    trades = []
    # Iterate over each date/row
    for date, row in df_sorted.iterrows():
        # Only enter if signal is True and exit_price is not NaN
        if row.get(signal_col) and pd.notna(row["exit_price"]):
            entry_price = row[open_col]
            exit_price  = row["exit_price"]
            ret         = exit_price / entry_price - 1
            pnl         = ret * position_size
            trades.append({
                "date": date,
                "entry": entry_price,
                "exit": exit_price,
                "ret": ret,
                "pnl": pnl
            })

    # If no trades were generated, return empty DataFrame
    if not trades:
        return pd.DataFrame()

    # Build a DataFrame of trades and set 'date' as the index
    trades_df = pd.DataFrame(trades).set_index("date")
    return trades_df


def aggregate_results(trades: pd.DataFrame) -> pd.Series:
    """
    Compute summary performance metrics from a series of trades.

    Args:
        trades: DataFrame of trades as returned by backtest_signals.

    Returns:
        Series with metrics:
          - n_trades: total number of trades
          - hit_rate: fraction of trades with ret > 0
          - avg_ret: average return per trade
          - total_pnl: total P&L across all trades
          - avg_pnl: average P&L per trade
          - max_dd: maximum drawdown on cumulative P&L curve
    """
    if trades.empty:
        # Return empty Series if no trades
        return pd.Series(dtype=float)

    n_trades = len(trades)
    hit_rate = (trades["ret"] > 0).mean()
    avg_ret  = trades["ret"].mean()
    total_pnl = trades["pnl"].sum()
    avg_pnl  = trades["pnl"].mean()

    # Compute equity curve and max drawdown
    equity = trades["pnl"].cumsum()
    max_dd = (equity.cummax() - equity).max()

    return pd.Series({
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "avg_ret": avg_ret,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_dd": max_dd
    })


def main() -> None:
    """
    Main entry point: executes backtests for oracle and RSI strategies,
    then prints summary metrics for each.
    """
    # Load the universe of tickers
    universe = load_universe()

    oracle_trades: List[pd.DataFrame] = []
    rsi_trades:    List[pd.DataFrame] = []

    # Process each ticker
    for ticker in universe:
        # Construct path to the Parquet file
        path = DATA_DIR / f"{ticker}.parquet"
        if not path.exists():
            # Skip tickers without data
            continue

        # Load labeled feature data from Parquet
        df = pd.read_parquet(path).rename_axis("date").sort_index()

        # Oracle strategy: use the true label as entry signal
        if LABEL_COL in df.columns:
            df_o = df.copy()
            df_o["oracle_signal"] = df_o[LABEL_COL] == 1
            trades_o = backtest_signals(df_o, "oracle_signal")
            if not trades_o.empty:
                oracle_trades.append(trades_o)

        # RSI oversold strategy: entry when RSI < 30
        if "rsi" in df.columns:
            df_r = df.copy()
            df_r["rsi_signal"] = df_r["rsi"] < 30
            trades_r = backtest_signals(df_r, "rsi_signal")
            if not trades_r.empty:
                rsi_trades.append(trades_r)

    # Concatenate all trades for each strategy
    oracle_df = pd.concat(oracle_trades) if oracle_trades else pd.DataFrame()
    rsi_df    = pd.concat(rsi_trades)    if rsi_trades    else pd.DataFrame()

    # Compute summary metrics
    oracle_summary = aggregate_results(oracle_df) if not oracle_df.empty else None
    rsi_summary    = aggregate_results(rsi_df)    if not rsi_df.empty    else None

    # Print Oracle summary
    print("\n=== ORACLE BACKTEST SUMMARY ===")
    if oracle_summary is not None:
        print(f"Trades:        {int(oracle_summary['n_trades'])}")
        print(f"Hit rate:      {oracle_summary['hit_rate']:.2%}")
        print(f"Avg return:    {oracle_summary['avg_ret']:.2%}")
        print(f"Total P&L:     ${oracle_summary['total_pnl']:,.2f}")
        print(f"Avg P&L/trade: ${oracle_summary['avg_pnl']:,.2f}")
        print(f"Max drawdown:  ${oracle_summary['max_dd']:,.2f}")
    else:
        print("No oracle trades to report.")

    # Print RSI summary
    print("\n=== RSI OVERSOLD BACKTEST SUMMARY ===")
    if rsi_summary is not None:
        print(f"Trades:        {int(rsi_summary['n_trades'])}")
        print(f"Hit rate:      {rsi_summary['hit_rate']:.2%}")
        print(f"Avg return:    {rsi_summary['avg_ret']:.2%}")
        print(f"Total P&L:     ${rsi_summary['total_pnl']:,.2f}")
        print(f"Avg P&L/trade: ${rsi_summary['avg_pnl']:,.2f}")
        print(f"Max drawdown:  ${rsi_summary['max_dd']:,.2f}")
    else:
        print("No RSI oversold trades to report.")


if __name__ == "__main__":
    main()
