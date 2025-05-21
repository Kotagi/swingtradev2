# src/backtest.py

#!/usr/bin/env python3
"""
backtest.py

Run backtests on labeled feature data for:
  1. Oracle strategy (perfect hindsight on label)
  2. RSI oversold strategy (RSI < 30 entry)

Configurable parameters include position size and holding horizon.
Outputs summary metrics for each strategy across the S&P 500 universe.
"""

import pandas as pd
from pathlib import Path
from typing import List, Union

# —— CONFIGURATION —— #
PROJECT_ROOT: Path = Path.cwd()
DATA_DIR: Path = PROJECT_ROOT / "data" / "features_labeled"
TICKERS_CSV: Path = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
POSITION_SIZE: float = 1000.0  # dollars per trade
HORIZON: int = 5               # holding period in days
LABEL_COL: str = f"label_{HORIZON}d"


def load_universe(tickers_csv: Path = TICKERS_CSV) -> List[str]:
    """
    Load the list of tickers from a CSV file.

    Args:
        tickers_csv: Path to CSV file with one ticker symbol per line.

    Returns:
        List of ticker symbols as strings.
    """
    df = pd.read_csv(tickers_csv, header=None)
    # First column contains ticker symbols
    return df.iloc[:, 0].astype(str).tolist()


def backtest_signals(
    df: pd.DataFrame,
    signal_col: str,
    horizon: int = HORIZON,
    position_size: float = POSITION_SIZE
) -> pd.DataFrame:
    """
    Backtest a boolean entry signal over a fixed holding period.

    For each True signal row, enter at next-day open and exit at close after `horizon` days.

    Args:
        df: DataFrame indexed by date, with columns including open/close prices and the signal.
        signal_col: Name of the boolean column indicating entry occurences.
        horizon: Holding period in trading days.
        position_size: Dollar amount allocated per trade.

    Returns:
        DataFrame of individual trades, indexed by entry date, with columns:
          - 'entry': entry price
          - 'exit': exit price
          - 'ret': return fraction (exit/entry - 1)
          - 'pnl': profit/loss in dollars
    """
    # Determine column names for open and close (case-insensitive)
    open_col = "Open" if "Open" in df.columns else "open"
    close_col = "Close" if "Close" in df.columns else "close"

    # Ensure sorted by date
    df_sorted = df.sort_index().copy()
    # Precompute exit price at horizon
    df_sorted["exit_price"] = df_sorted[close_col].shift(-horizon)

    trades = []
    for date, row in df_sorted.iterrows():
        # Trigger trade if signal True and exit_price is available
        if row.get(signal_col) and not pd.isna(row["exit_price"]):
            entry_price = row[open_col]
            exit_price = row["exit_price"]
            ret = exit_price / entry_price - 1
            pnl = ret * position_size
            trades.append({
                "date": date,
                "entry": entry_price,
                "exit": exit_price,
                "ret": ret,
                "pnl": pnl
            })

    if not trades:
        # No trades generated
        return pd.DataFrame()

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
          - hit_rate: fraction of trades with positive return
          - avg_ret: average return per trade
          - total_pnl: total profit/loss across all trades
          - avg_pnl: average P&L per trade
          - max_dd: maximum drawdown on cumulative P&L curve
    """
    if trades.empty:
        return pd.Series(dtype=float)

    n_trades = len(trades)
    hit_rate = (trades["ret"] > 0).mean()
    avg_ret = trades["ret"].mean()
    total_pnl = trades["pnl"].sum()
    avg_pnl = trades["pnl"].mean()

    # Cumulative P&L for drawdown calculation
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
    then prints summary metrics.
    """
    # Load ticker universe
    universe = load_universe()

    # Containers for trade DataFrames
    oracle_trades = []
    rsi_trades = []

    for ticker in universe:
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            continue

        # Load labeled feature CSV
        df = (
            pd.read_csv(path, index_col=0, parse_dates=True)
              .rename_axis("date")
              .sort_index()
        )

        # Oracle strategy: use the true label as entry signal
        if LABEL_COL in df.columns:
            df_o = df.copy()
            df_o["oracle_signal"] = df_o[LABEL_COL] == 1
            trades_o = backtest_signals(df_o, "oracle_signal")
            if not trades_o.empty:
                oracle_trades.append(trades_o)

        # RSI oversold strategy: RSI < 30 entry signal
        if "rsi" in df.columns:
            df_r = df.copy()
            df_r["rsi_signal"] = df_r["rsi"] < 30
            trades_r = backtest_signals(df_r, "rsi_signal")
            if not trades_r.empty:
                rsi_trades.append(trades_r)

    # Concatenate trades for aggregate stats
    oracle_df = pd.concat(oracle_trades) if oracle_trades else pd.DataFrame()
    rsi_df = pd.concat(rsi_trades) if rsi_trades else pd.DataFrame()

    # Compute summaries
    oracle_summary = aggregate_results(oracle_df) if not oracle_df.empty else None
    rsi_summary = aggregate_results(rsi_df) if not rsi_df.empty else None

    # Print results
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

