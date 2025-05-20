#!/usr/bin/env python3
"""
backtest.py

Backtester with RSI oversold test.
"""

import pandas as pd
from pathlib import Path
from typing import List

# —— CONFIGURATION —— #
PROJECT_ROOT  = Path.cwd()
DATA_DIR      = PROJECT_ROOT / "data" / "features_labeled"
TICKERS_CSV   = PROJECT_ROOT / "data" / "tickers" / "sp500_tickers.csv"
POSITION_SIZE = 1000.0  # dollars per trade
HORIZON       = 5       # days to hold
LABEL_COL     = f"label_{HORIZON}d"

# —— UNIVERSE LOADING —— #
def load_universe() -> List[str]:
    df = pd.read_csv(TICKERS_CSV, header=None)
    return df.iloc[:, 0].astype(str).tolist()

# —— SIGNAL BACKTESTER —— #
def backtest_signals(df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    """
    Given a DataFrame and a boolean signal column, return a trades DataFrame with:
      - entry (next-day open)
      - exit  (close after HORIZON days)
      - ret   (exit/entry - 1)
      - pnl   (ret * POSITION_SIZE)
    """
    open_col  = "Open"  if "Open" in df.columns  else "open"
    close_col = "Close" if "Close" in df.columns else "close"

    df = df.sort_index().copy()
    df["exit_price"] = df[close_col].shift(-HORIZON)

    trades = []
    for dt, row in df.iterrows():
        if row.get(signal_col) and not pd.isna(row["exit_price"]):
            entry_price = row[open_col]
            exit_price  = row["exit_price"]
            ret         = exit_price / entry_price - 1
            pnl         = ret * POSITION_SIZE
            trades.append({
                "date":  dt,
                "entry": entry_price,
                "exit":  exit_price,
                "ret":   ret,
                "pnl":   pnl
            })

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).set_index("date")

# —— METRICS AGGREGATOR —— #
def aggregate_results(trades: pd.DataFrame) -> pd.Series:
    """
    Compute summary metrics for a set of trades.
    """
    if trades.empty:
        return pd.Series(dtype=float)

    n_trades      = len(trades)
    hit_rate      = (trades["ret"] > 0).mean()
    avg_ret       = trades["ret"].mean()
    total_pnl     = trades["pnl"].sum()
    avg_pnl       = trades["pnl"].mean()

    equity_curve  = trades.sort_index()["pnl"].cumsum()
    max_drawdown  = (equity_curve.cummax() - equity_curve).max()

    return pd.Series({
        "n_trades":  n_trades,
        "hit_rate":  hit_rate,
        "avg_ret":   avg_ret,
        "total_pnl": total_pnl,
        "avg_pnl":   avg_pnl,
        "max_dd":    max_drawdown
    })

# —— MAIN —— #
def main():
    universe      = load_universe()
    oracle_trades = []
    rsi_trades    = []

    for ticker in universe:
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            continue

        df = (
            pd.read_csv(path, index_col=0, parse_dates=True)
              .sort_index()
              .rename_axis("date")
        )

        # Oracle (perfect foresight) signal
        if LABEL_COL in df.columns:
            df_o = df.copy()
            df_o["oracle_signal"] = df_o[LABEL_COL] == 1
            t_o = backtest_signals(df_o, "oracle_signal")
            if not t_o.empty:
                oracle_trades.append(t_o)

        # RSI oversold signal (RSI < 30)
        if "rsi" in df.columns:
            df_r = df.copy()
            df_r["rsi_signal"] = df_r["rsi"] < 30
            t_r = backtest_signals(df_r, "rsi_signal")
            if not t_r.empty:
                rsi_trades.append(t_r)

    oracle_df = pd.concat(oracle_trades) if oracle_trades else pd.DataFrame()
    rsi_df    = pd.concat(rsi_trades)    if rsi_trades    else pd.DataFrame()

    oracle_summary = aggregate_results(oracle_df) if not oracle_df.empty else None
    rsi_summary    = aggregate_results(rsi_df)    if not rsi_df.empty    else None

    print("\n=== ORACLE BACKTEST SUMMARY ===")
    if oracle_summary is not None:
        print(f"Trades:        {oracle_summary['n_trades']}")
        print(f"Hit rate:      {oracle_summary['hit_rate']:.2%}")
        print(f"Avg return:    {oracle_summary['avg_ret']:.2%}")
        print(f"Total P&L:     ${oracle_summary['total_pnl']:,.2f}")
        print(f"Avg P&L/trade: ${oracle_summary['avg_pnl']:,.2f}")
        print(f"Max drawdown:  ${oracle_summary['max_dd']:,.2f}")
    else:
        print("No oracle trades to report.")

    print("\n=== RSI OVERSOLD BACKTEST SUMMARY ===")
    if rsi_summary is not None:
        print(f"Trades:        {rsi_summary['n_trades']}")
        print(f"Hit rate:      {rsi_summary['hit_rate']:.2%}")
        print(f"Avg return:    {rsi_summary['avg_ret']:.2%}")
        print(f"Total P&L:     ${rsi_summary['total_pnl']:,.2f}")
        print(f"Avg P&L/trade: ${rsi_summary['avg_pnl']:,.2f}")
        print(f"Max drawdown:  ${rsi_summary['max_dd']:,.2f}")
    else:
        print("No RSI oversold trades to report.")

if __name__ == "__main__":
    main()

