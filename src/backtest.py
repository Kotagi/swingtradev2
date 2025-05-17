#!/usr/bin/env python
import os
import argparse
import logging
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def load_data(ticker, clean_dir):
    """
    Load a cleaned CSV for the given ticker.
    Assumes the first column is the date index.
    """
    path = os.path.join(clean_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean file not found for {ticker}: {path}")
    # Read with first column as datetime index
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def compute_indicators(df, atr_window=14, mom_window=5):
    """
    Compute ATR and momentum on the dataframe.
    """
    # True Range
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=atr_window, min_periods=1).mean()
    # Momentum: percent change over mom_window
    df["MOM"] = df["Close"].pct_change(periods=mom_window)
    return df

def run_backtest(
    tickers,
    clean_dir,
    sectors_file,
    momentum_threshold,
    stop_loss_atr_mult,
    time_exit_days,
    initial_capital,
    slippage,
    commission_per_trade,
    commission_per_share,
    risk_percentage,
    max_positions,
    max_sector_exposure,
):
    # load sector map
    sectors_df = pd.read_csv(sectors_file)
    # allow lowercase column names
    sectors_df.columns = [c.strip().lower() for c in sectors_df.columns]
    if "ticker" not in sectors_df.columns or "sector" not in sectors_df.columns:
        raise KeyError("sectors_file must have columns 'ticker' and 'sector'")
    sector_map = dict(zip(sectors_df["ticker"], sectors_df["sector"]))

    # preload data & indicators
    data_cache = {}
    for t in tickers:
        df = load_data(t, clean_dir)
        df = compute_indicators(df)
        data_cache[t] = df

    # build complete date list
    all_dates = sorted({date for df in data_cache.values() for date in df.index})
    logging.info(f"Backtest will run over {len(all_dates)} trading days.")

    equity = initial_capital
    positions = []  # list of dicts: {ticker, entry_date, entry_price, quantity, stop_price, sector}
    trade_records = []

    for i, current_date in enumerate(all_dates, start=1):
        # console feedback
        logging.info(f"[{i}/{len(all_dates)}] {current_date.date()}")

        # 1) Check exits first
        for pos in positions[:]:
            df = data_cache[pos["ticker"]]
            if current_date not in df.index:
                continue
            price = df.at[current_date, "Close"]
            # stop-loss?
            if price <= pos["stop_price"]:
                exit_price = pos["stop_price"]
            # time-exit?
            elif (current_date - pos["entry_date"]).days >= time_exit_days:
                exit_price = price
            else:
                continue

            # record exit
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            # apply slippage & commission
            pnl -= exit_price * pos["quantity"] * slippage
            pnl -= commission_per_trade + pos["quantity"] * commission_per_share

            equity += (pos["entry_price"] * pos["quantity"] + pnl)
            trade_records.append({
                "ticker": pos["ticker"],
                "entry_date": pos["entry_date"],
                "exit_date": current_date,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "quantity": pos["quantity"],
                "sector": pos["sector"],
                "PnL": pnl,
                "equity": equity,
            })
            positions.remove(pos)

        # 2) Check entries if under max_positions
        open_sectors = [pos["sector"] for pos in positions]
        sector_counts = {s: open_sectors.count(s) for s in set(open_sectors)}

        for t in tickers:
            if len(positions) >= max_positions:
                break
            sector = sector_map[t]
            if sector_counts.get(sector, 0) / max_positions > max_sector_exposure:
                continue

            df = data_cache[t]
            if current_date not in df.index:
                continue

            # momentum signal
            mom = df.at[current_date, "MOM"]
            if mom < momentum_threshold:
                continue

            entry_price = df.at[current_date, "Close"]
            atr = df.at[current_date, "ATR"]
            stop_price = entry_price - stop_loss_atr_mult * atr

            # position sizing
            risk_per_share = entry_price - stop_price
            dollar_risk = equity * risk_percentage
            quantity = int(dollar_risk // risk_per_share)
            if quantity <= 0:
                continue

            # enter
            equity -= entry_price * quantity
            positions.append({
                "ticker": t,
                "entry_date": current_date,
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_price": stop_price,
                "sector": sector,
            })
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # flush remaining positions at last date
    final_date = all_dates[-1]
    for pos in positions:
        df = data_cache[pos["ticker"]]
        if final_date not in df.index:
            continue
        exit_price = df.at[final_date, "Close"]
        pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
        pnl -= exit_price * pos["quantity"] * slippage
        pnl -= commission_per_trade + pos["quantity"] * commission_per_share
        equity += (pos["entry_price"] * pos["quantity"] + pnl)
        trade_records.append({
            "ticker": pos["ticker"],
            "entry_date": pos["entry_date"],
            "exit_date": final_date,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "quantity": pos["quantity"],
            "sector": pos["sector"],
            "PnL": pnl,
            "equity": equity,
        })

    trades_df = pd.DataFrame(trade_records)
    return trades_df

def main():
    p = argparse.ArgumentParser("backtest.py")
    p.add_argument("-t", "--tickers", nargs="+", required=True)
    p.add_argument("--clean-dir", default="data/clean")
    p.add_argument("--sectors-file", required=True)
    p.add_argument("--momentum-threshold", type=float, default=0.05)
    p.add_argument("--stop-loss-atr-mult", type=float, default=2.0)
    p.add_argument("--time-exit-days", type=int, default=5)
    p.add_argument("-i", "--initial-capital", type=float, default=100_000)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--commission-per-trade", type=float, default=0.0)
    p.add_argument("--commission-per-share", type=float, default=0.0)
    p.add_argument("--risk-percentage", type=float, default=0.01)
    p.add_argument("--max-positions", type=int, default=8)
    p.add_argument("--max-sector-exposure", type=float, default=0.25)
    p.add_argument("-o", "--output", required=True,
                   help="Where to write trade_log.csv")
    args = p.parse_args()

    trades = run_backtest(
        tickers=args.tickers,
        clean_dir=args.clean_dir,
        sectors_file=args.sectors_file,
        momentum_threshold=args.momentum_threshold,
        stop_loss_atr_mult=args.stop_loss_atr_mult,
        time_exit_days=args.time_exit_days,
        initial_capital=args.initial_capital,
        slippage=args.slippage,
        commission_per_trade=args.commission_per_trade,
        commission_per_share=args.commission_per_share,
        risk_percentage=args.risk_percentage,
        max_positions=args.max_positions,
        max_sector_exposure=args.max_sector_exposure,
    )

    # ensure output folder exists
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    trades.to_csv(args.output, index=False)
    logging.info(f"{len(trades)} trades saved to {args.output}")

if __name__ == "__main__":
    main()
