#!/usr/bin/env python3
import argparse
import logging
import os
import pandas as pd
import numpy as np

def load_data(ticker, clean_dir):
    path = os.path.join(clean_dir, f"{ticker}.csv")
    # parse first column as the date index
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # lowercase column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def compute_indicators(df, atr_window=14, momentum_window=5):
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=1).mean().rename('atr')
    momentum = close.pct_change(momentum_window).rename('momentum')
    return pd.concat([df, atr, momentum], axis=1)

def run_backtest(
    tickers,
    clean_dir,
    sectors_file,
    momentum_threshold,
    stop_loss_atr_mult,
    time_exit_days,
    initial_capital,
    risk_percentage,
    slippage,
    commission_per_trade,
    commission_per_share,
    max_positions=8,
    max_sector_exposure=0.25,
    output=None,                # <-- now optional
):
    # load sectors (lowercase headers)
    sectors_df = pd.read_csv(sectors_file)
    sectors_df.columns = [c.strip().lower() for c in sectors_df.columns]
    sector_map = dict(zip(sectors_df['ticker'], sectors_df['sector']))

    logging.info("Caching data in memory…")
    data = {}
    for t in tickers:
        df = load_data(t, clean_dir)
        data[t] = compute_indicators(df)

    all_dates = sorted({d for df in data.values() for d in df.index})
    equity = initial_capital
    sector_counts = {s:0 for s in set(sector_map.values())}
    positions = []
    trades = []

    for i, current_date in enumerate(all_dates, 1):
        logging.info(f"[{i}/{len(all_dates)}] {current_date.date()}")

        # EXIT LOGIC
        new_positions = []
        for pos in positions:
            df = data[pos['ticker']]
            if current_date not in df.index:
                new_positions.append(pos)
                continue

            days = (current_date - pos['entry_date']).days
            price = df.at[current_date, 'close']
            exit_price = None

            if price <= pos['stop_price']:
                exit_price = pos['stop_price']
            elif days >= time_exit_days:
                exit_price = price

            if exit_price is not None:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'],
                    'exit_date': current_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                })
                equity += pnl
                sector_counts[pos['sector']] -= 1
            else:
                new_positions.append(pos)

        positions = new_positions

        # ENTRY LOGIC
        if len(positions) < max_positions:
            for t in tickers:
                if len(positions) >= max_positions:
                    break
                sector = sector_map.get(t)
                if sector and sector_counts[sector] >= max_sector_exposure * max_positions:
                    continue

                df = data[t]
                if current_date not in df.index:
                    continue
                row = df.loc[current_date]
                if row['momentum'] <= momentum_threshold:
                    continue

                atr = row['atr'] * stop_loss_atr_mult
                stop_price = row['close'] - atr
                dollar_risk = risk_percentage * equity
                risk_per_share = row['close'] - stop_price
                if risk_per_share <= 0:
                    continue

                qty = int(dollar_risk // risk_per_share)
                if qty <= 0:
                    continue

                buy_price = row['close'] * (1 + slippage)
                commission = commission_per_trade + commission_per_share * qty

                positions.append({
                    'ticker': t,
                    'entry_date': current_date,
                    'entry_price': buy_price,
                    'shares': qty,
                    'stop_price': stop_price,
                    'sector': sector
                })
                sector_counts[sector] += 1
                equity -= buy_price * qty + commission

    trades_df = pd.DataFrame(trades)

    # only write if user supplied an output path
    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        trades_df.to_csv(output, index=False)
        logging.info(f"{len(trades_df)} trades saved to {output}")

    return trades_df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-t','--tickers', nargs='+', required=True)
    p.add_argument('--clean-dir', default='data/clean')
    p.add_argument('--sectors-file', required=True)
    p.add_argument('--momentum-threshold', type=float, default=0.02)
    p.add_argument('--stop-loss-atr-mult', type=float, default=2.0)
    p.add_argument('--time-exit-days', type=int, default=5)
    p.add_argument('-i','--initial-capital', type=float, default=100_000)
    p.add_argument('--risk-percentage', type=float, default=0.01)
    p.add_argument('--slippage', type=float, default=0.0005)
    p.add_argument('--commission-per-trade', type=float, default=0.0)
    p.add_argument('--commission-per-share', type=float, default=0.0)
    p.add_argument('--max-positions', type=int, default=8)
    p.add_argument('--max-sector-exposure', type=float, default=0.25)
    p.add_argument('-o','--output', help="Where to write trade log CSV")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    run_backtest(
        tickers=args.tickers,
        clean_dir=args.clean_dir,
        sectors_file=args.sectors_file,
        momentum_threshold=args.momentum_threshold,
        stop_loss_atr_mult=args.stop_loss_atr_mult,
        time_exit_days=args.time_exit_days,
        initial_capital=args.initial_capital,
        risk_percentage=args.risk_percentage,
        slippage=args.slippage,
        commission_per_trade=args.commission_per_trade,
        commission_per_share=args.commission_per_share,
        max_positions=args.max_positions,
        max_sector_exposure=args.max_sector_exposure,
        output=args.output,    # optional now
    )

if __name__=="__main__":
    main()
