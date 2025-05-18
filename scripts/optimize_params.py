#!/usr/bin/env python
"""
scripts/optimize_params.py

Grid‐search momentum threshold, ATR multiplier, time‐exit days and risk percentage
to maximize Sharpe ratio of your rule‐based backtest.
"""

import os
import sys
import argparse
import itertools
import pandas as pd

# allow imports from the src/ folder
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src.backtest import run_backtest
from src.evaluate import compute_performance

def main():
    parser = argparse.ArgumentParser(
        description="Grid‐search backtest parameters and save results."
    )
    parser.add_argument(
        "-t", "--tickers-file", required=True,
        help="One‐per‐line file of tickers to backtest"
    )
    parser.add_argument(
        "-c", "--clean-dir", default="data/clean",
        help="Directory containing cleaned CSVs"
    )
    parser.add_argument(
        "-s", "--sectors-file", required=True,
        help="CSV mapping Ticker→Sector"
    )
    parser.add_argument(
        "-i", "--initial-capital", type=float, default=100000,
        help="Starting capital for performance metrics"
    )
    parser.add_argument(
        "-o", "--output-dir", default="reports/param_sweep",
        help="Where to save param_sweep_results.csv"
    )
    args = parser.parse_args()

    # load tickers
    with open(args.tickers_file) as f:
        tickers = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_dir, exist_ok=True)

    # define grids
    momentum_list   = [0.005, 0.01, 0.015, 0.02]
    atr_mults       = [1.0, 1.5, 2.0]
    exit_days_list  = [3, 5, 7]
    risk_list       = [0.005, 0.01, 0.02]

    sweep_results = []

    for momentum, atr_mult, exit_days, risk in itertools.product(
            momentum_list, atr_mults, exit_days_list, risk_list
        ):
        print(f"▶ Testing: momentum={momentum}, ATR×{atr_mult}, exit={exit_days}d, risk={risk}")
        trades_df = run_backtest(
            tickers=tickers,
            clean_dir=args.clean_dir,
            sectors_file=args.sectors_file,
            momentum_threshold=momentum,
            stop_loss_atr_mult=atr_mult,
            time_exit_days=exit_days,
            slippage=0.0,
            commission_per_trade=0.0,
            commission_per_share=0.0,
            initial_capital=args.initial_capital,
            risk_percentage=risk,
            output=None  # don't write individual logs here
        )

        # compute performance metrics
        metrics_df, _ = compute_performance(trades_df, args.initial_capital)
        metrics = metrics_df.set_index("metric")["value"].to_dict()

        sweep_results.append({
            "momentum_threshold": momentum,
            "stop_loss_atr_mult": atr_mult,
            "time_exit_days": exit_days,
            "risk_percentage": risk,
            **metrics
        })

    # save the sweep
    out_csv = os.path.join(args.output_dir, "param_sweep_results.csv")
    pd.DataFrame(sweep_results).to_csv(out_csv, index=False)
    print(f"\n✅ Grid‐search complete. Results saved to {out_csv}")

    # report best by sharpe, if available
    df = pd.DataFrame(sweep_results)
    if "sharpe" in df.columns:
        best = df.loc[df["sharpe"].idxmax()]
        print("\n🏆 Best parameter set by sharpe:")
        print(f"   • momentum_threshold: {best['momentum_threshold']}")
        print(f"   • stop_loss_atr_mult:  {best['stop_loss_atr_mult']}")
        print(f"   • time_exit_days:      {best['time_exit_days']}")
        print(f"   • risk_percentage:     {best['risk_percentage']}")
        print(f"   • sharpe:              {best['sharpe']:.4f}")
        print(f"   • total_return:        {best.get('total_return', float('nan')):.2f}")
        print(f"   • win_rate:            {best.get('win_rate', float('nan')):.3f}")
        print(f"   • num_trades:          {best.get('num_trades', float('nan')):.0f}")
    else:
        print("⚠️ No 'sharpe' column found. Available columns:", list(df.columns))

if __name__ == "__main__":
    main()
