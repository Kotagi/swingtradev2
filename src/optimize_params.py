#!/usr/bin/env python3
import argparse
import logging
import tempfile
import os
import pandas as pd

# make src/ a package (i.e. add an __init__.py) so this import always works:
from .backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_tickers(path):
    """Load one-per-line CSV of tickers (no header)."""
    return pd.read_csv(path, header=None)[0].tolist()

def compute_metrics(trades: pd.DataFrame):
    """Compute total_return, num_trades, win_rate, sharpe from a trades DataFrame."""
    # detect share count column
    if "quantity" in trades.columns:
        qty = trades["quantity"]
    elif "shares" in trades.columns:
        qty = trades["shares"]
    else:
        raise KeyError(f"Neither 'quantity' nor 'shares' found in {list(trades.columns)}")

    df = trades.copy()
    df["pnl"]    = (df["exit_price"] - df["entry_price"]) * qty
    df["return"] = df["pnl"] / (df["entry_price"] * qty)

    total_return = df["return"].sum()
    num_trades   = len(df)
    win_rate     = (df["pnl"] > 0).mean() if num_trades > 0 else 0.0
    sharpe       = (
        df["return"].mean() /
        df["return"].std() *
        (252 ** 0.5)
        if df["return"].std() not in (0, None)
        else float("nan")
    )

    return {
        "total_return": total_return,
        "num_trades":   num_trades,
        "win_rate":     win_rate,
        "sharpe":       sharpe,
    }

def main():
    parser = argparse.ArgumentParser(description="Grid-search your backtest params")
    parser.add_argument("-t", "--tickers-file",
                        default="data/tickers/sp500_tickers.csv",
                        help="CSV with one ticker per line")
    parser.add_argument("-c", "--clean-dir", default="data/clean",
                        help="Folder of cleaned price CSVs")
    parser.add_argument("-s", "--sectors-file",
                        default="data/tickers/sectors.csv",
                        help="CSV mapping ticker→sector")
    parser.add_argument("-i", "--initial-capital", type=float, default=100000)
    parser.add_argument("-o", "--output-dir", default="reports/param_sweep",
                        help="Where to write param_sweep_results.csv")
    args = parser.parse_args()

    tickers = load_tickers(args.tickers_file)
    logging.info(f"▶ Loaded {len(tickers)} tickers from {args.tickers_file}")

    # parameter grids
    momentum_grid = [0.01, 0.015, 0.02, 0.025]
    atr_grid      = [1.0, 1.5, 2.0, 2.5]
    exit_grid     = [5, 7, 10]
    risk_grid     = [0.01]

    combos = [
        (m, a, e, r)
        for m in momentum_grid
        for a in atr_grid
        for e in exit_grid
        for r in risk_grid
    ]
    logging.info(f"▶ Running {len(combos)} parameter combinations…")

    results = []
    for momentum, atr_mult, exit_days, risk_pct in combos:
        logging.info(f"▶ Testing: momentum={momentum}, ATR×{atr_mult}, exit={exit_days}d, risk={risk_pct}")

        # create a temp directory per combo (avoids file-lock on Windows)
        with tempfile.TemporaryDirectory() as td:
            tmp_out = os.path.join(td, "trade_log.csv")
            try:
                trades_df = run_backtest(
                    tickers,
                    clean_dir=args.clean_dir,
                    sectors_file=args.sectors_file,
                    momentum_threshold=momentum,
                    stop_loss_atr_mult=atr_mult,
                    time_exit_days=exit_days,
                    risk_percentage=risk_pct,

                    # always supply these on every run:
                    slippage=0,
                    commission_per_trade=0,
                    commission_per_share=0,

                    initial_capital=args.initial_capital,
                    output=tmp_out
                )
                metrics = compute_metrics(trades_df)
                results.append({
                    "momentum_threshold": momentum,
                    "stop_loss_atr_mult": atr_mult,
                    "time_exit_days":     exit_days,
                    "risk_percentage":    risk_pct,
                    **metrics
                })
            except Exception as exc:
                logging.warning(f"  ✗ Failed at this combo: {exc!r}")

    # save grid results
    df_res = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "param_sweep_results.csv")
    df_res.to_csv(out_csv, index=False)
    logging.info(f"✅ Grid-search complete. Results saved to {out_csv}")

    # report best sharpe
    if "sharpe" in df_res.columns and not df_res["sharpe"].isna().all():
        best = df_res.loc[df_res["sharpe"].idxmax()]
        logging.info("🏆 Best parameter set by Sharpe:")
        for field in ["momentum_threshold", "stop_loss_atr_mult",
                      "time_exit_days", "risk_percentage",
                      "total_return", "num_trades", "win_rate", "sharpe"]:
            logging.info(f"   • {field}: {best[field]}")
    else:
        logging.warning("⚠️ No Sharpe data available in the results.")

if __name__ == "__main__":
    main()
