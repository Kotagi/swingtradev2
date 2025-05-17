#!/usr/bin/env python
import os
import argparse
import pandas as pd

def load_trades(path):
    df = pd.read_csv(path)
    return df

def compute_performance(trades: pd.DataFrame, initial_capital: float):
    """
    Given a DataFrame of trades, compute:
      - PnL per trade
      - cumulative PnL
      - returns and summary metrics
    Returns (metrics_df, report_md)
    """
    t = trades.copy()

    # determine quantity column (fallback to 1)
    if "quantity" in t.columns:
        qty = t["quantity"]
    elif "size" in t.columns:
        qty = t["size"]
    else:
        qty = 1

    # determine commission column (fallback to 0)
    comm = t["commission"] if "commission" in t.columns else 0

    # PnL per trade
    t["PnL"] = (t["exit_price"] - t["entry_price"]) * qty - comm
    t["cum_pnl"] = t["PnL"].cumsum()
    t["return"] = t["PnL"] / initial_capital

    # summary metrics
    total_return = t["return"].sum()
    num_trades   = len(t)
    win_rate     = (t["PnL"] > 0).mean() if num_trades > 0 else 0.0
    sharpe       = (t["return"].mean() / t["return"].std() * (252**0.5)
                    if t["return"].std() != 0 else float("nan"))

    # pack into a DataFrame
    metrics = {
        "total_return": total_return,
        "num_trades":   num_trades,
        "win_rate":     win_rate,
        "sharpe":       sharpe,
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])
    metrics_df.index.name = "metric"
    metrics_df = metrics_df.reset_index()

    # simple Markdown report
    report_md = (
        f"# Performance Report\n\n"
        f"- **Initial capital**: ${initial_capital:,.0f}\n"
        f"- **Total return**: {total_return:.2%}\n"
        f"- **Number of trades**: {num_trades}\n"
        f"- **Win rate**: {win_rate:.2%}\n"
        f"- **Annualized Sharpe**: {sharpe:.2f}\n\n"
        f"## Equity Curve\n\n"
        f"```csv\n"
        f"{t[['exit_date','cum_pnl']].to_csv(index=False)}"
        f"```\n"
    )

    return metrics_df, report_md

def main():
    p = argparse.ArgumentParser("evaluate.py", 
        description="Compute performance metrics from a backtest trade log")
    p.add_argument(
        "-l","--trade-log", required=True,
        help="Path to the backtest-generated trade_log.csv"
    )
    p.add_argument(
        "-i","--initial-capital", type=float, default=100000,
        help="Starting capital for return calculations"
    )
    p.add_argument(
        "-o","--output", required=True,
        help="Directory to write performance_metrics.csv and performance_report.md"
    )
    args = p.parse_args()

    trades = load_trades(args.trade_log)
    print(f"{len(trades)} trades loaded from {args.trade_log}")

    metrics_df, report_md = compute_performance(trades, args.initial_capital)

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "performance_metrics.csv")
    md_path  = os.path.join(args.output, "performance_report.md")

    metrics_df.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Metrics CSV: {csv_path}")
    print(f"Report MD:   {md_path}")

if __name__ == "__main__":
    main()
