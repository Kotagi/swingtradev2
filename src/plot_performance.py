#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_trades(path):
    """
    Load the trade log CSV, parse dates, unify the shares/quantity column,
    and compute per-trade PnL and cumulative PnL.
    """
    df = pd.read_csv(path, parse_dates=['entry_date', 'exit_date'])
    # Rename 'shares' to 'quantity' if needed
    if 'shares' in df.columns:
        df.rename(columns={'shares': 'quantity'}, inplace=True)
    elif 'quantity' not in df.columns:
        raise KeyError("Expected 'shares' or 'quantity' column in trade log")
    # Compute PnL
    df['PnL'] = (df['exit_price'] - df['entry_price']) * df['quantity']
    # Sort and cumulative
    df.sort_values('exit_date', inplace=True)
    df['cum_pnl'] = df['PnL'].cumsum()
    return df

def plot_equity_curve(trades, output_dir):
    """Plot and save the equity curve."""
    plt.figure()
    plt.plot(trades['exit_date'], trades['cum_pnl'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    plt.close()

def plot_drawdowns(trades, output_dir):
    """Plot and save drawdowns."""
    cum = trades['cum_pnl']
    running_max = cum.cummax()
    drawdown = cum - running_max
    plt.figure()
    plt.fill_between(trades['exit_date'], drawdown, 0)
    plt.title('Drawdowns')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown.png'))
    plt.close()

def plot_return_distribution(trades, output_dir):
    """Plot and save the distribution of individual trade PnL."""
    plt.figure()
    plt.hist(trades['PnL'], bins=50)
    plt.title('Return Distribution')
    plt.xlabel('Trade PnL')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'return_distribution.png'))
    plt.close()

def compute_yearly_summary(trades, initial_capital):
    """
    Build a DataFrame of year-by-year stats:
      - total_pnl
      - return_pct (on initial capital)
      - num_trades
      - avg_trade_value (average entry_price * quantity)
    """
    trades['year'] = trades['exit_date'].dt.year
    def year_stats(x):
        return pd.Series({
            'total_pnl': x['PnL'].sum(),
            'return_pct': x['PnL'].sum() / initial_capital * 100,
            'num_trades': x.shape[0],
            'avg_trade_value': (x['entry_price'] * x['quantity']).mean()
        })
    summary = trades.groupby('year').apply(year_stats)
    return summary

def save_yearly_summary_md(summary, output_dir):
    """Format the yearly summary as markdown and write to a .md file."""
    md_table = summary.reset_index()
    md_table['total_pnl'] = md_table['total_pnl'].map('${:,.2f}'.format)
    md_table['return_pct'] = md_table['return_pct'].map('{:.2f}%'.format)
    md_table['avg_trade_value'] = md_table['avg_trade_value'].map('${:,.2f}'.format)
    md = md_table.to_markdown(index=False)
    report_path = os.path.join(output_dir, 'yearly_summary.md')
    with open(report_path, 'w') as f:
        f.write('### Year-by-Year Performance Summary\n\n')
        f.write(md)

def main():
    parser = argparse.ArgumentParser(description='Plot backtest performance')
    parser.add_argument('-l', '--trade-log', required=True,
                        help='path to trade log CSV')
    parser.add_argument('-i', '--initial-capital', type=float, default=100000,
                        help='initial capital for calculating return percentage')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='directory to save plots and tables')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading trades and computing PnL…')
    trades = load_trades(args.trade_log)

    print('Plotting equity curve...')
    plot_equity_curve(trades, args.output_dir)

    print('Plotting drawdowns...')
    plot_drawdowns(trades, args.output_dir)

    print('Plotting return distribution...')
    plot_return_distribution(trades, args.output_dir)

    print('Computing yearly summary...')
    summary = compute_yearly_summary(trades, args.initial_capital)
    save_yearly_summary_md(summary, args.output_dir)

    print('All performance charts and tables saved to', args.output_dir)

if __name__ == '__main__':
    main()
