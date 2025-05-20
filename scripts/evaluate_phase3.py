#!/usr/bin/env python3
import sys
import os
import argparse
import glob
import logging
import pandas as pd
import joblib
from pathlib import Path

# Ensure src is on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from backtest import run_backtest

def load_features(features_dir):
    files = glob.glob(f"{features_dir}/*.csv")
    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")
    dfs, tickers = [], []
    for f in files:
        ticker = Path(f).stem
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        dfs.append(df)
        tickers.append(ticker)
    data = pd.concat(dfs, axis=0, keys=tickers, names=['ticker', 'date'])
    data = data.reset_index(level='ticker')
    return data

def generate_signals(data, model, feature_cols):
    mask = data[feature_cols].notnull().all(axis=1)
    data = data[mask].copy()
    data['signal'] = model.predict(data[feature_cols])
    return data

def evaluate(features_dir, model_path, sectors_file, report_path):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1) Load model and features
    model = joblib.load(model_path)
    data = load_features(features_dir)

    # 2) Validate features present
    feature_cols = ['5d_return', '10d_return', 'atr', 'bb_width',
                    'ema_cross', 'obv', 'rsi']
    for col in feature_cols:
        if col not in data.columns:
            raise KeyError(f"Missing feature column: {col}")

    # 3) Generate ML signals
    data = generate_signals(data, model, feature_cols)

    # DEBUG: log how many buy (1) vs no‐buy (0) signals
    logging.info("Signal distribution:\n%s", data['signal'].value_counts(dropna=False))

    # 4) Write per‐ticker CSVs with 'momentum' = ML signal
    temp_dir = Path("temp/eval_clean")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for ticker, grp in data.groupby('ticker'):
        df = grp.copy()
        df['momentum'] = df['signal']
        df.to_csv(temp_dir / f"{ticker}.csv")

    # 5) Run backtest
    trades = run_backtest(
        tickers=list(data['ticker'].unique()),
        clean_dir=str(temp_dir),
        sectors_file=sectors_file,
        momentum_threshold=0,
        stop_loss_atr_mult=2.0,
        time_exit_days=7,
        initial_capital=100000,
        risk_percentage=0.01,
        slippage=0.0,
        commission_per_trade=0.0,
        commission_per_share=0.0,
        max_positions=8,
        max_sector_exposure=0.25,
        output=None
    )

    # 6) Safely compute summary metrics
    if trades is None or (hasattr(trades, 'empty') and trades.empty) or 'pnl' not in trades.columns:
        total_pnl = 0.0
        num_trades = 0
        win_rate = 0.0
    else:
        total_pnl = trades['pnl'].sum()
        num_trades = len(trades)
        win_rate = (trades['pnl'] > 0).mean() if num_trades > 0 else 0.0

    # 7) Write the report
    summary = pd.DataFrame([{
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate
    }])
    with open(report_path, 'w') as f:
        f.write("# Phase3 Evaluation Results\n")
        f.write(summary.to_markdown(index=False))

    logging.info("Evaluation complete. Report saved to %s", report_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate ML strategy")
    parser.add_argument("--features-dir",  required=True, help="Labeled features CSV directory")
    parser.add_argument("--model",         required=True, help="Path to trained model .pkl")
    parser.add_argument("--sectors-file",  required=True, help="Path to sectors.csv")
    parser.add_argument("--report",        required=True, help="Output Markdown report path")
    args = parser.parse_args()
    evaluate(args.features_dir, args.model, args.sectors_file, args.report)

if __name__ == "__main__":
    main()
