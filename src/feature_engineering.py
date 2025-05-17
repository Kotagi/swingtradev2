#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, MACD, and ATR columns to the DataFrame."""
    # SMA & EMA
    for w in (10, 20, 50):
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    # RSI (14)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ATR (14)
    high_low = df["High"] - df["Low"]
    high_prev = (df["High"] - df["Close"].shift()).abs()
    low_prev = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()

    return df


def main(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for csv_file in sorted(input_path.glob("*.csv")):
        df = pd.read_csv(csv_file, parse_dates=True, index_col=0)
        df_feat = compute_features(df)
        df_feat.to_csv(output_path / csv_file.name)
        print(f"Features written for {csv_file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save technical indicators.")
    parser.add_argument("--input-dir",  required=True, help="Directory of cleaned CSVs")
    parser.add_argument("--output-dir", required=True, help="Directory to write feature CSVs")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
