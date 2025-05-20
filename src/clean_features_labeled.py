#!/usr/bin/env python3
"""
clean_features_labeled.py

Scan data/features_labeled/*.csv, drop any rows containing NaNs, and overwrite the files.
"""

import pandas as pd
from pathlib import Path

# —— CONFIG —— #
DATA_DIR = Path.cwd() / "data" / "features_labeled"

def clean_file(path: Path):
    # read with first column as datetime index
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    before = len(df)
    # drop any row that has at least one NaN
    df_clean = df.dropna()
    after = len(df_clean)
    # overwrite
    df_clean.to_csv(path)
    print(f"{path.name}: dropped {before-after} rows → {after} remaining")

def main():
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        print("No CSVs found in", DATA_DIR)
        return
    for f in csvs:
        clean_file(f)

if __name__ == "__main__":
    main()
