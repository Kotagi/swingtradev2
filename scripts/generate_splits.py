#!/usr/bin/env python3
import argparse
import pandas as pd
from utils.splits import walk_forward_splits, save_splits

def main(features_dir: str, out: str,
         train_days: int, test_days: int, step_days: int):
    # Load one ticker’s features to get the date index
    df = pd.read_csv(f"{features_dir}/AAPL.csv", index_col=0, parse_dates=True)
    dates = df.index

    # Generate splits
    splits = walk_forward_splits(
        dates,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days
    )

    # Save to JSON
    save_splits(splits, out)
    print(f"Saved {len(splits)} folds to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate walk-forward splits")
    p.add_argument("--features-dir", required=True,
                   help="Directory of labeled feature CSVs")
    p.add_argument("--out", required=True,
                   help="Path to write splits.json")
    p.add_argument("--train-days", type=int, default=2000,
                   help="Training window size in days")
    p.add_argument("--test-days",  type=int, default=250,
                   help="Test window size in days")
    p.add_argument("--step-days",  type=int, default=60,
                   help="Step size between folds in days")
    args = p.parse_args()
    main(args.features_dir, args.out,
         args.train_days, args.test_days, args.step_days)
