#!/usr/bin/env python3
"""
download_vix.py

Downloads the VIX index from FRED, computes common technical features,
and writes the result to a Parquet file.

Major steps:
  1. Read VIX closing series from FRED.
  2. Convert to DataFrame and set 'Date' index.
  3. Filter data from specified start date.
  4. Compute features:
       - Daily percent change
       - Rolling 5-day and 10-day moving averages
       - 5-day momentum
       - 20-day z-score
  5. Drop any rows containing NaNs.
  6. Reset index so 'Date' becomes a column again.
  7. Ensure output directory exists.
  8. Write processed DataFrame to Parquet.
Usage:
    python download_vix.py <output_folder> <start_date>
Example:
    python download_vix.py data/vix 2010-01-01
"""

import argparse                      # For command-line argument parsing
import os                            # For filesystem operations
from fredapi import Fred            # FRED API client
import pandas as pd                 # Data handling library


def download_and_process_vix(
    api_key: str,
    output_folder: str,
    start_date: str,
    output_filename: str = 'VIX.parquet'
) -> None:
    """
    Download VIX data from FRED, compute features, and save to Parquet.

    Args:
        api_key:        FRED API key string.
        output_folder:  Directory where output Parquet should be saved.
        start_date:     Only include data on or after this date (YYYY-MM-DD).
        output_filename:
                        Name of the output Parquet file (default 'VIX.parquet').
    """
    # 1) Initialize FRED client
    fred = Fred(api_key=api_key)

    # 2) Fetch full VIX closing series
    #    'VIXCLS' is the FRED series ID for the CBOE Volatility Index closing values
    vix_series = fred.get_series('VIXCLS')

    # 3) Convert to DataFrame with column name 'Close'
    vix_df = vix_series.to_frame(name='Close')

    # 4) Rename the index to 'Date' for clarity
    vix_df.index.name = 'Date'

    # 5) Filter rows to only include dates >= start_date
    #    Convert index to datetime to allow comparison
    start_dt = pd.to_datetime(start_date)
    vix_df = vix_df[vix_df.index >= start_dt]

    # 5b) Drop any rows that have NaN in any column
    vix_df = vix_df.dropna()

    # 6) Compute new feature columns:

    # 6a) Daily percent change in 'Close'
    vix_df['Pct_Change'] = vix_df['Close'].pct_change(fill_method=None)

    # 6b) 5-day simple moving average of 'Close'
    vix_df['MA_5'] = vix_df['Close'].rolling(window=5).mean()

    # 6c) 10-day simple moving average of 'Close'
    vix_df['MA_10'] = vix_df['Close'].rolling(window=10).mean()

    # 6d) 5-day momentum: difference between current close and close 5 days ago
    vix_df['Momentum_5'] = vix_df['Close'] - vix_df['Close'].shift(5)

    # 6e) 20-day rolling mean and standard deviation for z-score
    mean_20 = vix_df['Close'].rolling(window=20).mean()
    std_20  = vix_df['Close'].rolling(window=20).std()
    vix_df['ZScore_20'] = (vix_df['Close'] - mean_20) / std_20

    # 7) Drop any rows that have NaN in any column
    vix_df = vix_df.dropna()

    # 8) Reset index so 'Date' becomes a column instead of the index
    #vix_df = vix_df.reset_index()

    # 9) Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # 10) Build full output path and write to Parquet without the index
    output_path = os.path.join(output_folder, output_filename)
    vix_df.to_parquet(output_path, index=True)

    # 11) Notify user of successful save
    print(f"VIX feature file saved to: {output_path}")


def main() -> None:
    """
    Parse command-line arguments and invoke the download/process function.
    """
    parser = argparse.ArgumentParser(
        description="Download VIX data from FRED, compute features, and save to Parquet."
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Directory where the output Parquet file will be saved."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for the VIX series (inclusive) in YYYY-MM-DD format."
    )
    args = parser.parse_args()

    # 12) Load API key from a local file under data/api_keys/fred.txt
    api_key_path = os.path.join('data', 'api_keys', 'fred.txt')
    with open(api_key_path, 'r', encoding='utf-8') as f:
        api_key = f.read().strip()

    # 13) Call core function with parsed arguments
    download_and_process_vix(
        api_key=api_key,
        output_folder=args.output_folder,
        start_date=args.start_date
    )


if __name__ == "__main__":
    main()
