import argparse
import os
import pandas as pd
import yfinance as yf
import logging

def download_ticker_data(ticker, start_date, end_date, raw_folder):
    """
    Download OHLCV data for a single ticker and save to CSV.
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        if df.empty:
            logging.warning(f"No data for {ticker}")
            return
        file_path = os.path.join(raw_folder, f"{ticker}.csv")
        df.to_csv(file_path)
        logging.info(f"Saved data for {ticker} to {file_path}")
    except Exception as e:
        logging.error(f"Error downloading {ticker}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Download OHLCV data for a list of tickers"
    )
    parser.add_argument(
        "--tickers-file",
        default="data/tickers/sp500_tickers.csv",
        help="Path to CSV file containing tickers (comma-separated or one per line)"
    )
    parser.add_argument(
        "--start-date",
        default="2008-01-01",
        help="Start date for historical data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date for historical data (YYYY-MM-DD). Default is today"
    )
    parser.add_argument(
        "--raw-folder",
        default="data/raw",
        help="Folder to save raw ticker CSV files"
    )
    args = parser.parse_args()

    # Ensure the raw data directory exists
    os.makedirs(args.raw_folder, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    # Read tickers (handles both comma-separated or one-per-line)
    with open(args.tickers_file, "r") as f:
        text = f.read()
    # Normalize newlines to commas, split, strip whitespace, and drop empties
    tickers = [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]
    # Remove duplicates while preserving order
    tickers = list(dict.fromkeys(tickers))

    logging.info(f"Found {len(tickers)} tickers to download")

    # Download data for each ticker
    for idx, ticker in enumerate(tickers, 1):
        logging.info(f"[{idx}/{len(tickers)}] Downloading {ticker}")
        download_ticker_data(ticker, args.start_date, args.end_date, args.raw_folder)

if __name__ == "__main__":
    main()
