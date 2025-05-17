#!/usr/bin/env python3
"""
download_data.py

Downloads raw historical OHLCV data for a list of tickers using yfinance,
and writes out a sectors.csv mapping each ticker to its sector.
"""

import argparse
import logging
import os
import csv
import yfinance as yf


def download_data(tickers, start_date, end_date, raw_folder):
    """
    Downloads price data for each ticker and saves to raw_folder/<ticker>.csv.
    Also collects sector info to write to sectors.csv.
    """
    os.makedirs(raw_folder, exist_ok=True)
    sectors = []
    total = len(tickers)
    for i, ticker in enumerate(tickers, start=1):
        out_path = os.path.join(raw_folder, f"{ticker}.csv")
        logging.info(f"[{i}/{total}] Downloading {ticker} to {out_path}")
        # Download historical data
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        # Save to CSV
        df.to_csv(out_path)
        # Fetch sector info
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
        except Exception:
            sector = "Unknown"
        sectors.append((ticker, sector))
    return sectors


def write_sectors_csv(sectors, sectors_file):
    """
    Writes sectors.csv with columns: ticker,sector
    """
    os.makedirs(os.path.dirname(sectors_file), exist_ok=True)
    with open(sectors_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "sector"])
        for ticker, sector in sectors:
            writer.writerow([ticker, sector])
    logging.info(f"Wrote sector lookup to {sectors_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download raw price data and sector info"
    )
    parser.add_argument(
        "--tickers-file", "-f",
        default="data/tickers/sp500_tickers.csv",
        help="One ticker per line"
    )
    parser.add_argument(
        "--start-date", "-s",
        default="2008-01-01",
        help="Start date for historical download (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", "-e",
        default=None,
        help="End date for historical download (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--raw-folder", "-r",
        default="data/raw",
        help="Directory to save raw CSV files"
    )
    parser.add_argument(
        "--sectors-file", "-o",
        default="data/tickers/sectors.csv",
        help="Output CSV file for ticker–sector mapping"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    # Read tickers
    if not os.path.isfile(args.tickers_file):
        logging.error(f"Tickers file not found: {args.tickers_file}")
        return
    with open(args.tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Download data and collect sectors
    sectors = download_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_folder=args.raw_folder
    )

    # Write sectors CSV
    write_sectors_csv(sectors, args.sectors_file)

    logging.info("Done!")


if __name__ == "__main__":
    main()
