# src/download_data.py

#!/usr/bin/env python3
"""
download_data.py

Downloads historical OHLCV price data and sector information for a list of tickers
using yfinance, saving each ticker's data to a CSV file and producing a sectors.csv
mapping each ticker symbol to its sector.

Usage:
    python src/download_data.py \
        --tickers-file data/tickers/sp500_tickers.csv \
        --start-date 2008-01-01 \
        --end-date 2025-05-19 \
        --raw-folder data/raw \
        --sectors-file data/tickers/sectors.csv
"""

import argparse
import logging
import os
import csv
from typing import List, Tuple, Optional

import yfinance as yf


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    raw_folder: str
) -> List[Tuple[str, str]]:
    """
    Download OHLCV data and fetch sector info for each ticker.

    Args:
        tickers: List of ticker symbols to download.
        start_date: Historical start date (YYYY-MM-DD).
        end_date: Historical end date (YYYY-MM-DD) or None for today.
        raw_folder: Directory path to save raw CSV files.

    Returns:
        A list of (ticker, sector) tuples.
    """
    os.makedirs(raw_folder, exist_ok=True)
    sectors: List[Tuple[str, str]] = []
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        out_path = os.path.join(raw_folder, f"{ticker}.csv")
        logging.info(f"[{idx}/{total}] Downloading {ticker} to {out_path}")

        # Download historical OHLCV data
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )
        df.to_csv(out_path)

        # Fetch sector information
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
        except Exception as e:
            logging.warning(f"Sector lookup failed for {ticker}: {e}")
            sector = "Unknown"

        sectors.append((ticker, sector))

    return sectors


def write_sectors_csv(
    sectors: List[Tuple[str, str]],
    sectors_file: str
) -> None:
    """
    Write the ticker-to-sector mapping to a CSV file.

    Args:
        sectors: List of (ticker, sector) tuples.
        sectors_file: File path to write the sectors CSV.
    """
    os.makedirs(os.path.dirname(sectors_file), exist_ok=True)
    with open(sectors_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "sector"])
        for ticker, sector in sectors:
            writer.writerow([ticker, sector])

    logging.info(f"Wrote sector lookup to {sectors_file}")


def main() -> None:
    """
    Parse command-line arguments, download data, and write sector mapping.
    """
    parser = argparse.ArgumentParser(
        description="Download raw price data and sector info using yfinance"
    )
    parser.add_argument(
        "-f", "--tickers-file",
        default="data/tickers/sp500_tickers.csv",
        help="Path to a file with one ticker symbol per line"
    )
    parser.add_argument(
        "-s", "--start-date",
        default="2008-01-01",
        help="Start date for historical data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "-e", "--end-date",
        default=None,
        help="End date for historical data (YYYY-MM-DD); defaults to today"
    )
    parser.add_argument(
        "-r", "--raw-folder",
        default="data/raw",
        help="Directory to save downloaded raw CSV files"
    )
    parser.add_argument(
        "-o", "--sectors-file",
        default="data/tickers/sectors.csv",
        help="Output CSV for ticker–sector mapping"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    # Validate tickers file path
    if not os.path.isfile(args.tickers_file):
        logging.error(f"Tickers file not found: {args.tickers_file}")
        return

    # Read tickers into a list
    with open(args.tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Download data and collect sector information
    sectors = download_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_folder=args.raw_folder
    )

    # Write sector mapping to CSV
    write_sectors_csv(sectors, args.sectors_file)

    logging.info("Download and sector mapping complete.")


if __name__ == "__main__":
    main()

