#!/usr/bin/env python3
"""
download_data.py

Downloads historical OHLCV price data (incrementally by default, or fully with --full)
and sector information for a list of tickers using yfinance, saving each ticker's data
to a CSV file and producing a sectors.csv mapping each ticker symbol to its sector.

On incremental runs, it only fetches data *after* the last downloaded date for each ticker,
append the new rows, and overwrites the CSV. With --full, it re-downloads from scratch.

Usage:
    # Incremental (default)
    python src/download_data.py \
        --tickers-file data/tickers/sp500_tickers.csv \
        --start-date 2008-01-01 \
        --end-date 2025-05-19 \
        --raw-folder data/raw \
        --sectors-file data/tickers/sectors.csv

    # Full redownload
    python src/download_data.py --full \
        --tickers-file data/tickers/sp500_tickers.csv \
        --start-date 2008-01-01 \
        --raw-folder data/raw
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import pandas as pd
import yfinance as yf


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str],
    raw_folder: str,
    full_refresh: bool = False
) -> List[Tuple[str, str]]:
    """
    Download OHLCV data and fetch sector info for each ticker.

    Args:
        tickers: List of ticker symbols.
        start_date: Earliest date to download if no prior data (YYYY-MM-DD).
        end_date: Latest date (exclusive) for downloads, or None for today.
        raw_folder: Directory path to save raw CSV files.
        full_refresh: If True, ignore existing CSVs and re-download from start_date.

    Returns:
        A list of (ticker, sector) tuples for all tickers processed.
    """
    os.makedirs(raw_folder, exist_ok=True)
    sectors: List[Tuple[str, str]] = []
    total = len(tickers)

    # Compute global end date (yfinance end is exclusive)
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end_dt = datetime.today().date() + timedelta(days=1)

    for idx, ticker in enumerate(tickers, start=1):
        out_path = os.path.join(raw_folder, f"{ticker}.csv")
        logging.info(f"[{idx}/{total}] Ticker {ticker}:")

        # Determine where to start downloading
        if not full_refresh and os.path.exists(out_path):
            try:
                # Read existing CSV without parse_dates to avoid warnings
                existing = pd.read_csv(out_path, index_col=0)
                # Explicitly parse the index as dates
                existing.index = pd.to_datetime(
                    existing.index, format="%Y-%m-%d", errors="coerce"
                )
                last_date = existing.index.max().date()
                ticker_start = last_date + timedelta(days=1)
                logging.info(f"  existing data through {last_date}, starting {ticker_start}")
            except Exception as e:
                logging.warning(f"  could not read existing CSV, will re-download from start_date: {e}")
                existing = None
                ticker_start = datetime.strptime(start_date, "%Y-%m-%d").date()
        else:
            # full_refresh=True or no existing file: start at start_date
            existing = None
            ticker_start = datetime.strptime(start_date, "%Y-%m-%d").date()
            if full_refresh and os.path.exists(out_path):
                logging.info("  full refresh requested, ignoring existing data")
            else:
                logging.info(f"  no existing data, starting {ticker_start}")

        # Skip if already up-to-date
        if ticker_start >= end_dt:
            logging.info("  up-to-date, no new data to download")
        else:
            # Download the slice we need
            df_new = yf.download(
                ticker,
                start=ticker_start.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False
            )
            if df_new.empty:
                logging.info("  no new rows fetched")
            else:
                # Combine with existing or use new alone
                if existing is not None:
                    df_combined = pd.concat([existing, df_new])
                    # Drop duplicate dates, keep the first occurrence
                    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
                else:
                    df_combined = df_new

                # Overwrite CSV with updated history
                df_combined.to_csv(out_path)
                logging.info(f"  wrote {len(df_new)} new rows, total {len(df_combined)}")

        # Fetch sector info (cheap, so always run)
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
        except Exception as e:
            logging.warning(f"  sector lookup failed: {e}")
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
    df = pd.DataFrame(sectors, columns=["ticker", "sector"])
    df.to_csv(sectors_file, index=False)
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
        help="End date (exclusive) for historical data (YYYY-MM-DD); defaults to today"
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
    parser.add_argument(
        "--full", "--force-full",
        action="store_true",
        dest="full_refresh",
        help="Ignore existing files and re-download entire history for every ticker"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    # Load tickers list
    with open(args.tickers_file, "r", encoding="utf-8") as f:
        tickers = [line.strip() for line in f if line.strip()]

    # Perform download (incremental or full)
    sectors = download_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_folder=args.raw_folder,
        full_refresh=args.full_refresh
    )

    # Write out the sectors CSV
    write_sectors_csv(sectors, args.sectors_file)

    logging.info("Download complete.")


if __name__ == "__main__":
    main()
