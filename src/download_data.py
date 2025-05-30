#!/usr/bin/env python3
"""
download_data.py

Downloads historical OHLCV price data for a list of tickers in bulk (chunked) with
incremental updates (or full redownload with --full), fetches daily split factors,
and optionally records sector information.

Major steps:
  1. Load tickers list from file.
  2. Break tickers into chunks to avoid rate limits.
  3. For each chunk:
       a. Bulk-download OHLCV data for all tickers in chunk.
       b. In parallel, per ticker:
            i.   Use --full to force full redownload (ignore existing CSV).
           ii.   Read existing CSV (if any) and merge incremental rows.
          iii.   Fetch split factors (split_coefficient) and assign.
           iv.   Write updated CSV to raw folder.
            v.   If not --no-sectors, fetch sector info for ticker.
  4. If not --no-sectors, write out a sectors.csv mapping each ticker to its sector.

Usage:
    python src/download_data.py \\
        --tickers-file data/tickers/sp500_tickers.csv \\
        --start-date 2008-01-01 --end-date 2025-05-19 \\
        --raw-folder data/raw --sectors-file data/tickers/sectors.csv \\
        --chunk-size 100 --pause 1.0 \\
        [--full] [--no-sectors]
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

# —— CONFIGURATION —— #
DEFAULT_TICKERS_FILE = "data/tickers/sp500_tickers.csv"
DEFAULT_START_DATE   = "2008-01-01"
DEFAULT_END_DATE     = None   # None implies up to today
DEFAULT_RAW_FOLDER   = "data/raw"
DEFAULT_SECTORS_FILE = "data/tickers/sectors.csv"
DEFAULT_CHUNK_SIZE   = 100
DEFAULT_PAUSE        = 1.0    # seconds between chunks
MAX_WORKERS          = 8      # parallel threads for per-ticker processing


def chunked_list(items: list, chunk_size: int):
    """
    Yield successive chunks of size `chunk_size` from `items`.

    Args:
        items: List of items to split.
        chunk_size: Maximum size of each chunk.

    Yields:
        Sublists of `items` of length up to `chunk_size`.
    """
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def process_symbol(
    symbol: str,
    df_all: pd.DataFrame,
    raw_folder: str,
    start_date: str,
    yf_end: str,
    full_refresh: bool,
    write_sectors: bool
) -> tuple:
    """
    Process one ticker: merge data, fetch splits, write CSV, and optionally fetch sector.

    Steps:
      1. Extract ticker slice from bulk DataFrame.
      2. Full vs incremental update logic.
      3. Fetch and assign split_coefficient.
      4. Write per-ticker CSV.
      5. If write_sectors, fetch sector info via yfinance.

    Args:
        symbol: Ticker symbol to process.
        df_all: Bulk-downloaded DataFrame grouped by ticker.
        raw_folder: Directory to save raw CSVs.
        start_date: Earliest date for split fetch (YYYY-MM-DD).
        yf_end: End date string for yfinance queries (YYYY-MM-DD).
        full_refresh: If True, redownload full history ignoring existing CSV.
        write_sectors: If False, skip sector lookup and mapping.

    Returns:
        (symbol, sector) tuple if write_sectors, else (symbol, None).
    """
    # 1) Extract this ticker's data
    df_t = df_all.get(symbol)
    if df_t is None or df_t.empty:
        raise ValueError(f"{symbol}: no data returned")
    df_t.index = pd.to_datetime(df_t.index)

    out_path = Path(raw_folder) / f"{symbol}.csv"

    # 2) Full vs incremental update
    if full_refresh:
        df_combined = df_t
        logging.info(f"{symbol}: full refresh; writing {len(df_combined)} rows")
    else:
        if out_path.exists():
            existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
            last_date = existing.index.max()
            new_rows = df_t[df_t.index > last_date]
            if new_rows.empty:
                df_combined = existing
                logging.info(f"{symbol}: up-to-date (no new rows)")
            else:
                df_combined = pd.concat([existing, new_rows])
                df_combined = df_combined[~df_combined.index.duplicated(keep="first")]
                logging.info(f"{symbol}: added {len(new_rows)} new rows")
        else:
            df_combined = df_t
            logging.info(f"{symbol}: creating new CSV with {len(df_combined)} rows")

    # 3) Fetch split factors and assign
    try:
        actions = yf.Ticker(symbol).history(
            start=start_date,
            end=yf_end,
            actions=True
        )
        splits = actions.get("Stock Splits", pd.Series(dtype=float))
        splits = splits.reindex(df_combined.index, fill_value=0).replace(0, 1.0).astype(float)
        df_combined = df_combined.copy()
        df_combined.loc[:, "split_coefficient"] = splits
    except Exception as e:
        logging.warning(f"{symbol}: could not fetch splits: {e}")
        df_combined = df_combined.copy()
        df_combined.loc[:, "split_coefficient"] = 1.0

    # 4) Write per-ticker CSV
    df_combined.to_csv(out_path, index=True)

    # 5) Optionally fetch sector info
    sector = None
    if write_sectors:
        try:
            sector = yf.Ticker(symbol).info.get("sector", "Unknown")
        except Exception as e:
            logging.warning(f"{symbol}: sector lookup failed: {e}")
            sector = "Unknown"

    return symbol, sector


def download_data(
    tickers: list,
    start_date: str,
    end_date: str,
    raw_folder: str,
    chunk_size: int,
    pause: float,
    full_refresh: bool,
    write_sectors: bool
) -> list:
    """
    Bulk-download OHLCV data in chunks, then parallel-process each ticker.

    Steps:
      1. Break tickers into chunks.
      2. For each chunk:
         a. Bulk-download OHLCV for chunk.
         b. Submit each ticker to ThreadPoolExecutor for processing.
         c. Log a running count of processed tickers.
         d. Sleep to respect rate limits.
      3. Return sector mappings if requested.

    Args:
        tickers: List of ticker symbols.
        start_date: Earliest date for download (YYYY-MM-DD).
        end_date: Latest date (exclusive) or None for today.
        raw_folder: Directory to save per-ticker CSVs.
        chunk_size: Number of tickers per bulk request.
        pause: Seconds to wait between chunks.
        full_refresh: Force full download ignoring existing CSVs.
        write_sectors: If False, skip sector lookups and mapping.

    Returns:
        List of (symbol, sector) tuples; sector is None if write_sectors is False.
    """
    os.makedirs(raw_folder, exist_ok=True)
    sectors = []
    total = len(tickers)
    processed = 0
    yf_end = end_date or datetime.today().strftime("%Y-%m-%d")

    for chunk in chunked_list(tickers, chunk_size):
        logging.info(f"Downloading chunk of {len(chunk)} tickers...")
        df_all = yf.download(
            tickers=chunk,
            start=start_date,
            end=yf_end,
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=False
        )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_symbol,
                    symbol, df_all, raw_folder, start_date, yf_end,
                    full_refresh, write_sectors
                ): symbol
                for symbol in chunk
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    symbol, sector = fut.result()
                    if write_sectors:
                        sectors.append((symbol, sector))
                except Exception as e:
                    logging.error(f"{sym}: processing failed: {e}")
                processed += 1
                logging.info(f"Processed {processed}/{total} tickers")

        logging.info(f"Sleeping {pause}s before next chunk…")
        time.sleep(pause)

    return sectors


def write_sectors_csv(
    sectors: list,
    sectors_file: str
) -> None:
    """
    Write the ticker-to-sector mapping to a CSV file.

    Args:
        sectors: List of (symbol, sector) tuples.
        sectors_file: Path to write the sectors CSV.
    """
    os.makedirs(Path(sectors_file).parent, exist_ok=True)
    df = pd.DataFrame(sectors, columns=["ticker", "sector"])
    df.to_csv(sectors_file, index=False)
    logging.info(f"Wrote sector mapping to {sectors_file}")


def main() -> None:
    """
    Parse command-line arguments, download data, and optionally write sector mapping.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Bulk-download price data, split factors, and optional sector info"
    )
    parser.add_argument(
        "--tickers-file",
        default=DEFAULT_TICKERS_FILE,
        help="One ticker symbol per line"
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="End date (exclusive, YYYY-MM-DD); defaults to today"
    )
    parser.add_argument(
        "--raw-folder",
        default=DEFAULT_RAW_FOLDER,
        help="Directory for raw per-ticker CSVs"
    )
    parser.add_argument(
        "--sectors-file",
        default=DEFAULT_SECTORS_FILE,
        help="Output CSV for ticker→sector mapping"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of tickers per bulk HTTP request"
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=DEFAULT_PAUSE,
        help="Seconds to sleep between each chunk"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        dest="full_refresh",
        help="Force full redownload of all tickers (ignore existing CSVs)"
    )
    parser.add_argument(
        "--no-sectors",
        action="store_false",
        dest="write_sectors",
        help="Skip fetching and writing sector information"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    with open(args.tickers_file, "r", encoding="utf-8") as fh:
        tickers = [line.strip() for line in fh if line.strip()]

    sectors = download_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_folder=args.raw_folder,
        chunk_size=args.chunk_size,
        pause=args.pause,
        full_refresh=args.full_refresh,
        write_sectors=args.write_sectors
    )

    if args.write_sectors:
        write_sectors_csv(sectors, args.sectors_file)
    else:
        logging.info("Skipping sector file write (--no-sectors specified)")


if __name__ == "__main__":
    main()
