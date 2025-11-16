#!/usr/bin/env python3
"""
download_data.py

Enhanced download script with:
- Data validation
- Retry logic with exponential backoff
- Progress tracking
- Resume capability
- Network resilience
- Missing data detection
- Summary statistics
- Adaptive rate limiting

Downloads historical OHLCV price data for a list of tickers in bulk (chunked) with
incremental updates (or full redownload with --full), fetches daily split factors,
and optionally records sector information.

Usage:
    python src/download_data.py \\
        --tickers-file data/tickers/sp500_tickers.csv \\
        --start-date 2008-01-01 --end-date 2025-05-19 \\
        --raw-folder data/raw --sectors-file data/tickers/sectors.csv \\
        --chunk-size 100 --pause 1.0 \\
        [--full] [--no-sectors] [--resume] [--max-retries 3]
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict
import traceback

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# —— CONFIGURATION —— #
DEFAULT_TICKERS_FILE = "data/tickers/sp500_tickers.csv"
DEFAULT_START_DATE   = "2008-01-01"
DEFAULT_END_DATE     = None   # None implies up to today
DEFAULT_RAW_FOLDER   = "data/raw"
DEFAULT_SECTORS_FILE = "data/tickers/sectors.csv"
DEFAULT_CHUNK_SIZE   = 100
DEFAULT_PAUSE        = 1.0    # seconds between chunks
DEFAULT_MAX_RETRIES  = 3
MAX_WORKERS          = 8      # parallel threads for per-ticker processing
REQUEST_TIMEOUT      = 30     # seconds for yfinance requests
MIN_PAUSE            = 0.5    # minimum pause between chunks (adaptive rate limiting)
MAX_PAUSE            = 10.0   # maximum pause between chunks


def chunked_list(items: list, chunk_size: int):
    """Yield successive chunks of size `chunk_size` from `items`."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def validate_ohlcv_data(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
    """
    Validate OHLCV data quality.
    
    Args:
        df: DataFrame with OHLCV columns
        symbol: Ticker symbol for error messages
        
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Check for lowercase versions too
    lower_cols = {col: col.lower() for col in required_cols}
    
    # Find actual column names (case-insensitive)
    actual_cols = {}
    for req_col in required_cols:
        if req_col in df.columns:
            actual_cols[req_col] = req_col
        elif req_col.lower() in df.columns:
            actual_cols[req_col] = req_col.lower()
        else:
            issues.append(f"Missing required column: {req_col}")
            return False, issues
    
    # Validate data quality (only check non-NaN values)
    for col_name in ['Open', 'High', 'Low', 'Close']:
        col = actual_cols[col_name]
        if col in df.columns:
            # Only check non-NaN values
            non_na = df[col].dropna()
            if len(non_na) > 0:
                # Check for negative prices
                if (non_na < 0).any():
                    issues.append(f"Negative {col_name} prices found")
                # Check for zero prices (except in special cases)
                if (non_na == 0).any():
                    issues.append(f"Zero {col_name} prices found")
            # Note: NaN values are expected and will be cleaned, so don't flag as critical
    
    # Validate High >= Low (only check rows where both are non-NaN)
    high_col = actual_cols['High']
    low_col = actual_cols['Low']
    if high_col in df.columns and low_col in df.columns:
        valid_mask = df[high_col].notna() & df[low_col].notna()
        if valid_mask.any() and (df.loc[valid_mask, high_col] < df.loc[valid_mask, low_col]).any():
            issues.append("High < Low found (invalid price data)")
    
    # Validate High >= Close and Low <= Close (only check non-NaN rows)
    close_col = actual_cols['Close']
    if high_col in df.columns and close_col in df.columns:
        valid_mask = df[high_col].notna() & df[close_col].notna()
        if valid_mask.any() and (df.loc[valid_mask, high_col] < df.loc[valid_mask, close_col]).any():
            issues.append("High < Close found (invalid price data)")
    if low_col in df.columns and close_col in df.columns:
        valid_mask = df[low_col].notna() & df[close_col].notna()
        if valid_mask.any() and (df.loc[valid_mask, low_col] > df.loc[valid_mask, close_col]).any():
            issues.append("Low > Close found (invalid price data)")
    
    # Validate Volume
    vol_col = actual_cols.get('Volume', 'volume')
    if vol_col in df.columns:
        if (df[vol_col] < 0).any():
            issues.append("Negative volume found")
    
    return len(issues) == 0, issues


def retry_with_backoff(func, max_retries: int = DEFAULT_MAX_RETRIES, 
                       base_delay: float = 1.0, symbol: str = ""):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry (should return result or raise exception)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        symbol: Ticker symbol for logging
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                if symbol:
                    logging.warning(f"{symbol}: Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay:.1f}s...")
                else:
                    logging.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                if symbol:
                    logging.error(f"{symbol}: All {max_retries + 1} attempts failed. Last error: {e}")
                else:
                    logging.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
    
    raise last_exception


def detect_missing_data(df: pd.DataFrame, start_date: str, end_date: str, 
                        symbol: str) -> List[Tuple[str, str]]:
    """
    Detect significant gaps in historical data (only actual missing trading days).
    
    Args:
        df: DataFrame with datetime index
        start_date: Expected start date
        end_date: Expected end date
        symbol: Ticker symbol
        
    Returns:
        List of (gap_start, gap_end) tuples for gaps > 5 trading days
    """
    if df.empty:
        return [(start_date, end_date)]
    
    gaps = []
    df_sorted = df.sort_index()
    
    if len(df_sorted) < 2:
        return gaps
    
    # Calculate trading day gaps (accounting for weekends/holidays)
    # Only flag gaps larger than 5 trading days (approximately 1 week)
    date_diff = df_sorted.index.to_series().diff()
    
    # Convert time deltas to approximate trading days (exclude weekends)
    # A gap of 7 calendar days = ~5 trading days
    # A gap of 10 calendar days = ~7 trading days
    # Only report gaps > 10 calendar days (significant missing data)
    significant_gaps = date_diff[date_diff > pd.Timedelta(days=10)]
    
    for gap_date in significant_gaps.index:
        prev_date = df_sorted.index[df_sorted.index < gap_date].max() if len(df_sorted.index[df_sorted.index < gap_date]) > 0 else None
        next_date = gap_date
        if prev_date:
            # Only add if gap is truly significant (not just weekends/holidays)
            gap_days = (next_date - prev_date).days
            if gap_days > 10:  # More than ~7 trading days
                gaps.append((prev_date.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')))
    
    return gaps


def process_symbol(
    symbol: str,
    df_all: pd.DataFrame,
    raw_folder: str,
    start_date: str,
    yf_end: str,
    full_refresh: bool,
    write_sectors: bool,
    max_retries: int = DEFAULT_MAX_RETRIES
) -> Tuple[str, Optional[str], Dict]:
    """
    Process one ticker with validation and retry logic.
    
    Returns:
        (symbol, sector, stats_dict) where stats_dict contains processing statistics
    """
    stats = {
        'status': 'success',
        'rows_added': 0,
        'rows_total': 0,
        'validation_issues': [],
        'data_gaps': [],
        'retries': 0
    }
    
    # 1) Extract this ticker's data with retry
    def get_ticker_data():
        # Handle both dict and DataFrame formats
        if isinstance(df_all, dict):
            df_t = df_all.get(symbol)
        else:
            # DataFrame format (grouped by ticker)
            df_t = df_all.get(symbol) if hasattr(df_all, 'get') else None
        
        if df_t is None or df_t.empty:
            raise ValueError(f"{symbol}: no data returned from download")
        
        # If it's a DataFrame with MultiIndex columns, extract the symbol's data
        if isinstance(df_t, pd.DataFrame) and isinstance(df_t.columns, pd.MultiIndex):
            # MultiIndex columns - extract this symbol's columns
            if symbol in df_t.columns.levels[0] if hasattr(df_t.columns, 'levels') else False:
                df_t = df_t[symbol]
            else:
                raise ValueError(f"{symbol}: symbol not found in MultiIndex columns")
        
        df_t.index = pd.to_datetime(df_t.index)
        
        # Drop rows with all NaN values in OHLCV columns
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Find actual column names (case-insensitive)
        actual_cols = []
        for col in ohlcv_cols:
            if col in df_t.columns:
                actual_cols.append(col)
            elif col.lower() in df_t.columns:
                actual_cols.append(col.lower())
        
        if actual_cols:
            # Drop rows where all OHLCV are NaN
            df_t = df_t.dropna(subset=actual_cols, how='all')
            # Also drop rows where any critical column (Close) is NaN
            close_col = 'Close' if 'Close' in df_t.columns else 'close'
            if close_col in df_t.columns:
                df_t = df_t.dropna(subset=[close_col])
        
        return df_t
    
    try:
        df_t = retry_with_backoff(get_ticker_data, max_retries=max_retries, symbol=symbol)
    except Exception as e:
        stats['status'] = 'failed'
        stats['error'] = str(e)
        return symbol, None, stats
    
    # If dataframe is empty after cleaning, mark as failed
    if df_t.empty:
        stats['status'] = 'failed'
        stats['error'] = "No valid data after cleaning"
        return symbol, None, stats
    
    # 2) Validate data (only warn on critical issues)
    is_valid, issues = validate_ohlcv_data(df_t, symbol)
    stats['validation_issues'] = issues
    
    # Filter out minor issues (NaN warnings if we already cleaned them)
    critical_issues = [i for i in issues if 'NaN' not in i and 'negative' not in i.lower()]
    
    if not is_valid and critical_issues:
        # Only log critical validation issues (not NaN warnings since we clean those)
        logging.warning(f"{symbol}: Data validation issues: {', '.join(critical_issues)}")
    elif not is_valid and len(issues) > 0:
        # Log at debug level for minor issues
        logging.debug(f"{symbol}: Minor data validation issues (cleaned): {', '.join(issues[:2])}")
    
    out_path = Path(raw_folder) / f"{symbol}.csv"
    
    # 3) Full vs incremental update
    if full_refresh:
        df_combined = df_t
        stats['rows_added'] = len(df_combined)
        stats['rows_total'] = len(df_combined)
    else:
        if out_path.exists():
            try:
                existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
                last_date = existing.index.max()
                new_rows = df_t[df_t.index > last_date]
                if new_rows.empty:
                    df_combined = existing
                    stats['rows_added'] = 0
                    stats['rows_total'] = len(existing)
                else:
                    df_combined = pd.concat([existing, new_rows])
                    df_combined = df_combined[~df_combined.index.duplicated(keep="first")]
                    stats['rows_added'] = len(new_rows)
                    stats['rows_total'] = len(df_combined)
            except Exception as e:
                logging.warning(f"{symbol}: Error reading existing file, using new data: {e}")
                df_combined = df_t
                stats['rows_added'] = len(df_combined)
                stats['rows_total'] = len(df_combined)
        else:
            df_combined = df_t
            stats['rows_added'] = len(df_combined)
            stats['rows_total'] = len(df_combined)
    
    # 4) Detect missing data gaps (only significant ones)
    gaps = detect_missing_data(df_combined, start_date, yf_end, symbol)
    stats['data_gaps'] = gaps
    # Only log if there are significant gaps (reduce noise)
    if len(gaps) > 0:
        logging.debug(f"{symbol}: Found {len(gaps)} significant data gap(s)")
    
    # 5) Fetch split factors with retry
    def get_splits():
        ticker_obj = yf.Ticker(symbol)
        actions = ticker_obj.history(
            start=start_date,
            end=yf_end,
            actions=True,
            timeout=REQUEST_TIMEOUT
        )
        return actions
    
    try:
        actions = retry_with_backoff(get_splits, max_retries=max_retries, symbol=symbol)
        splits = actions.get("Stock Splits", pd.Series(dtype=float))
        splits = splits.reindex(df_combined.index, fill_value=0).replace(0, 1.0).astype(float)
        df_combined = df_combined.copy()
        df_combined.loc[:, "split_coefficient"] = splits
    except Exception as e:
        logging.warning(f"{symbol}: could not fetch splits: {e}")
        df_combined = df_combined.copy()
        df_combined.loc[:, "split_coefficient"] = 1.0
    
    # 6) Write per-ticker CSV
    try:
        df_combined.to_csv(out_path, index=True)
    except Exception as e:
        stats['status'] = 'failed'
        stats['error'] = f"Failed to write CSV: {e}"
        return symbol, None, stats
    
    # 7) Optionally fetch sector info with retry
    sector = None
    if write_sectors:
        def get_sector():
            ticker_obj = yf.Ticker(symbol)
            info = ticker_obj.info
            return info.get("sector", "Unknown")
        
        try:
            sector = retry_with_backoff(get_sector, max_retries=max_retries, symbol=symbol)
        except Exception as e:
            logging.warning(f"{symbol}: sector lookup failed: {e}")
            sector = "Unknown"
    
    return symbol, sector, stats


def load_checkpoint(checkpoint_file: Path) -> set:
    """Load completed tickers from checkpoint file."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get('completed', []))
        except Exception as e:
            logging.warning(f"Error loading checkpoint: {e}")
    return set()


def save_checkpoint(checkpoint_file: Path, completed: set):
    """Save completed tickers to checkpoint file."""
    try:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed': list(completed), 'timestamp': datetime.now().isoformat()}, f)
    except Exception as e:
        logging.warning(f"Error saving checkpoint: {e}")


def download_data(
    tickers: list,
    start_date: str,
    end_date: str,
    raw_folder: str,
    chunk_size: int,
    pause: float,
    full_refresh: bool,
    write_sectors: bool,
    max_retries: int = DEFAULT_MAX_RETRIES,
    resume: bool = False
) -> Tuple[List, Dict]:
    """
    Enhanced bulk-download with all improvements.
    
    Returns:
        (sectors_list, statistics_dict)
    """
    os.makedirs(raw_folder, exist_ok=True)
    sectors = []
    total = len(tickers)
    yf_end = end_date or datetime.today().strftime("%Y-%m-%d")
    
    # Resume capability
    checkpoint_file = Path(raw_folder) / '.download_checkpoint.json'
    completed_tickers = set()
    if resume and not full_refresh:
        completed_tickers = load_checkpoint(checkpoint_file)
        if completed_tickers:
            logging.info(f"Resuming: {len(completed_tickers)} tickers already completed")
    
    # Statistics tracking
    stats_summary = {
        'total': total,
        'completed': 0,
        'failed': 0,
        'skipped': 0,
        'up_to_date': 0,
        'new_rows': 0,
        'total_rows': 0,
        'validation_issues': 0,
        'data_gaps': 0,
        'failed_tickers': [],
        'tickers_with_gaps': []
    }
    
    # Adaptive rate limiting
    current_pause = pause
    chunk_times = []
    
    # Progress bar
    pbar = tqdm(total=total, desc="Downloading", unit="ticker")
    
    try:
        for chunk_idx, chunk in enumerate(chunked_list(tickers, chunk_size)):
            # Skip already completed tickers if resuming
            if resume and not full_refresh:
                chunk = [t for t in chunk if t not in completed_tickers]
                if not chunk:
                    continue
            
            chunk_start_time = time.time()
            logging.info(f"Downloading chunk {chunk_idx + 1} of {len(chunk)} tickers...")
            
            # Download chunk with retry - handle partial failures
            def download_chunk():
                try:
                    return yf.download(
                        tickers=chunk,
                        start=start_date,
                        end=yf_end,
                        group_by="ticker",
                        threads=True,
                        progress=False,
                        auto_adjust=False,
                        timeout=REQUEST_TIMEOUT * 2  # Longer timeout for bulk downloads
                    )
                except Exception as e:
                    # If bulk download fails, try individual downloads as fallback
                    logging.warning(f"Bulk download failed for chunk, will try individual downloads: {e}")
                    return None
            
            df_all = None
            try:
                df_all = retry_with_backoff(download_chunk, max_retries=max_retries)
            except Exception as e:
                logging.warning(f"Bulk download failed after retries, trying individual downloads: {e}")
                df_all = None
            
            # Fallback: if bulk download fails, try individual downloads
            if df_all is None or (isinstance(df_all, pd.DataFrame) and df_all.empty):
                logging.info(f"Attempting individual downloads for chunk of {len(chunk)} tickers...")
                df_all = {}
                for sym in chunk:
                    try:
                        def download_single():
                            result = yf.download(
                                tickers=sym,
                                start=start_date,
                                end=yf_end,
                                progress=False,
                                auto_adjust=False,
                                timeout=REQUEST_TIMEOUT
                            )
                            # Handle single ticker result (not grouped)
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                return {sym: result}
                            return {}
                        df_dict = retry_with_backoff(download_single, max_retries=max_retries, symbol=sym)
                        df_all.update(df_dict)
                    except Exception as e:
                        logging.warning(f"{sym}: Individual download also failed: {e}")
                        # Continue with other tickers
            
            # Process each ticker in parallel (only those that have data)
            available_symbols = [s for s in chunk if s in df_all or (isinstance(df_all, dict) and s in df_all)]
            if not available_symbols:
                logging.warning(f"No data available for any ticker in chunk, skipping...")
                stats_summary['failed'] += len(chunk)
                stats_summary['failed_tickers'].extend(chunk)
                pbar.update(len(chunk))
                continue
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(
                        process_symbol,
                        symbol, df_all, raw_folder, start_date, yf_end,
                        full_refresh, write_sectors, max_retries
                    ): symbol
                    for symbol in available_symbols
                }
                
                for fut in as_completed(futures):
                    sym = futures[fut]
                    try:
                        symbol, sector, stats = fut.result()
                        
                        # Update statistics
                        stats_summary['completed'] += 1
                        stats_summary['total_rows'] += stats.get('rows_total', 0)
                        stats_summary['new_rows'] += stats.get('rows_added', 0)
                        
                        if stats.get('status') == 'failed':
                            stats_summary['failed'] += 1
                            stats_summary['failed_tickers'].append(symbol)
                        elif stats.get('rows_added', 0) == 0 and not full_refresh:
                            stats_summary['up_to_date'] += 1
                        
                        if stats.get('validation_issues'):
                            stats_summary['validation_issues'] += len(stats['validation_issues'])
                        
                        # Only count significant gaps (>0 means there were gaps > 10 days)
                        if stats.get('data_gaps') and len(stats['data_gaps']) > 0:
                            stats_summary['data_gaps'] += len(stats['data_gaps'])
                            stats_summary['tickers_with_gaps'].append(symbol)
                        
                        if write_sectors and sector:
                            sectors.append((symbol, sector))
                        
                        # Update checkpoint
                        completed_tickers.add(symbol)
                        if resume:
                            save_checkpoint(checkpoint_file, completed_tickers)
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Completed': stats_summary['completed'],
                            'Failed': stats_summary['failed']
                        })
                        
                    except Exception as e:
                        logging.error(f"{sym}: processing failed: {e}")
                        stats_summary['failed'] += 1
                        stats_summary['failed_tickers'].append(sym)
                        pbar.update(1)
            
            # Adaptive rate limiting
            chunk_time = time.time() - chunk_start_time
            chunk_times.append(chunk_time)
            
            # Adjust pause based on recent chunk times (simple adaptive algorithm)
            if len(chunk_times) >= 3:
                avg_time = sum(chunk_times[-3:]) / 3
                if avg_time > 5.0:  # If chunks are taking too long, increase pause
                    current_pause = min(current_pause * 1.2, MAX_PAUSE)
                elif avg_time < 2.0:  # If chunks are fast, decrease pause
                    current_pause = max(current_pause * 0.9, MIN_PAUSE)
            
            if chunk_idx < len(list(chunked_list(tickers, chunk_size))) - 1:  # Not last chunk
                logging.info(f"Sleeping {current_pause:.1f}s before next chunk (adaptive)...")
                time.sleep(current_pause)
    
    finally:
        pbar.close()
        # Clean up checkpoint if successful
        if stats_summary['failed'] == 0 and not resume:
            if checkpoint_file.exists():
                checkpoint_file.unlink()
    
    return sectors, stats_summary


def write_sectors_csv(
    sectors: list,
    sectors_file: str
) -> None:
    """Write the ticker-to-sector mapping to a CSV file."""
    os.makedirs(Path(sectors_file).parent, exist_ok=True)
    df = pd.DataFrame(sectors, columns=["ticker", "sector"])
    df.to_csv(sectors_file, index=False)
    logging.info(f"Wrote sector mapping to {sectors_file}")


def print_summary(stats: Dict):
    """Print comprehensive summary statistics."""
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Total Tickers:        {stats['total']}")
    print(f"Successfully Completed: {stats['completed']}")
    print(f"Failed:              {stats['failed']}")
    print(f"Up-to-date (skipped): {stats['up_to_date']}")
    print(f"\nData Statistics:")
    print(f"  Total Rows:        {stats['total_rows']:,}")
    print(f"  New Rows Added:    {stats['new_rows']:,}")
    print(f"  Validation Issues: {stats['validation_issues']}")
    print(f"  Data Gaps Found:   {stats['data_gaps']}")
    
    if stats['failed_tickers']:
        print(f"\nFailed Tickers ({len(stats['failed_tickers'])}):")
        for ticker in stats['failed_tickers'][:10]:  # Show first 10
            print(f"  - {ticker}")
        if len(stats['failed_tickers']) > 10:
            print(f"  ... and {len(stats['failed_tickers']) - 10} more")
    
    if stats['tickers_with_gaps']:
        print(f"\nTickers with Data Gaps ({len(stats['tickers_with_gaps'])}):")
        for ticker in stats['tickers_with_gaps'][:10]:  # Show first 10
            print(f"  - {ticker}")
        if len(stats['tickers_with_gaps']) > 10:
            print(f"  ... and {len(stats['tickers_with_gaps']) - 10} more")
    
    print("="*80)


def main() -> None:
    """Parse command-line arguments, download data, and optionally write sector mapping."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced bulk-download price data with validation, retry, and resume"
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
        help="Initial seconds to sleep between each chunk (adaptive)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retry attempts for failed downloads"
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skips already completed tickers)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    with open(args.tickers_file, "r", encoding="utf-8") as fh:
        tickers = [line.strip() for line in fh if line.strip()]

    sectors, stats = download_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_folder=args.raw_folder,
        chunk_size=args.chunk_size,
        pause=args.pause,
        full_refresh=args.full_refresh,
        write_sectors=args.write_sectors,
        max_retries=args.max_retries,
        resume=args.resume
    )

    # Print summary
    print_summary(stats)

    if args.write_sectors:
        write_sectors_csv(sectors, args.sectors_file)
    else:
        logging.info("Skipping sector file write (--no-sectors specified)")


if __name__ == "__main__":
    main()
