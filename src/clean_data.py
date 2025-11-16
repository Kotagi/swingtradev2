#!/usr/bin/env python3
"""
clean_data.py

Cleans raw stock CSVs by:
  - Renaming the first column to 'date'
  - Parsing and coercing dates
  - Dropping invalid or duplicate rows
  - Lowercasing all column names
  - Coercing and casting volume to int and price columns to float
  - Dropping rows with missing or non-positive OHLCV values
  - Applying split-only adjustment to OHLC and volume
  - Running post-cleaning data integrity checks
  - Resetting the index so 'date' remains a column
  - Saving cleaned data as Parquet files

Enhanced with:
  - Parallel processing for faster cleaning
  - Progress bar for visual feedback
  - Resume capability (skips already-cleaned files)
  - Robust column detection (case-insensitive, handles missing columns)
  - Better error handling for edge cases
  - Enhanced summary statistics

By default, logs only high-level progress and real integrity failures.
Pass `--verbose` (or `-v`) to see every cleaning fix (volume replacements,
row drops, etc.) at DEBUG level.

Usage:
    python src/clean_data.py [--raw-dir data/raw] [--clean-dir data/clean] [--verbose] [--resume] [--workers N]
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# —— CONFIGURATION —— #
# Default input/output directories relative to project root
RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

# Default number of worker threads for parallel processing
DEFAULT_WORKERS = 4

# After cleaning, these are the expected dtypes on each column
EXPECTED_DTYPES = {
    "volume": "int64",
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
}

# Required OHLCV columns (case-insensitive matching)
REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

# Module-level logger
logger = logging.getLogger(__name__)


def find_column_case_insensitive(df: pd.DataFrame, col_name: str) -> Optional[str]:
    """
    Find a column name in DataFrame using case-insensitive matching.
    Returns the actual column name if found, None otherwise.
    """
    col_lower = col_name.lower()
    for col in df.columns:
        if col.lower() == col_lower:
            return col
    return None


def clean_file(
    path: Path, 
    clean_dir: Path,
    verbose: bool = False
) -> Tuple[str, Optional[List[str]], Dict]:
    """
    Clean a single raw CSV file and return results.
    
    Returns:
        (symbol, issues, stats) where:
        - symbol: ticker symbol (from filename)
        - issues: list of integrity issues (None if error occurred)
        - stats: dictionary with cleaning statistics
    """
    symbol = path.stem
    stats = {
        'rows_before': 0,
        'rows_after': 0,
        'rows_dropped': 0,
        'volume_replaced': 0,
        'status': 'success'
    }
    issues: List[str] = []
    
    try:
        # 1) Read raw CSV with error handling
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        stats['rows_before'] = len(df)
        
        # 2) Rename the first column to 'date'
        if len(df.columns) == 0:
            raise ValueError("CSV has no columns")
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "date"})
        
        # 3) Parse 'date' → datetime, invalid → NaT
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        
        # 4) Drop rows with invalid dates or exact duplicates
        initial_rows = len(df)
        df = df.dropna(subset=["date"]).drop_duplicates()
        dropped_dates = initial_rows - len(df)
        if dropped_dates > 0 and verbose:
            logger.debug(f"{symbol}: dropped {dropped_dates} rows with invalid dates/duplicates")
        
        # 5) Set 'date' as index & sort
        df = df.set_index("date").sort_index()
        
        # 6) Lowercase columns
        df.columns = [col.lower() for col in df.columns]
        
        # 7) Check for required columns (case-insensitive)
        missing_cols = []
        col_mapping = {}
        for req_col in REQUIRED_COLS:
            actual_col = find_column_case_insensitive(df, req_col)
            if actual_col is None:
                missing_cols.append(req_col)
            else:
                col_mapping[req_col] = actual_col
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # 8a) Coerce & cast 'volume' (handle missing/invalid values)
        vol_col = col_mapping['volume']
        vol = pd.to_numeric(df[vol_col], errors="coerce")
        n_inf = int(np.isinf(vol).sum())
        n_na = int(vol.isna().sum())
        n_nonf = n_inf + n_na
        stats['volume_replaced'] = n_nonf
        
        if n_nonf > 0 and verbose:
            logger.debug(f"{symbol}: replaced {n_nonf} non-finite/missing Volume values with 0")
        
        vol = vol.replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
        df['volume'] = vol
        
        # 8b) Cast price columns to float (handle errors gracefully)
        for req_col in ["open", "high", "low", "close"]:
            actual_col = col_mapping[req_col]
            try:
                df[req_col] = pd.to_numeric(df[actual_col], errors="coerce").astype(float)
            except Exception as e:
                raise ValueError(f"Failed to convert {req_col} to float: {e}")
        
        # 9) Drop rows with any missing or non-positive OHLCV
        ohlcv = df[REQUIRED_COLS]
        valid_mask = ohlcv.gt(0).all(axis=1) & ohlcv.notna().all(axis=1)
        n_bad = int((~valid_mask).sum())
        stats['rows_dropped'] = n_bad
        
        if n_bad > 0 and verbose:
            logger.debug(f"{symbol}: dropped {n_bad} rows with missing/non-positive OHLCV")
        
        df = df.loc[valid_mask]
        stats['rows_after'] = len(df)
        
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        # 10) Split-only adjustment (if present)
        if "split_coefficient" in df.columns:
            try:
                df["split_coefficient"] = pd.to_numeric(df["split_coefficient"], errors="coerce").astype(float)
                df["cum_split"] = df["split_coefficient"].cumprod()
                for col in ["open", "high", "low", "close"]:
                    df[col] *= df["cum_split"]
                df["volume"] = (df["volume"] / df["cum_split"]).round().astype(int)
                df = df.drop(columns=["split_coefficient", "cum_split"])
            except Exception as e:
                if verbose:
                    logger.debug(f"{symbol}: error applying split adjustment: {e}")
        
        # 11) Integrity checks (nulls, duplicates, monotonicity, dtypes)
        
        # 11a) Null dates in index?
        n_null_dates = int(df.index.isnull().sum())
        if n_null_dates:
            issues.append(f"Null dates in index: {n_null_dates}")
        
        # 11b) Duplicate dates?
        n_dup_dates = int(df.index.duplicated().sum())
        if n_dup_dates:
            issues.append(f"Duplicate dates: {n_dup_dates}")
        
        # 11c) Monotonic index?
        if not df.index.is_monotonic_increasing:
            issues.append("Index not monotonic")
        
        # 11d) Dtype mismatches
        actual = df.dtypes.apply(lambda dt: dt.name).to_dict()
        bad_cols = [col for col, exp in EXPECTED_DTYPES.items() if actual.get(col) != exp]
        if bad_cols:
            issues.append(f"Dtype mismatch in: {', '.join(bad_cols)}")
        
        # 12) Save to Parquet
        df.index.name = "date"
        clean_dir.mkdir(parents=True, exist_ok=True)
        out_path = clean_dir / f"{symbol}.parquet"
        
        try:
            df.to_parquet(out_path, index=True)
        except Exception as e:
            raise ValueError(f"Failed to save Parquet file: {e}")
        
        if verbose:
            logger.debug(f"{symbol}: cleaned {stats['rows_before']} → {stats['rows_after']} rows")
        
        return symbol, issues, stats
        
    except Exception as e:
        stats['status'] = 'failed'
        stats['error'] = str(e)
        return symbol, None, stats


def process_file_wrapper(args: Tuple) -> Tuple[str, Optional[List[str]], Dict]:
    """Wrapper function for parallel processing."""
    path, clean_dir, verbose = args
    return clean_file(path, clean_dir, verbose)


def main() -> None:
    """
    Entry point: parse args, set up logging, clean all CSVs in raw_dir.
    """
    parser = argparse.ArgumentParser(
        description="Clean raw CSVs into Parquet with parallel processing, progress tracking, and resume capability."
    )
    parser.add_argument(
        "--raw-dir", "-r",
        default=str(RAW_DIR),
        help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--clean-dir", "-c",
        default=str(CLEAN_DIR),
        help="Directory where cleaned Parquet files will be written"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed cleaning steps (DEBUG logs)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that have already been cleaned (resume from previous run)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})"
    )
    args = parser.parse_args()
    
    # Configure logging level & format
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    if args.verbose:
        logger.debug("Verbose mode enabled: detailed cleaning logs will appear")
    
    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)
    
    if not raw_dir.exists():
        logger.error(f"Raw directory does not exist: {raw_dir}")
        return
    
    # Find all CSV files
    files = sorted(raw_dir.glob("*.csv"))
    logger.info(f"Found {len(files)} raw CSV files in {raw_dir}")
    
    if len(files) == 0:
        logger.warning("No CSV files found to clean")
        return
    
    # Resume capability: skip already-cleaned files
    if args.resume:
        existing_parquets = {f.stem for f in clean_dir.glob("*.parquet") if clean_dir.exists()}
        files = [f for f in files if f.stem not in existing_parquets]
        skipped = len(existing_parquets)
        if skipped > 0:
            logger.info(f"Resume mode: skipping {skipped} already-cleaned files")
        if len(files) == 0:
            logger.info("All files already cleaned. Use without --resume to re-clean.")
            return
    
    # Prepare arguments for parallel processing
    clean_dir.mkdir(parents=True, exist_ok=True)
    file_args = [(f, clean_dir, args.verbose) for f in files]
    
    # Track results
    failures: List[Tuple[str, List[str]]] = []
    all_stats: Dict[str, Dict] = {}
    start_time = time.time()
    
    # Process files in parallel with progress bar
    logger.info(f"Cleaning {len(files)} files with {args.workers} workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file_wrapper, args): args[0] 
            for args in file_args
        }
        
        # Process with progress bar
        with tqdm(total=len(files), desc="Cleaning", unit="file") as pbar:
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    symbol, issues, stats = future.result()
                    all_stats[symbol] = stats
                    
                    if stats['status'] == 'failed':
                        error_msg = stats.get('error', 'Unknown error')
                        logger.error(f"{symbol}: {error_msg}")
                        failures.append((symbol, [error_msg]))
                    elif issues:
                        logger.warning(f"{symbol}: integrity issues: {', '.join(issues)}")
                        failures.append((symbol, issues))
                    else:
                        if args.verbose:
                            logger.debug(f"{symbol}: passed integrity checks")
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'OK': len(files) - len(failures) - pbar.n + pbar.last_print_n,
                        'Failed': len(failures)
                    })
                except Exception as e:
                    symbol = file_path.stem
                    logger.error(f"{symbol}: unexpected error: {e}")
                    failures.append((symbol, [str(e)]))
                    all_stats[symbol] = {'status': 'failed', 'error': str(e)}
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Enhanced summary statistics
    num_ok = len(files) - len(failures)
    total_rows_before = sum(s.get('rows_before', 0) for s in all_stats.values())
    total_rows_after = sum(s.get('rows_after', 0) for s in all_stats.values())
    total_rows_dropped = sum(s.get('rows_dropped', 0) for s in all_stats.values())
    total_volume_replaced = sum(s.get('volume_replaced', 0) for s in all_stats.values())
    
    logger.info("=" * 70)
    logger.info("=== CLEANING SUMMARY ===")
    logger.info(f"Files processed: {len(files)}")
    logger.info(f"Successfully cleaned: {num_ok}")
    logger.info(f"Failed: {len(failures)}")
    logger.info(f"Total rows before: {total_rows_before:,}")
    logger.info(f"Total rows after: {total_rows_after:,}")
    logger.info(f"Total rows dropped: {total_rows_dropped:,} ({100*total_rows_dropped/max(total_rows_before,1):.2f}%)")
    logger.info(f"Volume values replaced: {total_volume_replaced:,}")
    logger.info(f"Processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    logger.info(f"Average time per file: {elapsed_time/max(len(files),1):.3f} seconds")
    
    if failures:
        logger.warning(f"\nFailed files ({len(failures)}):")
        for symbol, error_list in failures[:10]:  # Show first 10
            logger.warning(f"  {symbol}: {', '.join(error_list)}")
        if len(failures) > 10:
            logger.warning(f"  ... and {len(failures) - 10} more")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
