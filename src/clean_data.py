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

By default, logs only high-level progress and real integrity failures.
Pass `--verbose` (or `-v`) to see every cleaning fix (volume replacements,
row drops, etc.) at DEBUG level.
Usage:
    python src/clean_data.py [--raw-dir data/raw] [--clean-dir data/clean] [--verbose]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# —— CONFIGURATION —— #
# Default input/output directories relative to project root
RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

# After cleaning, these are the expected dtypes on each column
EXPECTED_DTYPES = {
    "volume": "int64",
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
}

# Module-level logger
logger = logging.getLogger(__name__)


def clean_file(path: Path, clean_dir: Path) -> list[str]:
    """
    Clean a single raw CSV file and return any *integrity* issues
    (not cleaning fixes). Cleaning fixes are logged at DEBUG level.
    """
    issues: list[str] = []

    # 1) Read raw CSV
    df = pd.read_csv(path)

    # 2) Rename the first column to 'date'
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})

    # 3) Parse 'date' → datetime, invalid → NaT
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    # 4) Drop rows with invalid dates or exact duplicates
    df = df.dropna(subset=["date"]).drop_duplicates()

    # 5) Set 'date' as index & sort
    df = df.set_index("date").sort_index()

    # 6) Lowercase columns
    df.columns = [col.lower() for col in df.columns]

    # 7a) Coerce & cast 'volume'
    vol = pd.to_numeric(df.get("volume", pd.Series(dtype="float64")), errors="coerce")
    n_inf   = np.isinf(vol).sum()
    n_na    = vol.isna().sum()
    n_nonf  = int(n_inf + n_na)
    if n_nonf:
        # Log fixes at DEBUG only
        logger.debug(f"{path.name}: replaced {n_nonf} non-finite/missing Volume values with 0")
    vol = vol.replace([np.inf, -np.inf], np.nan).fillna(0).astype(int)
    df["volume"] = vol

    # 7b) Cast price columns to float
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # 8) Drop rows with any missing or non-positive OHLCV
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    valid_mask = ohlcv.gt(0).all(axis=1)
    n_bad = int((~valid_mask).sum())
    if n_bad:
        logger.debug(f"{path.name}: dropped {n_bad} rows with missing/non-positive OHLCV")
        df = df.loc[valid_mask]

    # 9) Split-only adjustment (if present)
    if "split_coefficient" in df.columns:
        df["split_coefficient"] = df["split_coefficient"].astype(float)
        df["cum_split"] = df["split_coefficient"].cumprod()
        for col in ["open", "high", "low", "close"]:
            df[col] *= df["cum_split"]
        df["volume"] = (df["volume"] / df["cum_split"]).round().astype(int)
        df = df.drop(columns=["split_coefficient", "cum_split"])

    # 10) Integrity checks (nulls, duplicates, monotonicity, dtypes)

    # 10a) Null dates in index?
    n_null_dates = int(df.index.isnull().sum())
    if n_null_dates:
        issues.append(f"Null dates in index: {n_null_dates}")

    # 10b) Duplicate dates?
    n_dup_dates = int(df.index.duplicated().sum())
    if n_dup_dates:
        issues.append(f"Duplicate dates: {n_dup_dates}")

    # 10c) Monotonic index?
    if not df.index.is_monotonic_increasing:
        issues.append("Index not monotonic")

    # 10d) Dtype mismatches
    actual = df.dtypes.apply(lambda dt: dt.name).to_dict()
    bad_cols = [col for col, exp in EXPECTED_DTYPES.items() if actual.get(col) != exp]
    if bad_cols:
        issues.append(f"Dtype mismatch in: {', '.join(bad_cols)}")

    # 11) Save to Parquet
    df.index.name = "date"
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_path = clean_dir / f"{path.stem}.parquet"
    df.to_parquet(out_path, index=True)

    logging.info(f"{path.name} cleaned → {out_path}")
    return issues

def main() -> None:
    """
    Entry point: parse args, set up logging, clean all CSVs in raw_dir.
    """
    parser = argparse.ArgumentParser(
        description="Clean raw CSVs into Parquet with optional verbose fixes logging."
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

    files = sorted(raw_dir.glob("*.csv"))
    logger.info(f"Found {len(files)} raw CSV files in {raw_dir}")

    failures: list[tuple[str, list[str]]] = []
    for file_path in files:
        logger.info(f"Cleaning {file_path.name}")
        try:
            issues = clean_file(file_path, clean_dir)
            if issues:
                # Real integrity issues reported as warnings
                logger.warning(f"{file_path.name} integrity issues: {issues}")
                failures.append((file_path.name, issues))
            else:
                logger.info(f"{file_path.name} passed integrity checks")
        except Exception as e:
            logger.error(f"Error cleaning {file_path.name}: {e}")
            failures.append((file_path.name, [str(e)]))

    num_ok = len(files) - len(failures)
    logger.info(f"=== Summary: {num_ok}/{len(files)} OK, {len(failures)} failed ===")


if __name__ == "__main__":
    main()
