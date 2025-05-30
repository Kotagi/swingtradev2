#!/usr/bin/env python3
"""
clean_data.py

Cleans raw stock CSVs by:
  - Renaming the first column (often 'price') to 'date'
  - Parsing and coercing dates
  - Dropping invalid or duplicate rows
  - Lowercasing all column names
  - Casting volume to int and price columns to float
  - Running data integrity checks (nulls, duplicates, monotonic index, dtype validation)
  - Saving cleaned data as Parquet files in specified clean directory (default data/clean/)
Usage:
    python src/clean_data.py --raw-dir data/raw --clean-dir data/clean
"""

import logging
from pathlib import Path
import argparse

import pandas as pd

# —— CONFIGURATION —— #
# Default directories for input raw data and output cleaned data
RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

# Expected dtypes for cleaned columns
EXPECTED_DTYPES = {
    "volume": "int64",
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
}


def clean_file(path: Path, clean_dir: Path = CLEAN_DIR) -> list:
    """
    Clean a single raw CSV file and return any integrity issues.

    Steps:
      1. Read the CSV into a DataFrame.
      2. Rename the first column to 'date' if not already.
      3. Parse 'date' column to datetime, dropping parse failures.
      4. Drop duplicate rows.
      5. Set 'date' as index and sort chronologically.
      6. Lowercase all column names for consistency.
      7. Cast 'volume' to int and price columns to float.
      8. Check for null dates, duplicate dates, non-monotonic index,
         and unexpected data types.
      9. Save the cleaned DataFrame to Parquet in `clean_dir`.
     10. Return list of any integrity issues encountered.
    """
    # Read raw data
    df = pd.read_csv(path)

    # Rename first column to 'date' if needed
    first_col = df.columns[0]
    if first_col.lower() != "date":
        df = df.rename(columns={first_col: "date"})

    # Parse 'date' column to datetime, coercing errors to NaT
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    # 1) Drop rows with invalid dates or exact duplicates
    df = df.dropna(subset=["date"]).drop_duplicates()

    # 2) Set 'date' as index and ensure it's sorted
    df = df.set_index("date").sort_index()

    # 3) Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # 4) Cast expected types
    df["volume"] = df["volume"].astype(int)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # Initialize list to collect any integrity issues
    issues = []

    # 5) Null dates in index?
    null_dates = df.index.isnull().sum()
    if null_dates:
        issues.append(f"Null dates: {null_dates}")

    # 6) Duplicate dates in index?
    dup_dates = df.index.duplicated().sum()
    if dup_dates:
        issues.append(f"Duplicate dates: {dup_dates}")

    # 7) Is index monotonic?
    if not df.index.is_monotonic_increasing:
        issues.append("Index not monotonic")

    # 8) Data type mismatches
    actual_dtypes = df.dtypes.apply(lambda dt: dt.name).to_dict()
    bad_types = [
        col for col, exp in EXPECTED_DTYPES.items()
        if actual_dtypes.get(col) != exp
    ]
    if bad_types:
        issues.append(f"Dtype mismatch in: {', '.join(bad_types)}")

    # Ensure output directory exists
    clean_dir.mkdir(parents=True, exist_ok=True)

    # 9) Save cleaned data as a Parquet file
    parquet_path = clean_dir / f"{path.stem}.parquet"
    df.to_parquet(parquet_path, index=True)
    logging.info(f"{path.name} cleaned → {parquet_path}")

    return issues


def main() -> None:
    """
    Entry point: cleans all CSVs under a raw directory and logs a summary.

    Behavior:
      - Parses CLI arguments for raw and clean directories.
      - Initializes logging.
      - Processes each '*.csv' file in the raw directory via clean_file().
      - Logs per-file integrity outcomes.
      - At end, logs a summary of total files, passes, and failures.
    """
    parser = argparse.ArgumentParser(
        description="Clean raw CSVs into Parquet files with integrity checks."
    )
    parser.add_argument(
        "--raw-dir", default=str(RAW_DIR),
        help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--clean-dir", default=str(CLEAN_DIR),
        help="Directory where cleaned Parquet files will be written"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    clean_dir = Path(args.clean_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    files = sorted(raw_dir.glob("*.csv"))
    total_files = len(files)
    logging.info(f"Found {total_files} raw CSV files in {raw_dir}")

    failures = []

    # Process each raw CSV
    for file_path in files:
        logging.info(f"Cleaning {file_path.name}")
        try:
            file_issues = clean_file(file_path, clean_dir)
            if file_issues:
                logging.warning(f"{file_path.name} issues: {file_issues}")
                failures.append((file_path.name, file_issues))
            else:
                logging.info(f"{file_path.name} passed integrity checks")
        except Exception as e:
            logging.error(f"Error cleaning {file_path.name}: {e}")
            failures.append((file_path.name, [str(e)]))

    # Summarize results
    passed = total_files - len(failures)
    logging.info("=== Clean Data Summary ===")
    if failures:
        logging.warning(
            f"{passed}/{total_files} files cleaned successfully, "
            f"{len(failures)} failed:"
        )
        for fname, issues in failures:
            logging.warning(f" - {fname}: {issues}")
    else:
        logging.info(f"All {total_files} files cleaned successfully")


if __name__ == "__main__":
    main()
