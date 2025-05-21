# src/clean_data.py

"""
clean_data.py

Cleans raw stock CSVs by:
  - Renaming the first column (often 'price') to 'date'
  - Parsing and coercing dates
  - Dropping invalid or duplicate rows
  - Lowercasing all column names
  - Casting volume to int and price columns to float
  - Running data integrity checks (nulls, duplicates, monotonic index, dtype validation)
  - Saving cleaned CSVs to data/clean/
  - Logging per-file outcomes and a summary of failures
"""

import logging
from pathlib import Path

import pandas as pd

# Directories for input raw data and output cleaned data
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

# Expected data types for cleaned columns
EXPECTED_DTYPES = {
    "volume": "int64",
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64",
}


def clean_file(path: Path) -> list:
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
      9. Save the cleaned DataFrame to CLEAN_DIR with the same filename.

    Args:
        path (Path): Path to the raw CSV file.

    Returns:
        List[str]: Descriptions of any integrity issues (empty if none).
    """
    # Read raw data
    df = pd.read_csv(path)

    # Rename first column to 'date'
    first_col = df.columns[0]
    if first_col.lower() != "date":
        df = df.rename(columns={first_col: "date"})

    # Parse 'date' column to datetime, coercing errors to NaT
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    # Drop rows with invalid dates or exact duplicates
    df = df.dropna(subset=["date"]).drop_duplicates()

    # Set 'date' as index and ensure it's sorted
    df = df.set_index("date").sort_index()

    # Standardize column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Cast expected types
    df["volume"] = df["volume"].astype(int)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # Gather integrity checks
    issues = []

    # 1) Null dates in index?
    null_dates = df.index.isnull().sum()
    if null_dates:
        issues.append(f"Null dates: {null_dates}")

    # 2) Duplicate dates in index?
    dup_dates = df.index.duplicated().sum()
    if dup_dates:
        issues.append(f"Duplicate dates: {dup_dates}")

    # 3) Is index monotonic?
    if not df.index.is_monotonic_increasing:
        issues.append("Index not monotonic")

    # 4) Data type mismatches
    actual_dtypes = df.dtypes.apply(lambda dt: dt.name).to_dict()
    bad_types = [col for col, exp in EXPECTED_DTYPES.items()
                 if actual_dtypes.get(col) != exp]
    if bad_types:
        issues.append(f"Dtype mismatch in: {', '.join(bad_types)}")

    # Ensure index is named correctly for CSV output
    df.index.name = "date"

    # Save cleaned data
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DIR / path.name)

    return issues


def main() -> None:
    """
    Entry point: cleans all CSVs under RAW_DIR and logs a summary.

    Behavior:
      - Initializes logging.
      - Processes each '*.csv' file in RAW_DIR via clean_file().
      - Logs per-file integrity outcomes.
      - At end, logs a summary of total files, passes, and failures.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))
    total_files = len(files)
    logging.info(f"Found {total_files} files to clean in {RAW_DIR}")

    failures = []

    # Process each raw CSV
    for file_path in files:
        logging.info(f"Cleaning {file_path.name}")
        try:
            file_issues = clean_file(file_path)
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
        logging.warning(f"{passed}/{total_files} files cleaned successfully, "
                        f"{len(failures)} failed:")
        for fname, issues in failures:
            logging.warning(f" - {fname}: {issues}")
    else:
        logging.info(f"All {total_files} files cleaned successfully")


if __name__ == "__main__":
    main()

