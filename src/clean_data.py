#!/usr/bin/env python3
import logging
from pathlib import Path

import pandas as pd

# adjust these if you run from elsewhere
RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

# Expected dtypes after cleaning
EXPECTED_DTYPES = {
    "volume": "int64",
    "open":   "float64",
    "high":   "float64",
    "low":    "float64",
    "close":  "float64"
}

def clean_file(path: Path):
    """
    Clean a single CSV file. Returns a list of integrity issues (empty if none).
    """
    df = pd.read_csv(path)

    # Rename first column to 'date'
    first_col = df.columns[0]
    if first_col.lower() != "date":
        df = df.rename(columns={first_col: "date"})

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

    # Drop bad rows & duplicates
    df = df.dropna(subset=["date"]).drop_duplicates()

    # Set index and sort
    df = df.set_index("date").sort_index()

    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # Cast types
    df["volume"] = df["volume"].astype(int)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)

    # Integrity checks
    issues = []

    nat_count = df.index.isnull().sum()
    if nat_count:
        issues.append(f"Null dates: {nat_count}")

    dup_count = df.index.duplicated().sum()
    if dup_count:
        issues.append(f"Duplicate dates: {dup_count}")

    if not df.index.is_monotonic_increasing:
        issues.append("Index not monotonic")

    # Dtype mismatches
    dtypes = df.dtypes.to_dict()
    mismatches = [col for col, exp in EXPECTED_DTYPES.items() if dtypes.get(col) != exp]
    if mismatches:
        issues.append(f"Dtype mismatch in: {', '.join(mismatches)}")

    # Ensure index name
    df.index.name = "date"

    # Save cleaned file
    output_path = CLEAN_DIR / path.name
    df.to_csv(output_path)

    return issues

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DIR.glob("*.csv"))
    logging.info(f"Found {len(files)} files to clean in {RAW_DIR}")

    failures = []

    for f in files:
        logging.info(f"Cleaning {f.name}")
        try:
            issues = clean_file(f)
            if issues:
                logging.warning(f"{f.name} integrity issues: {'; '.join(issues)}")
                failures.append((f.name, issues))
            else:
                logging.info(f"{f.name} passes all integrity checks")
        except Exception as e:
            logging.error(f"Failed cleaning {f.name}: {e}")
            failures.append((f.name, [str(e)]))

    # Summary
    total = len(files)
    failed = len(failures)
    passed = total - failed

    logging.info("=== Clean Data Summary ===")
    if failed == 0:
        logging.info(f"All {total} files passed integrity checks.")
    else:
        logging.warning(f"{failed}/{total} files failed integrity checks:")
        for name, issues in failures:
            logging.warning(f" - {name}: {', '.join(issues)}")

if __name__ == "__main__":
    main()

