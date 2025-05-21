#!/usr/bin/env python3
"""
clean_features_labeled.py

Cleans labeled feature Parquet files by:
  - Loading each Parquet from data/features_labeled/
  - Dropping any rows containing NaNs
  - Overwriting the original Parquet files with the cleaned data
  - Printing a summary of rows dropped vs. remaining

Each Parquet file is expected to have a datetime index in its first column.
"""

import pandas as pd
from pathlib import Path

# —— CONFIGURATION —— #
# Directory containing labeled feature Parquet files
DATA_DIR = Path.cwd() / "data" / "features_labeled"


def clean_file(path: Path) -> None:
    """
    Clean a single labeled feature Parquet file in-place.

    Steps:
      1. Read the Parquet file into a DataFrame (preserving datetime index).
      2. Count rows before cleaning.
      3. Drop any row with at least one NaN.
      4. Count rows after cleaning.
      5. Overwrite the original Parquet file with cleaned data.
      6. Print a summary showing dropped vs. remaining rows.

    Args:
        path: Path to the Parquet file to clean.
    """
    # 1) Load the Parquet file
    df = pd.read_parquet(path)

    # 2) Count rows before dropping NaNs
    before = len(df)

    # 3) Drop rows containing any NaN values
    df_clean = df.dropna()

    # 4) Count rows after cleaning
    after = len(df_clean)

    # 5) Overwrite with the cleaned DataFrame
    df_clean.to_parquet(path, index=True)

    # 6) Print summary for quick sanity check
    print(f"{path.name}: dropped {before - after} rows → {after} remaining")


def main() -> None:
    """
    Locate all Parquet files under DATA_DIR and clean each using clean_file().

    Behavior:
      - Retrieves all '*.parquet' files in DATA_DIR.
      - If none are found, prints a message and exits.
      - Otherwise, iterates through each file and calls clean_file().
    """
    # Find all Parquet files to clean
    files = sorted(DATA_DIR.glob("*.parquet"))

    # If none found, notify and exit
    if not files:
        print("No Parquet files found in", DATA_DIR)
        return

    # Clean each file one by one
    for f in files:
        clean_file(f)


if __name__ == "__main__":
    main()
