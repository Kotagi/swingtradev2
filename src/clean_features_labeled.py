# src/clean_features_labeled.py

"""
clean_features_labeled.py

Scan all CSV files in data/features_labeled/, drop any rows containing NaNs,
and overwrite the original files with the cleaned data.

Each CSV is expected to have a datetime index in its first column.
"""

import pandas as pd
from pathlib import Path

# —— CONFIGURATION —— #
# Directory containing labeled feature CSVs to clean
DATA_DIR = Path.cwd() / "data" / "features_labeled"


def clean_file(path: Path) -> None:
    """
    Read a single features_labeled CSV, drop rows with any NaNs, and overwrite it.

    Args:
        path (Path): Path to the CSV file to clean.

    Behavior:
        - Reads the CSV with the first column as a datetime index.
        - Drops all rows that contain at least one NaN value.
        - Overwrites the original CSV file with the cleaned DataFrame.
        - Prints a summary of how many rows were dropped and how many remain.
    """
    # Read CSV, parse index as datetime
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Count rows before cleaning
    before = len(df)

    # Drop any row that has at least one NaN value
    df_clean = df.dropna()

    # Count rows after cleaning
    after = len(df_clean)

    # Overwrite the original file with cleaned data
    df_clean.to_csv(path)

    # Print summary of cleaning
    print(f"{path.name}: dropped {before - after} rows → {after} remaining")


def main() -> None:
    """
    Locate all CSV files under DATA_DIR and clean each using clean_file().

    Behavior:
        - Retrieves all '.csv' files in DATA_DIR.
        - If no CSVs are found, prints a message and exits.
        - Otherwise, iterates through each file and calls clean_file().
    """
    # Find all CSV files to clean
    csvs = sorted(DATA_DIR.glob("*.csv"))

    # If none found, notify and exit
    if not csvs:
        print("No CSVs found in", DATA_DIR)
        return

    # Clean each file in turn
    for f in csvs:
        clean_file(f)


if __name__ == "__main__":
    main()
