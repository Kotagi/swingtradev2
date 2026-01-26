#!/usr/bin/env python3
"""
inspect_parquet.py

Utility script to load a Parquet file, display summary information, and
export the full contents to a CSV for manual inspection.

Usage:
    python src/inspect_parquet.py /path/to/data/clean/AAPL.parquet

This will:
  1. Print shape, index info, dtypes, null counts, head & tail.
  2. Write the full DataFrame to CSV under outputs/inspections/AAPL.csv.
"""

import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Inspect a Parquet file and export it to CSV."
    )
    parser.add_argument(
        "parquet_file",
        help="Path to the Parquet file to inspect (e.g. data/clean/AAPL.parquet)."
    )
    parser.add_argument(
        "--export-dir", "-e",
        default=Path(__file__).parent.parent / "outputs" / "inspections",
        help="Directory where the full CSV will be saved (default: outputs/inspections)"
    )
    args = parser.parse_args()

    in_path = Path(args.parquet_file)
    if not in_path.exists():
        print(f"Error: file not found: {in_path}")
        return

    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(in_path)

    # Display summary information for quick sanity checks
    print(f"\n=== Inspecting {in_path} ===\n")
    print("Shape:", df.shape)
    print("Index name:", df.index.name)
    print("Date range:", df.index.min(), "to", df.index.max())
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nNull counts per column:")
    print(df.isna().sum())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nLast 5 rows:")
    print(df.tail().to_string())
    print("\n=== End of inspection ===\n")

    # Ensure the export directory exists
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export the full DataFrame to CSV for manual inspection
    out_csv = export_dir / f"{in_path.stem}.csv"
    df.to_csv(out_csv, index=True)
    print(f"Exported full CSV to {out_csv}\n")

if __name__ == "__main__":
    main()

