#!/usr/bin/env python3
import logging
from pathlib import Path

import pandas as pd

# adjust these if you run from elsewhere
RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CLEAN_DIR = Path(__file__).parent.parent / "data" / "clean"

def clean_file(path: Path):
    # read
    df = pd.read_csv(path)

    # identify the date column (first column)
    date_col = df.columns[0]

    # parse dates with explicit format
    df[date_col] = pd.to_datetime(
        df[date_col],
        format="%Y-%m-%d",
        errors="coerce"
    )

    # drop rows where date failed + full‐row duplicates
    df = df.dropna(subset=[date_col]).drop_duplicates()

    # set date index and sort
    df = df.set_index(date_col).sort_index()

    # cast types
    df["Volume"] = df["Volume"].astype(int)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)

    # collect stats
    nat_count = df.index.isnull().sum()
    dup_count = df.index.duplicated().sum()
    is_mono   = df.index.is_monotonic_increasing
    dtypes    = df.dtypes.to_dict()

    # log them
    logging.info(
        f"{path.name} | NaT={nat_count} | Dups={dup_count} | "
        f"Monotonic={is_mono} | dtypes={dtypes}"
    )

    # save to clean folder
    output_path = CLEAN_DIR / path.name
    df.to_csv(output_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DIR.glob("*.csv"))
    logging.info(f"Found {len(files)} files to clean in {RAW_DIR}")

    for f in files:
        logging.info(f"Cleaning {f.name}")
        try:
            clean_file(f)
            logging.info(f"Cleaned data saved for {f.stem}: {CLEAN_DIR/f.name}")
        except Exception as e:
            logging.error(f"Failed cleaning {f.name}: {e}")


if __name__ == "__main__":
    main()
