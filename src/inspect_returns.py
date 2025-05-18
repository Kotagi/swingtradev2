#!/usr/bin/env python3
import pandas as pd

def main():
    # Load the feature-enhanced CSV for AAPL
    df = pd.read_csv('data/feature_tests/AAPL.csv', index_col=0, parse_dates=True)

    # Print the DataFrame columns for debugging
    print("Output CSV columns:", df.columns.tolist())
    print("\nFirst 10 rows of 5d_return and 10d_return:")
    print(df[['5d_return', '10d_return']].head(10))

    # Print non-null counts for sanity check
    print("\nCounts:")
    print("5d_return non-null count:", df['5d_return'].count())
    print("10d_return non-null count:", df['10d_return'].count())

if __name__ == '__main__':
    main()
