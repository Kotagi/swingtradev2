#!/usr/bin/env python3
"""
Test script for NASDAQ short interest download.

This script tests the NASDAQ short interest download functionality
with your API key.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.nasdaq_short_interest import download_nasdaq_short_interest

def main():
    # Get API key from user or environment
    api_key = os.getenv('NASDAQ_DATA_LINK_API_KEY')
    
    if not api_key:
        print("NASDAQ Data Link API key not found in environment.")
        print("\nTo set it, choose one of these options:")
        print("\nOption 1: Set environment variable (Windows PowerShell):")
        print('  $env:NASDAQ_DATA_LINK_API_KEY="your_api_key_here"')
        print("\nOption 2: Set environment variable (Windows CMD):")
        print('  set NASDAQ_DATA_LINK_API_KEY=your_api_key_here')
        print("\nOption 3: Set environment variable (Linux/Mac):")
        print('  export NASDAQ_DATA_LINK_API_KEY="your_api_key_here"')
        print("\nOption 4: Enter it now (temporary, for this session only):")
        api_key = input("Enter your NASDAQ Data Link API key: ").strip()
    
    if not api_key:
        print("Error: API key is required")
        return 1
    
    # Test with a well-known symbol
    test_symbol = "AAPL"
    print(f"\nTesting NASDAQ short interest download for {test_symbol}...")
    print(f"Date range: 2024-01-01 to 2024-12-31")
    
    try:
        df = download_nasdaq_short_interest(
            symbol=test_symbol,
            start_date="2024-01-01",
            end_date="2024-12-31",
            api_key=api_key,
            max_retries=3
        )
        
        if df is not None and not df.empty:
            print(f"\n✓ Success! Downloaded {len(df)} short interest records")
            print(f"\nFirst few records:")
            print(df.head(10))
            print(f"\nLast few records:")
            print(df.tail(10))
            print(f"\nData types:")
            print(df.dtypes)
            print(f"\nDate range: {df.index.min()} to {df.index.max()}")
        else:
            print("\n✗ No data returned. This could mean:")
            print("  - The dataset code format is incorrect")
            print("  - The symbol is not available in NASDAQ Data Link")
            print("  - The date range has no data")
            print("\nYou may need to verify the dataset code on Nasdaq Data Link website.")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✓ Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

