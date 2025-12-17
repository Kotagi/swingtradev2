#!/usr/bin/env python3
"""
Inspect SEC EDGAR data for a ticker to understand float vs outstanding shares.

Usage:
    python inspect_sec_data.py AMZN 2020-06-30
"""

import sys
import requests
import pandas as pd
import yfinance as yf
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.sec_edgar_shares import get_cik_from_ticker

USER_AGENT = os.environ.get("SEC_EDGAR_USER_AGENT", "SwingTradeApp (contact@example.com)")

def inspect_ticker(ticker: str, date: str = None):
    """Inspect SEC EDGAR data for a ticker."""
    
    print("=" * 80)
    print(f"Inspecting SEC EDGAR Data for {ticker}")
    print("=" * 80)
    
    # Get CIK
    cik = get_cik_from_ticker(ticker)
    if not cik:
        print(f"ERROR: Could not find CIK for {ticker}")
        return
    
    print(f"\nCIK: {cik}")
    print(f"API URL: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    
    # Download company facts
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {'User-Agent': USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"ERROR: Failed to download data: {e}")
        return
    
    # Extract shares outstanding
    print("\n" + "=" * 80)
    print("1. SHARES OUTSTANDING")
    print("=" * 80)
    print("Path: facts.us-gaap.CommonStockSharesOutstanding")
    
    facts = data.get('facts', {})
    us_gaap = facts.get('us-gaap', {})
    shares_data = us_gaap.get('CommonStockSharesOutstanding', {})
    
    if not shares_data:
        print("  NOT FOUND")
    else:
        units = shares_data.get('units', {})
        print(f"\n  Found {len(units)} unit type(s)")
        
        for unit_type, unit_data in units.items():
            if isinstance(unit_data, list):
                print(f"\n  Unit type: {unit_type}")
                print(f"  Total entries: {len(unit_data)}")
                
                # Filter by date if provided
                if date:
                    target_date = pd.to_datetime(date)
                    filtered = [e for e in unit_data if pd.to_datetime(e.get('end', '')) == target_date]
                    if filtered:
                        print(f"\n  Entries for {date}:")
                        for entry in filtered:
                            print(f"    Date: {entry.get('end')}")
                            print(f"    Value: {entry.get('val'):,.0f} shares")
                            print(f"    Form: {entry.get('form')}")
                            print(f"    Frame: {entry.get('frame')}")
                    else:
                        print(f"\n  No entries found for {date}")
                        print(f"  Showing 5 most recent entries:")
                        for entry in unit_data[-5:]:
                            print(f"    {entry.get('end')}: {entry.get('val'):,.0f} shares ({entry.get('form')})")
                else:
                    print(f"\n  Showing 5 most recent entries:")
                    for entry in unit_data[-5:]:
                        print(f"    {entry.get('end')}: {entry.get('val'):,.0f} shares ({entry.get('form')})")
    
    # Extract EntityPublicFloat
    print("\n" + "=" * 80)
    print("2. ENTITY PUBLIC FLOAT (Market Value)")
    print("=" * 80)
    print("Path: facts.dei.EntityPublicFloat")
    print("Note: This is MARKET VALUE in USD, not number of shares!")
    
    dei = facts.get('dei', {})
    public_float_data = dei.get('EntityPublicFloat', {})
    
    if not public_float_data:
        print("  NOT FOUND")
    else:
        units = public_float_data.get('units', {})
        print(f"\n  Found {len(units)} unit type(s)")
        
        for unit_type, unit_data in units.items():
            if isinstance(unit_data, list):
                print(f"\n  Unit type: {unit_type}")
                print(f"  Total entries: {len(unit_data)}")
                
                # Filter by date if provided
                if date:
                    target_date = pd.to_datetime(date)
                    filtered = [e for e in unit_data if pd.to_datetime(e.get('end', '')) == target_date]
                    if filtered:
                        print(f"\n  Entries for {date}:")
                        for entry in filtered:
                            print(f"    Date: {entry.get('end')}")
                            print(f"    Value: ${entry.get('val'):,.0f} (market value)")
                            print(f"    Form: {entry.get('form')}")
                            print(f"    Frame: {entry.get('frame')}")
                            
                            # Calculate float shares using stock price
                            report_date = pd.to_datetime(entry.get('end'))
                            public_float_value = entry.get('val')
                            
                            # Get stock price
                            ticker_obj = yf.Ticker(ticker)
                            stock_data = ticker_obj.history(
                                start=report_date - pd.Timedelta(days=5),
                                end=report_date + pd.Timedelta(days=5)
                            )
                            
                            if not stock_data.empty:
                                # Normalize timezone
                                if stock_data.index.tz is not None:
                                    stock_data.index = stock_data.index.tz_localize(None)
                                
                                # Get price on or before report date
                                price_data = stock_data[stock_data.index <= report_date]
                                if price_data.empty:
                                    price_data = stock_data[stock_data.index >= report_date]
                                
                                if not price_data.empty:
                                    closest_date = price_data.index[-1] if price_data.index[0] <= report_date else price_data.index[0]
                                    stock_price = price_data.loc[closest_date, 'Close']
                                    
                                    print(f"\n    Stock Price Calculation:")
                                    print(f"      Date used: {closest_date.date()}")
                                    print(f"      Stock price: ${stock_price:.2f}")
                                    print(f"      Calculated float shares: {public_float_value / stock_price:,.0f}")
                                    
                                    # Compare with shares outstanding
                                    if shares_data:
                                        shares_units = shares_data.get('units', {})
                                        for su_type, su_data in shares_units.items():
                                            if isinstance(su_data, list):
                                                for se in su_data:
                                                    if pd.to_datetime(se.get('end', '')) == report_date:
                                                        shares_out = se.get('val')
                                                        print(f"\n    Comparison:")
                                                        print(f"      Shares Outstanding: {shares_out:,.0f}")
                                                        print(f"      Calculated Float: {public_float_value / stock_price:,.0f}")
                                                        print(f"      Difference: {(public_float_value / stock_price) - shares_out:,.0f}")
                                                        print(f"      Float > Outstanding? {(public_float_value / stock_price) > shares_out}")
                    else:
                        print(f"\n  No entries found for {date}")
                        print(f"  Showing 5 most recent entries:")
                        for entry in unit_data[-5:]:
                            print(f"    {entry.get('end')}: ${entry.get('val'):,.0f} ({entry.get('form')})")
                else:
                    print(f"\n  Showing 5 most recent entries:")
                    for entry in unit_data[-5:]:
                        print(f"    {entry.get('end')}: ${entry.get('val'):,.0f} ({entry.get('form')})")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nTo research further:")
    print(f"1. Visit SEC EDGAR: https://www.sec.gov/edgar/searchedgar/companysearch.html")
    print(f"2. Search for {ticker}")
    print(f"3. Find the 10-Q or 10-K filing for the date in question")
    print(f"4. Check the 'Cover Page' section for:")
    print("   - Shares Outstanding")
    print("   - Public Float (both value and shares)")
    print(f"\n5. Direct API URL: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_sec_data.py <TICKER> [DATE]")
        print("Example: python inspect_sec_data.py AMZN 2020-06-30")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    date = sys.argv[2] if len(sys.argv) > 2 else None
    
    inspect_ticker(ticker, date)

