# Float Shares vs Shares Outstanding - Data Sources Explained

## Overview

This document explains where both data points come from and how they're calculated, so you can research why float shares might exceed shares outstanding.

## 1. Shares Outstanding

### Source
- **API**: SEC EDGAR Company Facts API
- **Endpoint**: `https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json`
- **Field Path**: `facts.us-gaap.CommonStockSharesOutstanding`
- **Module**: `src/sec_edgar_shares.py`

### How It Works
1. Get CIK (Central Index Key) for the ticker symbol
2. Download company facts JSON from SEC EDGAR
3. Navigate to: `facts.us-gaap.CommonStockSharesOutstanding`
4. Extract all entries with:
   - `end`: Reporting date (fiscal period end)
   - `val`: Number of shares outstanding
   - `form`: Filing type (10-K, 10-Q, etc.)
   - `frame`: Fiscal period frame

### Example for AMZN
```json
{
  "facts": {
    "us-gaap": {
      "CommonStockSharesOutstanding": {
        "units": {
          "shares": [
            {
              "end": "2020-06-30",
              "val": 501000000,
              "form": "10-Q",
              "frame": "CY2020Q2"
            }
          ]
        }
      }
    }
  }
}
```

### Data Characteristics
- **Frequency**: Quarterly (10-Q) and Annual (10-K)
- **Units**: Shares (raw count)
- **Date**: Fiscal period end date
- **Direct Value**: Yes, this is the actual number of shares

---

## 2. Float Shares

### Source
- **API**: SEC EDGAR Company Facts API (same endpoint)
- **Endpoint**: `https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json`
- **Field Path**: `facts.dei.EntityPublicFloat`
- **Module**: `src/sec_edgar_float.py`

### How It Works
1. Get CIK for the ticker symbol
2. Download company facts JSON from SEC EDGAR
3. Navigate to: `facts.dei.EntityPublicFloat`
4. Extract all entries with:
   - `end`: Reporting date (fiscal period end)
   - `val`: **Market value** of public float (in USD)
   - `form`: Filing type (10-K, 10-Q, etc.)
   - `frame`: Fiscal period frame
5. **Calculate Float Shares**: `EntityPublicFloat (USD) / Stock Price (USD per share)`

### Example for AMZN
```json
{
  "facts": {
    "dei": {
      "EntityPublicFloat": {
        "units": {
          "USD": [
            {
              "end": "2020-06-30",
              "val": 23846135567,  // $23.8 billion (market value)
              "form": "10-Q",
              "frame": "CY2020Q2"
            }
          ]
        }
      }
    }
  }
}
```

### Stock Price Source
- **Source**: Yahoo Finance (via `yfinance` library)
- **Method**: `yf.Ticker(symbol).history(start_date, end_date)`
- **Price Used**: Close price on or before the reporting date
- **Calculation**: `float_shares = EntityPublicFloat / stock_price`

### Data Characteristics
- **Frequency**: Quarterly (10-Q) and Annual (10-K)
- **Units**: Market value in USD (not shares directly)
- **Date**: Fiscal period end date
- **Calculated Value**: Yes, derived from market value / stock price

---

## The Problem: Why Float > Outstanding?

### Possible Causes

1. **Date Mismatch**
   - EntityPublicFloat is reported as of fiscal period end (e.g., 2020-06-30)
   - Stock price might be from a different date
   - If stock price dropped after the reporting date, float shares calculation would be higher

2. **Stock Price Issues**
   - Using wrong price (pre-split vs post-split)
   - Using adjusted vs unadjusted price
   - Price from wrong date (before vs after reporting date)

3. **EntityPublicFloat Definition**
   - EntityPublicFloat might include something different than expected
   - Could include convertible securities, options, etc.
   - Might be calculated differently than shares outstanding

4. **Shares Outstanding Issues**
   - The shares outstanding value might be incorrect
   - Could be missing some share classes
   - Might be from a different reporting period

5. **Timing Differences**
   - EntityPublicFloat might be calculated at a different point in time
   - Shares outstanding might be from a different filing

---

## How to Research

### 1. Check SEC EDGAR Directly
Visit: `https://data.sec.gov/api/xbrl/companyfacts/CIK0001018724.json` (AMZN's CIK)

Look for:
- `facts.us-gaap.CommonStockSharesOutstanding` - shares outstanding
- `facts.dei.EntityPublicFloat` - public float market value

### 2. Check the Actual Filings
- Go to SEC EDGAR: https://www.sec.gov/edgar/searchedgar/companysearch.html
- Search for AMZN
- Look at the 10-Q or 10-K filing for the date in question
- Check the "Cover Page" section for:
  - Shares Outstanding
  - Public Float (both value and shares)

### 3. Verify Stock Price
- Check what stock price was used in the calculation
- Verify it matches the reporting date
- Check if there were any stock splits around that time

### 4. Compare Definitions
- **Shares Outstanding**: Total shares issued by the company
- **Public Float**: Shares available for public trading (excludes closely held shares)
- **EntityPublicFloat**: Market value of public float (not number of shares)

---

## Current Fix

The code now:
1. Calculates float shares from EntityPublicFloat / stock price
2. Validates that float_shares ≤ shares_outstanding
3. Caps float_shares at shares_outstanding if it exceeds
4. Logs a warning when capping occurs

This ensures data integrity but doesn't fix the root cause.

---

## Next Steps for Investigation

1. **Check SEC Filings Directly**: Look at AMZN's 10-Q for 2020-06-30
2. **Verify Stock Price**: Check what price was used on 2020-06-30
3. **Compare Definitions**: Understand how SEC defines EntityPublicFloat vs shares outstanding
4. **Check for Splits**: Verify if there were stock splits that might affect the calculation

