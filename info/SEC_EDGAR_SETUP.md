# SEC EDGAR Shares Outstanding Setup

## Overview

The SEC EDGAR integration downloads historical shares outstanding data directly from the SEC's Company Facts API. This provides free, unlimited access to quarterly historical data.

## Setup

### 1. User-Agent Configuration

The SEC requires a User-Agent header with contact information. You can set this in two ways:

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:SEC_EDGAR_USER_AGENT = "Your Name (your.email@example.com)"

# Windows CMD
set SEC_EDGAR_USER_AGENT=Your Name (your.email@example.com)

# Linux/Mac
export SEC_EDGAR_USER_AGENT="Your Name (your.email@example.com)"
```

**Option B: Edit the Code**
Edit `src/sec_edgar_shares.py` and update the default USER_AGENT:
```python
USER_AGENT = "Your Name (your.email@example.com)"
```

### 2. Rate Limiting

The module automatically implements rate limiting (0.11 seconds between requests = ~9 requests/second) to comply with SEC guidelines.

## Usage

### Standalone Testing

```bash
python src/sec_edgar_shares.py
```

This will test downloading AAPL shares outstanding data.

### Integrated in Download Pipeline

The SEC EDGAR shares outstanding download is automatically integrated into the main download pipeline:

```bash
python src/download_data.py --tickers-file data/tickers/sp500_tickers.csv --start-date 2020-01-01 --end-date 2024-12-31
```

Shares outstanding data will be downloaded and added to the raw CSV files.

## Data Format

- **Frequency**: Quarterly (10-Q) and Annual (10-K) filings
- **Forward-filling**: Data is forward-filled to daily dates (since it's reported quarterly)
- **Column name**: `shares_outstanding` in the raw CSV files

## API Details

- **Endpoint**: `https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json`
- **Data field**: `CommonStockSharesOutstanding` from US-GAAP facts
- **Free**: Yes, unlimited use
- **Rate limit**: SEC recommends max 10 requests/second (we use ~9/sec)

## Troubleshooting

### "CIK not found for ticker"
- Some tickers may not be in the SEC database (e.g., foreign companies, delisted stocks)
- Check if the ticker is listed on a US exchange

### "No shares outstanding data found"
- Company may not have filed recent reports
- Data may be under a different field name (rare)

### Rate limiting errors
- The module automatically retries with exponential backoff
- If you see frequent errors, increase `REQUEST_DELAY` in `src/sec_edgar_shares.py`

## Notes

- Shares outstanding data reflects post-split adjusted values
- Data is updated quarterly when companies file 10-Q reports
- Historical data goes back many years (depends on when company started filing XBRL)

