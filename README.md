# SwingTradeV1 Data Pipeline

## Description

This repository implements Phase 1 of the SwingTrade stock analysis application. It provides:

- **Data Ingestion Module**: Downloads historical price data for specified tickers into `data/raw/`.
- **Data Cleaning Module**: Cleans and adjusts raw data for splits/dividends, fills missing dates/bars, and outputs to `data/clean/`.

## Project Structure

```
.
├── data
│   ├── raw        # Raw CSV files downloaded from source
│   └── clean      # Processed and cleaned CSV files
├── src
│   ├── ingest_data.py    # Ingest raw historical data
│   └── clean_data.py     # Clean and process raw data
├── tests
│   └── test_clean_data.py  # Unit tests for cleaning module
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Configuration

- **Ticker List**: Modify the `TICKERS` list in `src/ingest_data.py` to control which symbols are downloaded.
- **Date Range**: Adjust `START_DATE` and `END_DATE` in `src/ingest_data.py`.

## Usage

1. **Ingest raw data**  
   ```bash
   python src/ingest_data.py
   ```
2. **Clean data**  
   ```bash
   python src/clean_data.py
   ```

Cleaned files will be saved to `data/clean/`.

## Validation & Testing

Run unit tests on sample tickers:
```bash
pytest
```
Tests have been executed successfully on tickers AAPL, MSFT, AMZN, GOOGL.

## Next Steps

- Phase 2: Feature engineering and backtesting modules.
- Expand documentation and add command‑line interface.

## License

This project is licensed under the MIT License.
