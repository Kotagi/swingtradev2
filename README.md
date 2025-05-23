# Project README

*Auto-generated by `generate_readme.py`*

## Quickstart

```bash
python src/download_data.py --tickers data/tickers/sp500_tickers.csv
python src/clean_data.py
run_features_labels.bat
python src/train_model.py
```

```
bats/
  ├─ run_feature_pipeline.bat
config/
  ├─ features.yaml
  ├─ train_features.yaml
data/
  ├─ clean/
  ├─ features_labeled/
  ├─ raw/
  ├─ tickers/
    ├─ Archive/
features/
  ├─ __init__.py
  ├─ registry.py
  ├─ technical.py
models/
notebooks/
reports/
  ├─ trade_logs/
scripts/
src/
  ├─ __init__.py
  ├─ backtest.py
  ├─ clean_data.py
  ├─ clean_features_labeled.py
  ├─ download_data.py
  ├─ feature_pipeline.py
  ├─ train_model.py
tests/
  ├─ archive/
    ├─ test_clean_data.py
    ├─ test_features.py
    ├─ test_features_base.py
    ├─ test_labeling.py
utils/
  ├─ __init__.py
  ├─ labeling.py
  ├─ logger.py
clean_data.bat
clean_features.bat
profile_pipeline.bat
readme_generator.py
redownload_and_clean.bat
run_features_labels.bat
```

## Dependencies

**requirements.txt:**
- `pandas>=1.3`
- `numpy>=1.21`
- `requests>=2.25`
- `pytest>=7.0`
- `python-dotenv>=0.19`
- `tabulate>=0.8.9`
- `matplotlib>=3.4`
- `pandas-ta>=0.3.14`


## Config Files

- **config\features.yaml**: pipeline feature toggles (keys: features)
- **config\train_features.yaml**: training feature toggles (keys: features)


## CLI Reference

### `src\download_data.py`

| Flag | Help | Default |
| ---- | ---- | ------- |
| `-f, --tickers-file` | Path to a file with one ticker symbol per line | 'data/tickers/sp500_tickers.csv' |
| `-s, --start-date` | Start date for historical data (YYYY-MM-DD) | '2008-01-01' |
| `-e, --end-date` | End date for historical data (YYYY-MM-DD); defaults to today | None |
| `-r, --raw-folder` | Directory to save downloaded raw CSV files | 'data/raw' |
| `-o, --sectors-file` | Output CSV for ticker–sector mapping | 'data/tickers/sectors.csv' |

### `src\feature_pipeline.py`

| Flag | Help | Default |
| ---- | ---- | ------- |
| `--input-dir` | Clean data folder | - |
| `--output-dir` | Features_labeled output folder | - |
| `--config` | YAML of feature toggles | - |
| `--horizon` | Label lookahead days | 5 |
| `--threshold` | Positive return threshold | 0.0 |


## Modules

### `src\backtest.py`
backtest.py
**Functions:**
- `load_universe(tickers_csv)`  
  Load the list of tickers from a CSV file.
- `backtest_signals(df, signal_col, horizon, position_size)`  
  Backtest a boolean entry signal over a fixed holding period.
- `aggregate_results(trades)`  
  Compute summary performance metrics from a series of trades.
- `main()`  
  Main entry point: executes backtests for oracle and RSI strategies,

### `src\clean_data.py`
clean_data.py
**Functions:**
- `clean_file(path)`  
  Clean a single raw CSV file and return any integrity issues.
- `main()`  
  Entry point: cleans all CSVs under RAW_DIR and logs a summary.

### `src\clean_features_labeled.py`
clean_features_labeled.py
**Functions:**
- `clean_file(path)`  
  Read a single features_labeled CSV, drop rows with any NaNs, and overwrite it.
- `main()`  
  Locate all CSV files under DATA_DIR and clean each using clean_file().

### `src\download_data.py`
download_data.py
**Functions:**
- `download_data(tickers, start_date, end_date, raw_folder)`  
  Download OHLCV data and fetch sector info for each ticker.
- `write_sectors_csv(sectors, sectors_file)`  
  Write the ticker-to-sector mapping to a CSV file.
- `main()`  
  Parse command-line arguments, download data, and write sector mapping.

### `src\feature_pipeline.py`
feature_pipeline.py
**Functions:**
- `apply_features(df, enabled_features, logger)`  
  Apply all enabled feature functions to a single DataFrame.
- `process_file(csv_file, output_path, enabled, label_horizon, label_threshold, log_file)`  
  Worker function to process one ticker's CSV end-to-end.
- `main(input_dir, output_dir, config_path, label_horizon, label_threshold)`  
  Entry point for the parallelized feature pipeline.

### `src\train_model.py`
train_model.py
**Functions:**
- `load_data()`  
  Load all ticker feature CSVs into one concatenated DataFrame.
- `prepare(df)`  
  Clean and split the raw DataFrame into X (features) and y (target).
- `main()`  
  Main routine to train and evaluate the model.

### `utils\labeling.py`
labeling.py
**Functions:**
- `label_future_return(df, close_col, horizon, threshold, label_name)`  
  Add a binary label column to indicate if the forward return exceeds a threshold.
- `label_future_return_regression(df, close_col, horizon, label_name)`  
  Add a continuous target column representing the future return.

### `utils\logger.py`
logger.py
**Functions:**
- `setup_logger(name, log_file, level)`  
  Configure and return a Logger instance.

### `features\registry.py`
registry.py
**Functions:**
- `load_enabled_features(config_path)`  
  Load the feature-toggle YAML and return only the enabled features.

### `features\technical.py`
technical.py
**Functions:**
- `_get_close_series(df)`  
  Return the 'close' price series, handling both lowercase and uppercase.
- `feature_5d_return(df)`  
  Compute 5-day forward return: (close_{t+5} / close_t) - 1.
- `feature_10d_return(df)`  
  Compute 10-day forward return: (close_{t+10} / close_t) - 1.
- `feature_atr(df, period)`  
  Compute Average True Range (ATR) over a given period via pandas-ta.
- `feature_bb_width(df, period, std_dev)`  
  Compute Bollinger Band width: (upper_band - lower_band) / middle_band.
- `feature_ema_cross(df, span_short, span_long)`  
  Compute EMA difference: EMA(short) - EMA(long).
- `feature_obv(df)`  
  Compute On-Balance Volume (OBV) via pandas-ta.
- `feature_obv_pct(df, length)`  
  Compute daily percent change of OBV via Rate-of-Change.
- `feature_obv_zscore(df, length)`  
  Compute z-score of OBV relative to its moving average.
- `feature_rsi(df, period)`  
  Compute Relative Strength Index (RSI) via pandas-ta.
- `feature_sma_5(df)`  
  Compute 5-day Simple Moving Average via pandas-ta.
- `feature_ema_5(df)`  
  Compute 5-day Exponential Moving Average via pandas-ta.
- `feature_sma_10(df)`  
  Compute 10-day Simple Moving Average via pandas-ta.
- `feature_ema_10(df)`  
  Compute 10-day Exponential Moving Average via pandas-ta.
- `feature_sma_50(df)`  
  Compute 50-day Simple Moving Average via pandas-ta.
- `feature_ema_50(df)`  
  Compute 50-day Exponential Moving Average via pandas-ta.
- `feature_adx_14(df, period)`  
  Compute 14-day Average Directional Index (ADX) via pandas-ta.

---
## Enabled Features

- **10d_return**: ✅
- **5d_return**: ✅
- **adx_14**: ✅
- **atr**: ✅
- **bb_width**: ✅
- **ema_10**: ✅
- **ema_5**: ✅
- **ema_50**: ✅
- **ema_cross**: ✅
- **obv**: ✅
- **obv_pct**: ✅
- **obv_z20**: ✅
- **rsi**: ✅
- **sma_10**: ✅
- **sma_5**: ✅
- **sma_50**: ✅

