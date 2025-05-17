@echo off
setlocal enabledelayedexpansion

REM === STEP 1: Download raw data ===
echo.
echo ==============================================================
echo STEP 1: Downloading raw data
echo ==============================================================
IF NOT EXIST data\tickers (
    mkdir data\tickers
)
python -u src\download_data.py ^
  -f data\tickers\sp500_tickers.csv ^
  -s 2008-01-01 ^
  -e 2025-05-17 ^
  -r data\raw ^
  -o data\tickers\sectors.csv
IF ERRORLEVEL 1 (
    echo [ERROR] Download step failed!
    pause
    exit /b 1
)

REM === STEP 2: Clean data ===
echo.
echo ==============================================================
echo STEP 2: Cleaning data
echo ==============================================================
IF NOT EXIST data\clean (
    mkdir data\clean
)
python -u src\clean_data.py ^
  -i data\raw ^
  -o data\clean
IF ERRORLEVEL 1 (
    echo [ERROR] Cleaning step failed!
    pause
    exit /b 1
)

REM === STEP 3: Running backtest ===
echo.
echo ==============================================================
echo STEP 3: Running backtest
echo ==============================================================
IF NOT EXIST reports\trade_logs (
    mkdir reports\trade_logs
)
python -u src\backtest.py ^
  -t AAPL MSFT GOOGL AMZN ^
  --clean-dir data\clean ^
  --sectors-file data\tickers\sectors.csv ^
  --slippage 0 ^
  --risk-percentage .01 ^
  --commission-per-trade 0 ^
  --commission-per-share 0 ^
  -i 100000 ^
  --max-positions 8 ^
  --max-sector-exposure 0.25 ^
  -o reports\trade_logs\trade_log.csv
IF ERRORLEVEL 1 (
    echo [ERROR] Backtest step failed!
    pause
    exit /b 1
)

REM === STEP 4: Evaluating performance ===
echo.
echo ==============================================================
echo STEP 4: Evaluating performance
echo ==============================================================
IF NOT EXIST reports\performance_metrics (
    mkdir reports\performance_metrics
)
python -u src\evaluate.py ^
  -l reports\trade_logs\trade_log.csv ^
  -i 100000 ^
  -o reports\performance_metrics
IF ERRORLEVEL 1 (
    echo [ERROR] Evaluation step failed!
    pause
    exit /b 1
)

REM === STEP 5: Plot performance charts ===
echo.
echo ==============================================================
echo STEP 5: Plotting performance charts
echo ==============================================================
python -u src\plot_performance.py ^
    -l reports\trade_logs\trade_log.csv ^
    -o reports\performance_metrics
if errorlevel 1 (
  echo [ERROR] Plotting step failed!
  pause
  exit /b 1
)

REM === STEP 6: Yearly summary table ===
python -u src\plot_performance.py --trade-log reports\trade_logs\trade_log.csv --initial-capital 100000 --output-dir reports\performance_metrics


echo.
echo All steps completed successfully!
pause
