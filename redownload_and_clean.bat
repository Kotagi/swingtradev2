@echo off
REM --------------------------------------------------------------------------
REM redownload_and_clean.bat
REM   1) Download raw OHLCV + rebuild sectors.csv via src/download_data.py
REM   2) Clean raw files via src/clean_data.py
REM --------------------------------------------------------------------------

echo.
echo === 1) Redownloading raw data ^& rebuilding sectors.csv ===
python src\download_data.py ^
  --tickers-file data\tickers\sp500_tickers.csv ^
  --start-date   2008-01-01                ^
  --raw-folder   data\raw                  ^
  --sectors-file data\tickers\sectors.csv
IF %ERRORLEVEL% NEQ 0 (
  echo.
  echo [ERROR] download_data.py failed.
  pause
  exit /b %ERRORLEVEL%
)

echo.
echo === 2) Cleaning raw data into data\clean ===
python src\clean_data.py
IF %ERRORLEVEL% NEQ 0 (
  echo.
  echo [ERROR] clean_data.py failed.
  pause
  exit /b %ERRORLEVEL%
)

echo.
echo Redownload ^& clean complete.  
pause
