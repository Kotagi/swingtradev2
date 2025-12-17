@echo off
REM --------------------------------------------------------------------------
REM redownload_and_clean.bat
REM   1) Download raw OHLCV + rebuild sectors.csv via src/download_data.py
REM   
REM --------------------------------------------------------------------------

echo.
echo === 1) Downloading Raw Data ^& Building Sectors.csv ===
python src\download_data.py ^
  --tickers-file data\tickers\sp500_tickers.csv ^
  --start-date   2008-01-01                ^
  --raw-folder   data\raw                  ^
  --sectors-file data\tickers\sectors.csv  ^
  --chunk-size 500^
  --pause 10^
  --full
	
IF %ERRORLEVEL% NEQ 0 (
  echo.
  echo [ERROR] Downloading raw data failed.
  pause
  exit /b %ERRORLEVEL%
)

echo === 1) Downloading Macro Raw Data ===
python src\download_data.py ^
  --tickers-file data\tickers\macro_tickers.csv ^
  --start-date   2008-01-01                ^
  --raw-folder   data\macro\raw                  ^
  --no-sectors  ^
  --chunk-size 100^
  --pause 0.1^
  --full
	
IF %ERRORLEVEL% NEQ 0 (
  echo.
  echo [ERROR] Downloading macro raw data failed.
  pause
  exit /b %ERRORLEVEL%
)

echo === 1) Downloading VIX Data ===
python src\download_vix.py ^
  --output-folder data\macro\clean ^
  --start-date   2008-01-01

	
IF %ERRORLEVEL% NEQ 0 (
  echo.
  echo [ERROR] Downloading VIX data failed.
  pause
  exit /b %ERRORLEVEL%
)
echo.
echo Download complete.  
pause
