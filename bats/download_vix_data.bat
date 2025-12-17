@echo off
REM --------------------------------------------------------------------------
REM redownload_and_clean.bat
REM   1) Download raw OHLCV + rebuild sectors.csv via src/download_data.py
REM   
REM --------------------------------------------------------------------------

echo.
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
