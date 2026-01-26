@echo off
REM --------------------------------------------------------------------------
REM   clean_data.bat
REM   Clean raw files via src/clean_data.py
REM --------------------------------------------------------------------------


echo.
echo === 2) Cleaning raw data into data\clean ===
python src\clean_data.py ^
	--raw-dir data/macro/raw ^
	--clean-dir data/macro/clean 
IF %ERRORLEVEL% NEQ 0 (
  echo.
  echo [ERROR] clean_data.py failed.
  pause
  exit /b %ERRORLEVEL%
)

echo.
echo Clean complete.  
pause
