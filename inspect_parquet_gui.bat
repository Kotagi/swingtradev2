@echo off
REM ==================================================
REM Parquet File Inspector GUI
REM Usage: Double-click this file to run the GUI
REM ==================================================

echo.
echo Starting Parquet Inspector GUI...
echo.

python src/inspect_parquet_gui.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo *** Error: GUI failed to start ***
    echo Make sure Python and required packages are installed.
    pause
    exit /b %ERRORLEVEL%
)

pause

