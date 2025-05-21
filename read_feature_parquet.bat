@echo off
REM ==================================================
REM Feature Parquet Inspector
REM Usage: Double-click or run from command line
REM ==================================================

echo.
echo ▶ Running clean parquet inspector...
python src/inspect_parquet.py data/features_labeled/AAPL.parquet
if %ERRORLEVEL% neq 0 (
    echo.
    echo *** Error: profiling run failed with exit code %ERRORLEVEL% ***
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Inspection Complete
echo.

