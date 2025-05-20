@echo off
REM ==================================================
REM Clean all features_labeled CSVs by dropping NaNs
REM Usage: Double-click or run from command line
REM ==================================================

echo.
echo Running clean_features_labeled.py…
python src\clean_features_labeled.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo *** Error: clean_features_labeled.py failed with exit code %ERRORLEVEL% ***
) else (
    echo.
    echo ✅ clean_features_labeled.py completed successfully.
)
echo.
pause
