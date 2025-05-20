@echo off
REM ==================================================
REM Profile feature_pipeline.py with cProfile + SnakeViz
REM Usage: Double-click or run from command line
REM ==================================================

echo.
echo ▶ Running feature pipeline under cProfile...
python -m cProfile -o profile.out src\feature_pipeline.py --input-dir data\clean --output-dir data\features_labeled --config config\features.yaml
if %ERRORLEVEL% neq 0 (
    echo.
    echo *** Error: profiling run failed with exit code %ERRORLEVEL% ***
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Profiling complete — output written to profile.out
echo.

echo ▶ Launching SnakeViz visualization...
REM If “snakeviz” isn’t on your PATH, use: python -m snakeviz profile.out
python -m snakeviz profile.out

echo.
pause
