@echo off
REM -------------------------------
REM Run Feature Pipeline & Inspect
REM -------------------------------
pushd %~dp0

REM 1. Run the feature pipeline
python -m src.feature_pipeline --input-dir data\clean_tests --output-dir data\feature_tests --config config\features.yaml

REM 2. Inspect AAPL returns
python src\inspect_returns.py

popd
pause
