@echo off
REM Navigate to the directory containing this script (project root)
pushd %~dp0

REM Ensure Python can find `utils` and `features` modules by adding project root to PYTHONPATH
set PYTHONPATH=%CD%

REM Run the feature pipeline
python src\feature_pipeline.py --input-dir data\clean --output-dir data\features --config config\features.yaml

REM Return to original directory
popd

pause
