@echo off
REM --------------------------------------------------
REM Run Feature Engineering + Labeling Pipeline
REM --------------------------------------------------

pushd %~dp0

REM Call the pipeline script in src\
python src\feature_pipeline.py ^
    --input-dir data\clean ^
    --output-dir data\features_labeled ^
    --config config\features.yaml ^
    --horizon 5 ^
    --threshold 0.0

popd
pause
