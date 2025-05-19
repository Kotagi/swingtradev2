@echo off
pushd %~dp0
python scripts\generate_splits.py ^
  --features-dir data\features_labeled ^
  --out data\splits.json ^
  --train-days 2000 ^
  --test-days 250 ^
  --step-days 60
popd
pause
