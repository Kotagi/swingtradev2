@echo off
REM ───────────────────────────────────────────────────────
REM Full Phase 3 pipeline launcher
REM ───────────────────────────────────────────────────────

REM 0) Make sure src/ is on Python’s import path
set PYTHONPATH=%~dp0\src;%PYTHONPATH%

REM 1) Generate features & labels
echo.
echo =====================================================
echo 1) Feature & Label Generation
echo =====================================================
call run_features_labels.bat
IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Feature & label step failed.
  pause
  exit /b %ERRORLEVEL%
)

REM 2) Build walk-forward splits
echo.
echo =====================================================
echo 2) Walk-Forward Split Generation
echo =====================================================
call run_generate_splits.bat
IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Split generation failed.
  pause
  exit /b %ERRORLEVEL%
)

REM 3) Model Training
echo.
echo =====================================================
echo 3) Model Training
echo =====================================================
if not exist models   mkdir models
if not exist reports  mkdir reports

python scripts\train_model.py ^
  --features-dir data\features_labeled ^
  --splits      data\splits.json    ^
  --model-out   models\xgb_phase3.pkl ^
  --report-out  reports\phase3_train_results.csv
IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Model training failed.
  pause
  exit /b %ERRORLEVEL%
)

REM 4) Evaluation & Backtest
echo.
echo =====================================================
echo 4) Evaluation ^& Backtest
echo =====================================================
python scripts\evaluate_phase3.py ^
  --features-dir  data\features_labeled ^
  --model         models\xgb_phase3.pkl   ^
  --sectors-file  data\tickers\sectors.csv ^
  --report        reports\phase3_evaluation.md
IF %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Evaluation/backtest failed.
  pause
  exit /b %ERRORLEVEL%
)

echo.
echo =====================================================
echo Full Phase 3 pipeline complete!
echo Outputs:
echo  - models\xgb_phase3.pkl
echo  - reports\phase3_train_results.csv
echo  - reports\phase3_evaluation.md
echo =====================================================
pause
