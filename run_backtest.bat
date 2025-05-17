@echo off
REM ───────────────────────────────────────────────────────────────
REM Batch script to run the rule‐based backtester with slippage & commission
REM Usage: double‐click this file or run it from a cmd prompt
REM ───────────────────────────────────────────────────────────────

REM === Customize these settings ===
set TICKERS=AAPL MSFT AMZN GOOGL
set MOMENTUM=0.05
set ATR_MULT=2
set TIME_EXIT=5
set SLIPPAGE=0.0005
set COMM_TRADE=1.00
set COMM_SHARE=0.005
set INITIAL_CAPITAL=100000
set CLEAN_DIR=data\clean
set OUTPUT=trade_log.csv

REM === Invoke backtest ===
python src\backtest.py ^
  -t %TICKERS% ^
  --momentum-threshold %MOMENTUM% ^
  --stop-loss-atr-mult %ATR_MULT% ^
  --time-exit-days %TIME_EXIT% ^
  --slippage %SLIPPAGE% ^
  --commission-per-trade %COMM_TRADE% ^
  --commission-per-share %COMM_SHARE% ^
  -i %INITIAL_CAPITAL% ^
  --clean-dir %CLEAN_DIR% ^
  -o %OUTPUT%

if errorlevel 1 (
  echo.
  echo *** Backtest failed! See errors above.
) else (
  echo.
  echo *** Backtest complete: %OUTPUT% generated.
)

pause
