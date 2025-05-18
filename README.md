# Swing Trading System

## Version 2.1

Bump version to 2.1 and commit optimized default parameters.

### Default Parameters
- **momentum_threshold**: 0.015
- **stop_loss_atr_mult**: 2.0
- **time_exit_days**: 7.0
- **risk_percentage**: 0.01

## Phase 3 – Feature Engineering & Basic ML

[ ] 1. Feature List  
    [ ] 5-day return, 10-day return  
    [ ] ATR(14), Bollinger Band width  
    [ ] EMA(12/26) crossover value  
    [ ] OBV, RSI(14)

[ ] 2. Engineering Script  
    [ ] Build features.py to compute & merge features per ticker/day  
    [ ] Save feature matrix + future-return label to data/clean/features.parquet

[ ] 3. Labeling  
    [ ] Define target = 1 if return over next 5 days > 0%; else 0

[ ] 4. Train/Test Splits  
    [ ] Implement walk-forward splits (e.g., train on 2008–2015, test 2016; slide quarterly)

[ ] 5. Model Training  
    [ ] Train XGBoost classifier on features  
    [ ] Use Optuna to tune basic hyperparameters

[ ] 6. Evaluation  
    [ ] Track classification metrics (accuracy, precision/recall)  
    [ ] Compare ML-driven backtest vs rule-only backtest

[ ] 7. Integration  
    [ ] Modify backtest.py to accept ML signals in place of rule entry  
    [ ] Re-run backtest; log results to reports/

[ ] 8. Commit Phase 3 code & findings
