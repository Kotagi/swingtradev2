# Swing Trading System

## Version 3.0-rc5

Phase 3 update: Core return features, ATR, Bollinger Band width, EMA crossover, OBV, RSI, and labeling fully implemented and validated.

### Phase 3 – Feature Engineering & Basic ML

#### ✅ Completed
1. **Project Scaffolding & Configuration**  
   - `features/` package with `technical.py`, `registry.py`  
   - `config/features.yaml` feature toggles  
   - `utils/logger.py` for console & file logging  
   - `src/feature_pipeline.py` dynamic pipeline script  
   - Unit tests in `tests/test_features.py` for feature calculations  
   - Integration tests in `tests/test_features_integration.py` with live data  
   - Unit tests in `tests/test_labeling.py` for label logic  

2. **Implemented & Validated Features**  
   - **5-Day Return** (`feature_5d_return`)  
   - **10-Day Return** (`feature_10d_return`)  
   - **Average True Range (ATR)** (`feature_atr`)  
   - **Bollinger Band Width** (`feature_bb_width`)  
   - **EMA(12/26) Crossover** (`feature_ema_cross`)  
   - **On-Balance Volume (OBV)** (`feature_obv`)  
   - **Relative Strength Index (RSI)** (`feature_rsi`)  
   - **Labeling** (`label_future_return` in `utils/labeling.py`, `label_5d`)

#### 🔜 Next Steps
3. **Train/Test Splits (Walk-Forward)**  
   - Implement `utils/splits.py` for walk-forward splits  
   - Unit tests for no-leakage splits  
   - Serialize splits (e.g. `data/splits.json`)

4. **Basic Model Training & Saving**  
   - `scripts/train_model.py` (XGBoost + Optuna)  
   - Evaluate on folds; log metrics; save best model  

5. **Evaluation & Baseline Backtest**  
   - `scripts/evaluate_phase3.py` to compare ML vs. rule baseline  
   - Generate report (`reports/phase3_evaluation.md`)

6. **Finalize Phase 3**  
   - Review documentation  
   - Tag release `v3.0-rc5`  
   - Push to GitHub  

### Usage

1. **Run unit tests**  
   ```bash
   pytest tests/test_features.py tests/test_labeling.py -q
   ```

2. **Run integration smoke tests**  
   ```bash
   pytest tests/test_features_integration.py -q
   ```

3. **Generate features & labels**  
   ```bash
   run_features_labels.bat
   ```

4. **Inspect sample output**  
   ```bash
   python inspect_returns.py
   ```

---

*End of README for Version 3.0-rc5*  
