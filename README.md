# Swing Trading System

## Version 3.0-rc3

Phase 3 update: Core return features, ATR, Bollinger Band width, EMA crossover, and OBV fully implemented and validated.

### Phase 3 – Feature Engineering & Basic ML

#### ✅ Completed
1. **Project Scaffolding & Configuration**  
   - `features/` package with `technical.py`, `registry.py`  
   - `config/features.yaml` feature toggles  
   - `utils/logger.py` for console & file logging  
   - `src/feature_pipeline.py` dynamic pipeline script  
   - Unit tests in `tests/test_features.py` for return, ATR, BB width, EMA crossover, OBV  
   - Integration tests in `tests/test_features_integration.py` with live data  

2. **Implemented & Validated Features**  
   - **5-Day Return** (`feature_5d_return`)  
   - **10-Day Return** (`feature_10d_return`)  
   - **Average True Range (ATR)** (`feature_atr`)  
   - **Bollinger Band Width** (`feature_bb_width`)  
   - **EMA(12/26) Crossover** (`feature_ema_cross`)  
   - **On-Balance Volume (OBV)** (`feature_obv`)  

#### 🔜 Next Steps
3. **Relative Strength Index (RSI)**  
   - Write unit test for `feature_rsi`  
   - Implement calculation and confirm test passes  
   - Run integration tests

4. **Finalize Phase 3**  
   - Commit all changes  
   - Tag release `v3.0-rc3`  
   - Push to GitHub  

### Usage

1. **Run unit tests**  
   ```bash
   pytest tests/test_features.py -q
   ```

2. **Run integration smoke tests**  
   ```bash
   pytest tests/test_features_integration.py -q
   ```

3. **Generate features**  
   ```bash
   python -m src.feature_pipeline \
     --input-dir data/clean \
     --output-dir data/features \
     --config config/features.yaml
   ```

4. **Inspect sample output**  
   ```bash
   python inspect_returns.py
   ```

---

*End of README for Version 3.0-rc3*  
