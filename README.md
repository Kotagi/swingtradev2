# Swing Trading System

## Version 3.0-beta

Phase 3 update: Core return features and ATR fully implemented and validated.

### Phase 3 – Feature Engineering & Basic ML

#### ✅ Completed
1. **Project Scaffolding & Configuration**  
   - `features/` package with `technical.py`, `registry.py`  
   - `config/features.yaml` toggles for all features  
   - Unit tests in `tests/test_features.py` for return and ATR features  
   - `tests/test_features_integration.py` for real-data smoke tests  
   - `utils/logger.py` for consistent logging  
   - `src/feature_pipeline.py` dynamic pipeline script

2. **Implemented & Validated Features**  
   - **5-Day Return** (`feature_5d_return`)  
   - **10-Day Return** (`feature_10d_return`)  
   - **Average True Range (ATR)** (`feature_atr`)  

#### 🔜 Next Steps
3. **Bollinger Band Width**  
   - Write unit test for `feature_bb_width`  
   - Implement calculation and confirm test passes  
   - Run integration tests

4. **EMA(12/26) Crossover**  
   - Unit test for `feature_ema_cross`  
   - Implementation and validation  

5. **On-Balance Volume (OBV)**  
   - Unit test for `feature_obv`  
   - Implementation and validation  

6. **Relative Strength Index (RSI)**  
   - Unit test for `feature_rsi`  
   - Implementation and validation  

7. **Finalize Phase 3**  
   - Commit all changes  
   - Tag release `v3.0-beta`  
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

*End of README for Version 3.0-beta*  
