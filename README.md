# Swing Trading System

## Version 3.0-alpha

Initial Phase 3 snapshot: Core return features and pipeline scaffold implemented.

### Phase 3 – Feature Engineering & Basic ML

#### ✅ Completed
1. **Project Scaffolding & Configuration**  
   - Created `features/` package with `technical.py` and `registry.py`  
   - Added `config/features.yaml` for feature toggles  
   - Set up `tests/test_features.py` with unit tests for 5d and 10d returns  
   - Added `utils/logger.py` for console & file logging  
   - Created `src/feature_pipeline.py` to dynamically load and apply features  

2. **Implemented Core Return Features**  
   - `feature_5d_return(df)`  
   - `feature_10d_return(df)`  
   - Unit tests passing and integration tests on real tickers (AAPL, MSFT, GOOGL, AMZN)  

#### 🔜 Next Steps
3. **Average True Range (ATR)**  
   - [ ] Write unit test for `feature_atr` (period=3 example)  
   - [ ] Implement `feature_atr(df, period)` in `features/technical.py`  
   - [ ] Confirm test passes (`pytest tests/test_features.py::test_feature_atr`)  
   - [ ] Run integration tests to verify on real data (`pytest tests/test_features_integration.py`)  
   - [ ] Spot-check `data/features/*.csv` for `atr` column  

4. **Continue Feature Rollout**  
   - Follow the established workflow for each new indicator:  
     - Unit test → Implementation → Registration → Integration test → Smoke-run → Spot-check → Commit  

### Usage

1. **Run unit tests**  
   ```bash
   pytest -q
   ```

2. **Generate features**  
   ```bash
   python -m src.feature_pipeline \
     --input-dir data/clean \
     --output-dir data/features \
     --config config/features.yaml
   ```

3. **Inspect sample output**  
   ```bash
   python inspect_returns.py
   ```

---

*End of README for Version 3.0-alpha*  
