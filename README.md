# Swing Trading System

## Version 3.0-rc6

Phase 3 update: Completed labeling and walk‐forward splits.

### Phase 3 – Feature Engineering & Basic ML

#### ✅ Completed
1. **Project Scaffolding & Configuration**  
   - `features/` package with `technical.py`, `registry.py`  
   - `config/features.yaml` feature toggles  
   - `utils/logger.py` for logging  
2. **Feature Engineering & Labeling Pipeline**  
   - `src/feature_pipeline.py` dynamic pipeline for features + labels  
   - **Labeling** in `utils/labeling.py` with `label_future_return`  
3. **Walk-Forward Splits**  
   - `utils/splits.py` with `walk_forward_splits` & `save_splits`  
   - `tests/test_splits.py` unit tests for split logic  
   - `scripts/generate_splits.py` to serialize splits to `data/splits.json`  

#### 🔜 Next Steps
4. **Basic Model Training & Saving**  
   - `scripts/train_model.py` to train XGBoost on splits  
   - Record metrics per fold; save final model  

5. **Evaluation & Baseline Backtest**  
   - `scripts/evaluate_phase3.py` for comparing ML vs. rule baseline  
   - Generate report (`reports/phase3_evaluation.md`)

### Usage

1. **Run unit tests**  
   ```bash
   pytest tests/test_features.py tests/test_labeling.py tests/test_splits.py -q
   ```

2. **Generate splits**  
   ```bash
   run_generate_splits.bat
   ```

3. **Inspect splits**  
   ```bash
   python inspect_splits.py
   ```

---

*End of README for Version 3.0-rc6*  
