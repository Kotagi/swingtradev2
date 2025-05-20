# SwingTradeV1 – Phase 3 Baseline (v3.0-alpha)

This commit captures the working end-to-end baseline for Phase 3:  
– Core technical features (5d/10d returns, ATR, BB-width, EMA crossover, OBV, RSI)  
– Feature-engineering pipeline with per-feature toggles  
– Labeling (5-day future return > 0 → 1/0)  
– Walk-forward train/test splits  
– XGBoost model training  
– Evaluation & backtest scaffolding

---

## 📂 Project Structure

```
.
├── src/
│   ├── features/
│   │   ├── technical.py
│   │   └── registry.py
│   └── feature_pipeline.py
├── utils/
│   ├── logger.py
│   └── splits.py
├── config/
│   └── features.yaml
├── data/
│   ├── raw/
│   ├── clean/
│   ├── features_labeled/
│   └── tickers/
│       ├── sp_500_tickers.csv
│       └── sectors.csv
├── scripts/
│   ├── download_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   └── evaluate_phase3.py
├── tests/
│   ├── test_features.py
│   ├── test_splits.py
│   └── test_train_model.py
├── models/
│   └── xgb_phase3.pkl
├── reports/
│   ├── phase3_train_results.csv
│   └── phase3_evaluation.md
├── run_features_labels.bat
├── run_generate_splits.bat
├── run_train_model.bat
├── run_evaluate.bat
├── run_full_pipeline.bat
└── README.md
```

---

## ⚙️ Prerequisites

```bash
pip install -r requirements.txt
```

---

## ▶️ Full Phase 3 Pipeline

1. **(Re-)Download & Clean Data**  
   ```bat
   redownload_and_clean.bat
   ```
2. **Build Features & Labels**  
   ```bat
   run_features_labels.bat
   ```
3. **Generate Splits**  
   ```bat
   run_generate_splits.bat
   ```
4. **Train Model**  
   ```bat
   run_train_model.bat
   ```
5. **Evaluate & Backtest**  
   ```bat
   run_evaluate.bat
   ```
6. **Or do all at once**:  
   ```bat
   run_full_pipeline.bat
   ```

---

## 📈 Outputs

- **Model:** `models/xgb_phase3.pkl`  
- **Train metrics:** `reports/phase3_train_results.csv`  
- **Evaluation:** `reports/phase3_evaluation.md`

---

## 🔜 Next Steps

- Wire the ML `signal` into the backtester entry logic  
- Tune hyperparameters & feature set  
- Analyze P&L, feature importances, and robustness

*v3.0-alpha – commit this baseline before further enhancements*
