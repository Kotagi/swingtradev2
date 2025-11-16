# Testing Guide

## Quick Test (If You Already Have Data)

Since you already have data in `data/clean/` and `data/features_labeled/`, you can test the application quickly:

### Option 1: Quick Feature Test (5 minutes)
Test with a small subset using existing features:

```bash
# 1. Test feature pipeline (rebuild features for a few tickers)
python src/swing_trade_app.py features --horizon 5 --threshold 0.05

# 2. Test model training
python src/swing_trade_app.py train

# 3. Test backtest
python src/swing_trade_app.py backtest --horizon 5 --return-threshold 0.05 --strategy model

# 4. Test trade identification
python src/swing_trade_app.py identify --min-probability 0.5 --top-n 10
```

### Option 2: Full Pipeline Test (15-30 minutes)
Run the complete pipeline:

```bash
python src/swing_trade_app.py full-pipeline \
  --horizon 5 \
  --return-threshold 0.05 \
  --skip-download
```

### Option 3: Individual Component Tests

#### Test 1: Verify Imports Work
```bash
python -c "from utils.logger import setup_logger; from features.registry import load_enabled_features; print('✓ Imports work!')"
```

#### Test 2: Test Feature Pipeline
```bash
python src/feature_pipeline.py \
  --input-dir data/clean \
  --output-dir data/features_labeled \
  --config config/features.yaml \
  --horizon 5 \
  --threshold 0.05
```

#### Test 3: Test Model Training
```bash
python src/train_model.py
```

#### Test 4: Test Enhanced Backtest
```bash
python src/enhanced_backtest.py \
  --horizon 5 \
  --return-threshold 0.05 \
  --strategy model \
  --position-size 1000
```

#### Test 5: Test Trade Identification
```bash
python src/identify_trades.py \
  --min-probability 0.5 \
  --top-n 10
```

## Expected Results

### Feature Pipeline
- Should process all tickers in `data/clean/`
- Should create/update Parquet files in `data/features_labeled/`
- Should show progress messages

### Model Training
- Should load feature data
- Should train XGBoost model
- Should show ROC AUC score (typically 0.50-0.65)
- Should save model to `models/xgb_classifier_selected_features.pkl`

### Backtest
- Should show performance metrics:
  - Total Trades
  - Win Rate
  - Average Return
  - Total P&L
  - Sharpe Ratio
  - Profit Factor

### Trade Identification
- Should analyze all tickers
- Should show top opportunities with probabilities
- Should include current prices

## Troubleshooting

### If feature pipeline fails:
- Check that `data/clean/` has Parquet files
- Verify `config/features.yaml` exists
- Check that `utils/` and `features/` modules are importable

### If model training fails:
- Ensure `data/features_labeled/` has Parquet files
- Check that `config/train_features.yaml` exists
- Verify features are computed (no excessive NaNs)

### If backtest fails:
- Ensure model file exists: `models/xgb_classifier_selected_features.pkl`
- Check that features match model expectations
- Verify data has required columns (Open, Close, etc.)

### If trade identification fails:
- Ensure model is trained
- Check that feature data is up-to-date
- Verify yfinance can fetch current prices

## Quick Verification Commands

```bash
# Check if data exists
dir data\clean\*.parquet | measure-object | select-object Count
dir data\features_labeled\*.parquet | measure-object | select-object Count

# Check if model exists
dir models\xgb_classifier_selected_features.pkl

# Test Python imports
python -c "import pandas_ta; import xgboost; import yfinance; print('All packages imported!')"
```

