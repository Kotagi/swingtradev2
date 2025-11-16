# Complete Pipeline Steps

## Overview
Your swing trading ML pipeline consists of 7 main steps that transform raw stock data into actionable trading signals.

---

## Step-by-Step Pipeline

### **Step 1: Download Stock Data**
**Purpose:** Download historical OHLCV (Open, High, Low, Close, Volume) data from yfinance

**Command:**
```bash
python src/swing_trade_app.py download --full
```

**What it does:**
- Downloads raw stock data for all tickers in `data/tickers/sp500_tickers.csv`
- Fetches split factors and sector information
- Saves individual CSV files to `data/raw/`
- Creates/updates `data/tickers/sectors.csv`

**Output:** Raw CSV files in `data/raw/` (one per ticker)

**Alternative:** Use existing batch file
```bash
download_data_full.bat
```

---

### **Step 2: Clean Data**
**Purpose:** Clean and normalize raw stock data with parallel processing and robust validation

**Command:**
```bash
python src/swing_trade_app.py clean
```

Or directly:
```bash
python src/clean_data.py --raw-dir data/raw --clean-dir data/clean
```

**Options:**
- `--raw-dir` / `-r`: Input directory (default: `data/raw`)
- `--clean-dir` / `-c`: Output directory (default: `data/clean`)
- `--resume`: Skip already-cleaned files (resume from previous run)
- `--workers` / `-w`: Number of parallel workers (default: 4)
- `--verbose` / `-v`: Show detailed cleaning steps

**What it does:**
- **Parallel processing:** Cleans multiple files concurrently for faster processing
- **Progress tracking:** Shows real-time progress and statistics
- **Resume capability:** Skips already-cleaned files when using `--resume`
- **Robust validation:** Case-insensitive column detection, handles missing columns gracefully
- **Data cleaning:**
  - Renames columns to lowercase
  - Parses and validates dates
  - Removes invalid/duplicate rows
  - Applies split adjustments to prices
  - Converts to Parquet format for efficiency
- **Enhanced error handling:** Handles empty files, malformed CSVs, and edge cases
- **Summary statistics:** Reports rows dropped, processing time, success/failure counts

**Output:** Cleaned Parquet files in `data/clean/` (one per ticker)

**Note:** This step is automatically included in the full pipeline. Use `--resume` to skip already-cleaned files.

---

### **Step 3: Build Features**
**Purpose:** Compute technical indicators and create labels based on your trading parameters

**Command:**
```bash
python src/swing_trade_app.py features --horizon 5 --threshold 0.05
```

**Parameters:**
- `--horizon`: Trade window in days (e.g., 5 = hold for 5 days)
- `--threshold`: Minimum return threshold (e.g., 0.05 = 5% return)
- `--config`: Feature configuration file (default: `config/features.yaml`)
- `--full`: Force full recomputation (ignores cache)

**What it does:**
- **Parallel processing:** Uses joblib for multi-core feature computation
- **Feature validation:** Automatically checks for infinities, excessive NaNs, and constant values
- **Robust error handling:** Continues processing other tickers if one fails
- **Caching:** Skips up-to-date files automatically (use `--full` to force recompute)
- **Performance optimized:** Uses efficient DataFrame concatenation (no fragmentation warnings)
- Loads cleaned data from `data/clean/`
- Computes **105+ technical indicators** across multiple categories
- **Data leakage prevention:** Forward-looking features removed from registry
- **Ichimoku fixes:** Leading spans modified to avoid lookahead bias
- Creates binary labels: 1 if future return > threshold, else 0
- Saves feature + label data to `data/features_labeled/`
- **Summary statistics:** Reports features computed, validation issues, processing time

**Output:** Feature Parquet files in `data/features_labeled/` (one per ticker)

**Key Features Computed (105+ total):**

**Original Features (~50):**
- **Momentum indicators:** RSI, MACD, Stochastic (with variants)
- **Trend indicators:** SMA, EMA (multiple timeframes), ADX, Ichimoku
- **Volatility indicators:** ATR, Bollinger Bands
- **Volume indicators:** OBV (raw, percent change, z-score), volume ratios
- **Price action:** Returns, gaps, breakouts, z-scores, percentiles
- **Pattern recognition:** Candlestick patterns (engulfing, hammer, doji, morning/evening star, etc.)

**Phase 1 New Features (55):**
- **Support & Resistance (12):** Resistance/support levels (20d, 50d), distances, touches, pivot points, Fibonacci levels
- **Volatility Regime (10):** ATR percentiles, volatility trends, BB squeeze/expansion, volatility flags
- **Trend Strength (11):** ADX variations (20d, 50d), trend consistency, EMA alignment, MA slopes, trend duration, reversal signals
- **Multi-Timeframe Weekly (14):** Weekly SMAs, EMAs, RSI, MACD, returns, volume ratios, trend strength
- **Volume Profile (8):** VWAP, price vs VWAP, volume climax/dry-up, volume trends, breakouts, distribution

**Feature Organization:**
- Features are categorized (momentum, trend, volume, volatility, pattern, price_action, support_resistance, volatility_regime, trend_strength, multi_timeframe, volume_profile)
- See `FEATURE_REDUNDANCY_GUIDE.md` for feature selection recommendations
- See `FEATURE_ROADMAP.md` for complete feature roadmap and future additions
- See `PHASE1_FEATURE_ANALYSIS.md` for Phase 1 feature breakdown
- See `FUTURE_FEATURES.md` for features requiring additional data sources
- See `features/metadata.py` for feature metadata and categories

**Performance Notes:**
- Feature computation is optimized with efficient DataFrame operations
- Weekly features use resampling and forward-fill for multi-timeframe analysis
- Percentile calculations use optimized vectorized operations
- Processing time scales with number of features and tickers (expect ~2-5 minutes for 493 tickers with 105 features)

---

### **Step 4: Train ML Model**
**Purpose:** Train XGBoost classifier with hyperparameter tuning and comprehensive evaluation

**Command:**
```bash
# Quick training (improved default hyperparameters)
python src/swing_trade_app.py train

# With hyperparameter tuning (recommended for best results)
python src/swing_trade_app.py train --tune --n-iter 30

# With cross-validation (more robust, slower)
python src/swing_trade_app.py train --tune --cv

# Full diagnostics with SHAP analysis
python src/swing_trade_app.py train --tune --diagnostics
```

**Options:**
- `--tune`: Perform hyperparameter tuning (RandomizedSearchCV) - **Recommended**
- `--n-iter`: Number of hyperparameter search iterations (default: 20)
- `--cv`: Use cross-validation for hyperparameter tuning (slower but more robust)
- `--no-early-stop`: Disable early stopping (train for full n_estimators)
- `--plots`: Generate training curves and feature importance charts (requires matplotlib)
- `--diagnostics`: Show SHAP diagnostics (requires shap library)

**What it does:**
- **Enhanced data splitting:** Train/Validation/Test split by date:
  - Train: up to 2022-12-31
  - Validation: 2023-01-01 to 2023-12-31 (for early stopping)
  - Test: 2024-01-01 onwards
- Loads all feature data from `data/features_labeled/`
- Filters features based on `config/train_features.yaml`
- Trains baseline models (DummyClassifier, LogisticRegression)
- **Hyperparameter tuning:** Searches 9 hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- **Early stopping:** Prevents overfitting using validation set
- Trains XGBoost classifier with improved class imbalance handling
- **Comprehensive evaluation:** ROC AUC, Average Precision, F1 Score, Confusion Matrix, Precision/Recall/Specificity
- **Feature importance:** Always displays top 20 features with rankings
- Saves model, features, and training metadata

**Output:** 
- Trained model file: `models/xgb_classifier_selected_features.pkl`
- Training metadata: `models/training_metadata.json`
- Training curves plot: `models/xgb_training_curves.png` (if `--plots` used)
- Feature importance chart: `models/feature_importance_chart.png` (if `--plots` used)
- Console metrics: Comprehensive evaluation metrics

**Metrics Shown:**
- **Baseline comparisons:** DummyClassifier, LogisticRegression, XGBoost AUC scores
- **Test set metrics:**
  - ROC AUC (higher is better, >0.70 is good)
  - Average Precision (AP)
  - F1 Score
  - Confusion Matrix (TN, FP, FN, TP)
  - Precision, Recall, Specificity, Accuracy
  - Classification report
- **Feature importance:** Top 20 features with cumulative importance percentages
- **Training metadata:** Hyperparameters, metrics, feature importances, date ranges

**Performance Tips:**
- Use `--tune` for best results (adds ~5-15 minutes but significantly improves performance)
- Start with `--n-iter 20` for faster tuning, increase to 30-50 for better results
- Use `--cv` only if you have time (adds significant computation time)
- Early stopping is enabled by default (use `--no-early-stop` to disable)
- Use `--plots` to visualize training progress and feature importance (helpful for analysis)

---

### **Step 5: Run Backtest**
**Purpose:** Test the trained model on historical data to verify profitability

**Command:**
```bash
python src/swing_trade_app.py backtest \
  --horizon 5 \
  --return-threshold 0.05 \
  --strategy model \
  --model-threshold 0.5 \
  --position-size 1000
```

**Parameters:**
- `--horizon`: Trade window (must match features)
- `--return-threshold`: Return threshold (must match features)
- `--strategy`: `model`, `oracle`, or `rsi`
- `--model-threshold`: Probability threshold for model signals (0.0-1.0)
- `--position-size`: Dollar amount per trade
- `--output`: Optional CSV file to save trades

**What it does:**
- Loads trained model and feature data
- Generates entry signals based on strategy
- Simulates trades: enter at open, exit after horizon days
- Calculates returns and P&L for each trade
- Aggregates performance metrics

**Output:** 
- Console metrics: Win rate, P&L, Sharpe ratio, etc.
- Optional CSV file with all trades

**Metrics Shown:**
- Total Trades
- Win Rate (%)
- Average Return (%)
- Total P&L ($)
- Average P&L per Trade ($)
- Maximum Drawdown ($)
- Sharpe Ratio
- Profit Factor
- Average Holding Days

---

### **Step 6: Identify Current Trades**
**Purpose:** Find current trading opportunities using the trained model

**Command:**
```bash
python src/swing_trade_app.py identify \
  --min-probability 0.6 \
  --top-n 20 \
  --output current_opportunities.csv
```

**Parameters:**
- `--min-probability`: Minimum prediction probability (0.0-1.0)
- `--top-n`: Maximum number of opportunities to return
- `--output`: Optional CSV file to save results
- `--model`: Path to model file (default: `models/xgb_classifier_selected_features.pkl`)

**What it does:**
- Loads trained model and feature list
- For each ticker, loads latest feature data
- Generates predictions using the model
- Filters by probability threshold
- Fetches current prices from yfinance
- Ranks opportunities by confidence

**Output:**
- Console table: Top opportunities with probabilities and prices
- Optional CSV file with detailed results

**Columns Shown:**
- Ticker symbol
- Prediction probability (0.0-1.0)
- Current price
- Date of analysis

---

### **Step 7: Full Pipeline** (Optional - Runs All Steps)
**Purpose:** Run complete pipeline in one command

**Command:**
```bash
python src/swing_trade_app.py full-pipeline \
  --horizon 5 \
  --return-threshold 0.05 \
  --position-size 1000 \
  --min-probability 0.5 \
  --top-n 20
```

**Options:**
- `--skip-download`: Skip data download step
- `--skip-train`: Skip model training step
- `--full-download`: Force full data redownload
- `--full-features`: Force full feature recomputation

**What it does:**
- Executes steps 1-6 in sequence:
  1. Download data (if not skipped)
  2. **Clean data** (automatically included, uses resume mode)
  3. Build features
  4. Train model (if not skipped)
  5. Run backtest
  6. Identify trades
- Stops if any step fails
- Shows progress for each step

---

## Data Flow Diagram

```
Raw Data (CSV)
    ↓ [Step 1: Download]
data/raw/*.csv
    ↓ [Step 2: Clean]
data/clean/*.parquet
    ↓ [Step 3: Features]
data/features_labeled/*.parquet
    ↓ [Step 4: Train]
models/xgb_classifier_selected_features.pkl
    ↓ [Step 5: Backtest]
Performance Metrics
    ↓ [Step 6: Identify]
Current Trading Opportunities
```

---

## Typical Workflow

### First Time Setup:
1. Download data (Step 1)
2. Clean data (Step 2) - if needed
3. Build features (Step 3)
4. Train model (Step 4)
5. Backtest (Step 5)
6. Identify trades (Step 6)

### Daily/Weekly Updates:
1. Download data (Step 1) - incremental updates
2. Build features (Step 3) - only if parameters changed
3. Train model (Step 4) - only if features changed
4. Backtest (Step 5) - verify performance
5. Identify trades (Step 6) - find current opportunities

### Testing New Parameters:
1. Build features (Step 3) - with new horizon/threshold
2. Train model (Step 4)
3. Backtest (Step 5) - compare results
4. Identify trades (Step 6)

---

## Key Configuration Files

- **`config/features.yaml`**: Enable/disable technical indicators
- **`config/train_features.yaml`**: Select features for model training
- **`config/trading_config.yaml`**: Default trading parameters
- **`data/tickers/sp500_tickers.csv`**: List of tickers to analyze

## Documentation Files

- **`FEATURE_ROADMAP.md`**: Complete feature roadmap with all planned features (Phase 1-3)
- **`PHASE1_FEATURE_ANALYSIS.md`**: Detailed analysis of Phase 1 features and data requirements
- **`PHASE1_IMPLEMENTATION_SUMMARY.md`**: Summary of Phase 1 feature implementation
- **`FUTURE_FEATURES.md`**: Features requiring additional data sources (intraday, market indices, etc.)
- **`FEATURE_REDUNDANCY_GUIDE.md`**: Guide to redundant features and selection strategies
- **`features/metadata.py`**: Feature metadata system (categories, ranges, descriptions)
- **`DOWNLOAD_IMPROVEMENTS_SUMMARY.md`**: Details on download step improvements
- **`DOWNLOAD_ERROR_FIXES.md`**: Common download errors and fixes

---

## Quick Reference Commands

```bash
# Complete workflow
python src/swing_trade_app.py full-pipeline --horizon 5 --return-threshold 0.05

# Individual steps
python src/swing_trade_app.py download --full
python src/swing_trade_app.py clean --resume  # Optional, included in full pipeline
python src/swing_trade_app.py features --horizon 5 --threshold 0.05
python src/swing_trade_app.py train
python src/swing_trade_app.py backtest --horizon 5 --return-threshold 0.05
python src/swing_trade_app.py identify --min-probability 0.6 --top-n 20
```

## Recent Improvements

### Download Step
- ✅ Robust error handling with retry logic and exponential backoff
- ✅ Progress bar with real-time statistics
- ✅ Resume capability from checkpoint
- ✅ Data validation and missing data detection
- ✅ Adaptive rate limiting

### Clean Step
- ✅ Parallel processing for faster cleaning
- ✅ Progress tracking and summary statistics
- ✅ Resume capability (skip already-cleaned files)
- ✅ Robust column detection (case-insensitive)
- ✅ Enhanced error handling for edge cases

### Feature Step
- ✅ **105+ features** across 11 categories
- ✅ **Phase 1 features implemented:** Support/Resistance, Volatility Regime, Trend Strength, Multi-Timeframe, Volume Profile
- ✅ Performance optimized (efficient DataFrame operations, no fragmentation)
- ✅ Feature validation (infinities, NaNs, constant values)
- ✅ Data leakage prevention (forward-looking features removed)
- ✅ Ichimoku lookahead bias fixes
- ✅ Standardized column name handling
- ✅ Parallel processing with joblib
- ✅ Optimized percentile calculations
- ✅ Comprehensive summary statistics
- ✅ Feature metadata system with categories
- ✅ Feature redundancy documentation

### Training Step
- ✅ Hyperparameter tuning with RandomizedSearchCV
- ✅ Early stopping to prevent overfitting
- ✅ Cross-validation support
- ✅ Comprehensive evaluation metrics
- ✅ Feature importance display
- ✅ Model versioning and metadata tracking
- ✅ Training curves visualization (optional)
- ✅ Optimized parallelization
- ✅ Fast mode for quicker tuning

