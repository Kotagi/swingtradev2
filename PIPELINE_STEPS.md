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
- `--cv-folds`: Number of CV folds (default: 3, or 5 for TimeSeriesSplit)
- `--no-early-stop`: Disable early stopping (train for full n_estimators)
- `--plots`: Generate training curves and feature importance charts (requires matplotlib)
- `--diagnostics`: Show SHAP diagnostics (requires shap library)
- `--fast`: Use reduced hyperparameter space for quicker tuning
- `--imbalance-multiplier`: Class imbalance multiplier (default: 1.0, try 2.0-3.0 for more trades)
- `--train-end`: Training data end date (YYYY-MM-DD, default: 2022-12-31)
- `--val-end`: Validation data end date (YYYY-MM-DD, default: 2023-12-31)
- `--horizon`: Trade horizon in days (e.g., 5, 30). Used to auto-detect label column.
- `--label-col`: Label column name (e.g., 'label_5d', 'label_30d'). Auto-detected if not specified.

**What it does:**
- **Enhanced data splitting:** Train/Validation/Test split by date:
  - Train: up to 2022-12-31
  - Validation: 2023-01-01 to 2023-12-31 (for early stopping)
  - Test: 2024-01-01 onwards
- Loads all feature data from `data/features_labeled/`
- Filters features based on `config/train_features.yaml`
- **Feature normalization/scaling:**
  - Automatically identifies which features need scaling vs those already normalized
  - Features already normalized (kept as-is): Pattern features (0-1), percentile features (0-1), crossover signals (-1, 0, 1)
  - Features scaled (StandardScaler): Price-based features, unbounded percentages, z-scores, MACD, OBV, etc.
  - Fits StandardScaler on training data only (prevents data leakage)
  - Transforms train/validation/test sets using the fitted scaler
  - Saves scaler with model for consistent inference
- Trains baseline models (DummyClassifier, LogisticRegression with scaling)
- **Hyperparameter tuning:** Searches 9 hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- **Early stopping:** Prevents overfitting using validation set
- Trains XGBoost classifier with improved class imbalance handling
- **Comprehensive evaluation:** ROC AUC, Average Precision, F1 Score, Confusion Matrix, Precision/Recall/Specificity
- **Feature importance:** Always displays top 20 features with rankings
- Saves model, scaler, features, and training metadata

**Output:** 
- Trained model file: `models/xgb_classifier_selected_features.pkl` (includes model, scaler, features, metadata)
- Training metadata: `models/training_metadata.json`
- Training curves plot: `models/xgb_training_curves.png` (if `--plots` used)
- Feature importance chart: `models/feature_importance_chart.png` (if `--plots` used)
- Console metrics: Comprehensive evaluation metrics including feature normalization summary

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
- `--stop-loss`: Stop-loss threshold as decimal (e.g., -0.075 for -7.5%). If not specified and return-threshold is provided, uses 2:1 risk-reward (return_threshold / 2)
- `--output`: Optional CSV file to save trades

**What it does:**
- Loads trained model, scaler, and feature data
- **Applies feature scaling:** Automatically scales features during prediction using the saved scaler
- Generates entry signals based on strategy
- Simulates trades with configurable exit logic:
  - Exit at return threshold (if reached)
  - Exit at stop-loss (2:1 risk-reward by default, configurable)
  - Exit at time horizon (if neither threshold reached)
- Prevents overlapping trades (only one position per ticker at a time)
- Calculates returns and P&L for each trade
- Tracks exit reasons (target_reached, stop_loss, time_limit, end_of_data)
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
- Exit Reasons Breakdown (target_reached, stop_loss, time_limit, end_of_data)
- Time Limit Trades Analysis (win/loss breakdown, return distribution)

---

### **Step 5.5: Analyze Stop-Loss Trades** (Optional)
**Purpose:** Analyze patterns in stop-loss trades to identify risk factors and improve entry filters

**Command:**
```bash
python src/analyze_stop_losses.py \
  --horizon 30 \
  --return-threshold 0.15 \
  --model-threshold 0.80 \
  --stop-loss -0.075
```

**Parameters:**
- `--horizon`: Trade window (must match features)
- `--return-threshold`: Return threshold (must match features)
- `--model-threshold`: Model probability threshold
- `--stop-loss`: Stop-loss threshold (e.g., -0.075 for -7.5%)
- `--trades-csv`: Optional CSV file with existing trades (if not provided, runs backtest)
- `--use-validation`: Use validation data (2023) for analysis instead of test data (default: True)
- `--train-end`: Training data end date (default: 2022-12-31)
- `--val-end`: Validation data end date (default: 2023-12-31)
- `--output`: Optional JSON file to save analysis results

**What it does:**
- Runs backtest on validation data (2023) to generate trades (or loads from CSV)
- Identifies stop-loss trades vs winning trades
- **Feature analysis:** Compares feature distributions between stop-loss and winners
  - Calculates mean differences and effect sizes
  - Identifies top features that predict stop-losses
  - Generates filter recommendations based on significant differences
- **Timing analysis:** Analyzes patterns in entry timing
  - Day of week distribution
  - Month distribution
  - Days to stop-loss distribution
- **Return analysis:** Analyzes stop-loss return distribution
- Provides actionable filter recommendations

**Output:**
- Console analysis report with:
  - Stop-loss vs winner statistics
  - Top features that differ (with effect sizes)
  - Filter recommendations with thresholds
  - Timing patterns (day of week, month, days to stop-loss)
  - Return distribution analysis
- Optional JSON file with detailed results

**Key Insights:**
- Identifies which features correlate with stop-loss trades
- Reveals timing patterns (e.g., Monday/Tuesday entries, October risk)
- Provides specific filter thresholds to reduce stop-loss rate
- **Important:** Uses validation data (2023) by default to avoid data leakage when creating filters

**Example Output:**
- Top risk factors: Negative RSI slope, low market correlation, price near support
- Timing risks: Monday/Tuesday entries (53% of stop-losses), October (24.8% of stop-losses)
- Filter recommendations: `rsi_slope > -1.349`, `market_correlation_20d > 0.532`, etc.

---

### **Step 5.6: Apply Entry Filters** (Optional)
**Purpose:** Apply entry filters to reduce stop-loss trades and improve backtest performance

**Command:**
```bash
# Use default filters (from stop-loss analysis)
python src/apply_entry_filters.py \
  --horizon 30 \
  --return-threshold 0.15 \
  --model-threshold 0.80 \
  --stop-loss -0.075

# Disable timing filters
python src/apply_entry_filters.py \
  --horizon 30 \
  --return-threshold 0.15 \
  --no-timing-filters

# Add custom filters
python src/apply_entry_filters.py \
  --horizon 30 \
  --return-threshold 0.15 \
  --custom-filter rsi_slope ">" 0.5 \
  --custom-filter market_correlation_20d ">" 0.6
```

**Parameters:**
- `--horizon`: Trade window (must match features)
- `--return-threshold`: Return threshold (must match features)
- `--model-threshold`: Model probability threshold (default: 0.80)
- `--stop-loss`: Stop-loss threshold (e.g., -0.075 for -7.5%)
- `--position-size`: Position size per trade (default: 1000.0)
- `--test-start-date`: Test data start date (default: 2024-01-01)
- `--test-end-date`: Test data end date (default: None, all available)
- `--use-default-filters`: Use default filters from stop-loss analysis (default: True)
- `--no-timing-filters`: Disable timing filters (Monday/Tuesday, October)
- `--custom-filter`: Add custom filter (can be used multiple times)
  - Format: `--custom-filter FEATURE OPERATOR THRESHOLD`
  - Example: `--custom-filter rsi_slope ">" 0.5`

**What it does:**
- Loads trained model, scaler, and feature data
- **Applies default filters** (17 feature-based filters from stop-loss analysis):
  - Momentum filters: `rsi_slope > -1.349`, `log_return_1d > -0.017`, `log_return_5d > -0.057`
  - Market context: `relative_strength_spy_5d > -4.269`, `market_correlation_20d > 0.532`
  - Price action: `candle_body_pct > -25.37`, `close_position_in_range > 0.368`, `price_near_support < 0.443`
  - Overbought: `weekly_rsi_14w < 44.455`, `trend_consistency < 35.638`
  - Time-based: `month_of_year_cos > 0.172`, `day_of_month_cos > 0.039`
- **Applies timing filters** (if enabled):
  - Avoids Monday/Tuesday entries (53% of stop-losses in validation data)
  - Avoids October entries (24.8% of stop-losses in validation data)
- Filters entry signals: Only enters when both model signal AND all filters pass
- Runs backtest with filtered signals
- Compares performance to baseline (without filters)

**Output:**
- Console metrics: Same as regular backtest, plus:
  - Filter configuration summary
  - Stop-loss rate comparison (with vs without filters)
  - Exit reasons breakdown

**Default Filters (from validation data analysis):**
- **Momentum:** RSI slope, 1-day return, 5-day return thresholds
- **Market Context:** Relative strength vs SPY, market correlation
- **Price Action:** Candle body, close position, support/resistance proximity
- **Overbought:** Weekly RSI, trend consistency, high vs close
- **Time-Based:** Month/day cyclical features
- **Timing:** Avoid Monday/Tuesday, avoid October

**Best Practices:**
1. **First analyze stop-losses** on validation data (Step 5.5)
2. **Test filters on validation data** to verify they help
3. **Then test on test data** (2024+) for out-of-sample evaluation
4. **Start with default filters**, then customize based on results
5. **Monitor trade count** - if filters reduce trades too much, relax thresholds

**Note:** Filters are derived from validation data (2023) to prevent data leakage. Test on test data (2024+) for true out-of-sample performance.

---

### **Step 6: Identify Current Trades**
**Purpose:** Find current trading opportunities using the trained model with optional entry filters

**Command:**
```bash
# Basic identification
python src/swing_trade_app.py identify \
  --min-probability 0.6 \
  --top-n 20 \
  --output current_opportunities.csv

# With recommended filters (reduces false positives)
python src/identify_trades.py \
  --min-probability 0.6 \
  --top-n 20 \
  --use-recommended-filters \
  --output current_opportunities.csv

# With custom filters
python src/identify_trades.py \
  --min-probability 0.6 \
  --top-n 20 \
  --custom-filter candle_body_pct ">" -10 \
  --custom-filter close_position_in_range ">" 0.40 \
  --custom-filter weekly_rsi_14w "<" 42 \
  --output current_opportunities.csv
```

**Parameters:**
- `--min-probability`: Minimum prediction probability (0.0-1.0)
- `--top-n`: Maximum number of opportunities to return
- `--output`: Optional CSV file to save results
- `--model`: Path to model file (default: `models/xgb_classifier_selected_features.pkl`)
- `--data-dir`: Directory containing feature Parquet files (default: `data/features_labeled`)
- `--tickers-file`: Path to CSV file with ticker symbols (default: `data/tickers/sp500_tickers.csv`)
- `--use-recommended-filters`: Use recommended filters from stop-loss analysis (default: False)
- `--custom-filter`: Add custom filter (can be used multiple times)
  - Format: `--custom-filter FEATURE OPERATOR THRESHOLD`
  - Example: `--custom-filter candle_body_pct ">" -10`
  - Operators: `>`, `<`, `>=`, `<=`

**What it does:**
- Loads trained model, scaler, and feature list
- **Applies feature scaling:** Automatically scales features during prediction using the saved scaler
- For each ticker, loads latest feature data
- **Applies entry filters** (if enabled):
  - Filters are applied **before** making predictions
  - Only tickers that pass all filters get predictions
  - Reduces false positives by filtering out high-risk entry conditions
- Generates predictions using the model (with proper feature scaling)
- Filters by probability threshold
- Fetches current prices from yfinance
- Ranks opportunities by confidence

**Recommended Filters (from stop-loss analysis):**
When using `--use-recommended-filters`, the following 5 filters are applied:
1. **`candle_body_pct > -10`** - Avoids very bearish candles (effect size: -0.725)
2. **`close_position_in_range > 0.40`** - Only enters when price is in upper portion of daily range (effect size: -0.646)
3. **`weekly_rsi_14w < 42`** - Avoids overbought weekly RSI conditions (effect size: 0.603)
4. **`market_correlation_20d > 0.60`** - Only enters when stock is moving with the market (effect size: -0.497)
5. **`volatility_regime > 60`** - Prefers higher volatility environments (effect size: -0.464)

**Output:**
- Console table: Top opportunities with probabilities and prices
- Filter summary: Shows which filters were applied
- Optional CSV file with detailed results

**Columns Shown:**
- Ticker symbol
- Prediction probability (0.0-1.0)
- Current price
- Date of analysis

**Best Practices:**
1. **Start without filters** to see baseline opportunities
2. **Add recommended filters** to reduce false positives
3. **Customize filters** based on your risk tolerance and backtest results
4. **Monitor trade count** - if filters reduce opportunities too much, relax thresholds
5. **Use same filters** in backtest and identification for consistency

---

### **Step 6.5: Compare Filters** (Optional)
**Purpose:** Compare backtest performance with and without entry filters

**Command:**
```bash
python src/compare_filters.py \
  --horizon 30 \
  --return-threshold 0.15 \
  --model-threshold 0.80 \
  --stop-loss -0.075
```

**Parameters:**
- `--horizon`: Trade window (must match features)
- `--return-threshold`: Return threshold (must match features)
- `--model-threshold`: Model probability threshold
- `--stop-loss`: Stop-loss threshold (e.g., -0.075 for -7.5%)
- `--position-size`: Position size per trade (default: 1000.0)
- `--test-start-date`: Test data start date (default: 2024-01-01)
- `--test-end-date`: Test data end date (default: None)

**What it does:**
- Runs backtest **without filters** (baseline)
- Runs backtest **with all default filters**
- Compares metrics side-by-side:
  - Total trades, win rate, returns, P&L, Sharpe ratio, profit factor
  - Stop-loss rate comparison
  - Trade count reduction
- Analyzes which filters are applied (checks feature availability)
- Identifies missing features (filters that couldn't be applied)

**Output:**
- Side-by-side comparison table
- Change metrics (baseline vs filtered)
- Filter analysis (which filters were applied, which are missing)

**Use Cases:**
- Verify filters improve performance before deploying
- Identify which filters are too restrictive
- Check feature availability in test data
- Optimize filter thresholds

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
    ↓ [Step 5.5: Analyze Stop-Losses] (Optional)
Stop-Loss Analysis & Filter Recommendations
    ↓ [Step 5.6: Apply Entry Filters] (Optional)
Filtered Backtest Results
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
5. Identify trades (Step 6) - find current opportunities (with filters if desired)

### Testing New Parameters:
1. Build features (Step 3) - with new horizon/threshold
2. Train model (Step 4)
3. Backtest (Step 5) - compare results
4. Analyze stop-losses (Step 5.5) - identify risk factors
5. Apply entry filters (Step 5.6) - improve performance
6. Compare filters (Step 6.5) - verify improvements
7. Identify trades (Step 6)

### Improving Model Performance:
1. Run backtest (Step 5) - establish baseline
2. Analyze stop-losses (Step 5.5) - on validation data
3. Apply entry filters (Step 5.6) - test on validation data first
4. Compare filters (Step 6.5) - verify filters help
5. Test on test data (Step 5.6) - out-of-sample evaluation
6. Identify trades (Step 6) - with filters applied

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

# Optional: Improve performance with filters
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075
python src/compare_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075

python src/swing_trade_app.py identify --min-probability 0.6 --top-n 20

# With filters (recommended)
python src/identify_trades.py --min-probability 0.6 --top-n 20 --use-recommended-filters
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
- ✅ **Feature normalization/scaling:**
  - Automatic identification of features to scale vs keep as-is
  - StandardScaler applied to price-based and unbounded features
  - Pattern/percentile features kept in original scale (already normalized)
  - Scaler fitted on training data only (prevents data leakage)
  - Scaler saved with model for consistent inference
- ✅ Hyperparameter tuning with RandomizedSearchCV
- ✅ Early stopping to prevent overfitting
- ✅ Cross-validation support
- ✅ Comprehensive evaluation metrics
- ✅ Feature importance display
- ✅ Model versioning and metadata tracking
- ✅ Training curves visualization (optional)
- ✅ Optimized parallelization
- ✅ Fast mode for quicker tuning
- ✅ Configurable data splits (train/val/test dates)
- ✅ Configurable class imbalance handling (imbalance multiplier)
- ✅ Dynamic label column detection (supports different horizons)

### Backtest Step
- ✅ **Data leakage prevention:** Uses only test data (2024-01-01 onwards) by default
- ✅ Configurable exit logic (return threshold, stop-loss, time limit)
- ✅ Stop-loss functionality (2:1 risk-reward by default)
- ✅ Prevents overlapping trades (one position per ticker)
- ✅ Comprehensive metrics (annual return, max capital invested, date range)
- ✅ Exit reason tracking (target_reached, stop_loss, time_limit)
- ✅ Time limit trade analysis

### Stop-Loss Analysis (New)
- ✅ **Pattern identification:** Analyzes stop-loss trades vs winners
- ✅ **Feature analysis:** Identifies which features predict stop-losses
- ✅ **Timing analysis:** Reveals day-of-week and month patterns
- ✅ **Filter recommendations:** Provides specific thresholds to reduce stop-loss rate
- ✅ **Validation data focus:** Uses validation data (2023) by default to prevent data leakage

### Entry Filters (New)
- ✅ **Default filters:** 17 feature-based filters from stop-loss analysis
- ✅ **Timing filters:** Avoids high-risk entry periods (Monday/Tuesday, October)
- ✅ **Custom filters:** Support for user-defined filter thresholds
- ✅ **Integration:** Seamlessly integrates with backtest workflow
- ✅ **Performance comparison:** Compares filtered vs baseline results
- ✅ **Identify pipeline filters:** Entry filters now available in trade identification (Step 6)
  - Recommended filters: 5 key filters based on stop-loss analysis
  - Custom filter support: Add your own feature-based filters
  - Applied before predictions: Reduces false positives proactively

