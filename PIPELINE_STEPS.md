# Complete Pipeline Steps

## Overview
Your swing trading ML pipeline consists of 7 main steps that transform raw stock data into actionable trading signals.

---

## Step-by-Step Pipeline

### **Step 1: Download Stock Data**
**Purpose:** Download historical OHLCV (Open, High, Low, Close, Volume) data from yfinance, including shares outstanding and public float from SEC EDGAR

**Command (via main app):**
```bash
python src/swing_trade_app.py download --full
```

**Command (direct):**
```bash
python src/download_data.py --tickers-file data/tickers/sp500_tickers.csv --start-date 2008-01-01 --end-date 2025-12-31
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tickers-file` | `data/tickers/sp500_tickers.csv` | Path to file with one ticker symbol per line |
| `--start-date` | `2008-01-01` | Start date for historical data (YYYY-MM-DD) |
| `--end-date` | Today's date (if not specified) | End date for historical data (YYYY-MM-DD, exclusive). If omitted, defaults to today |
| `--raw-folder` | `data/raw` | Directory where raw CSV files are saved |
| `--sectors-file` | `data/tickers/sectors.csv` | Output file for ticker→sector mapping |
| `--chunk-size` | `100` | Number of tickers per bulk HTTP request |
| `--pause` | `1.0` | Initial seconds to sleep between chunks (adaptive) |
| `--max-retries` | `3` | Maximum retry attempts for failed downloads |
| `--full` | `False` | Force full redownload of all tickers (ignore existing CSVs) |
| `--no-sectors` | `False` | Skip fetching and writing sector information |
| `--resume` | `False` | Resume from last checkpoint (skips already completed tickers) |

**What it does:**
- Downloads raw stock data for all tickers in the tickers file
- Fetches split factors and sector information
- Saves individual CSV files to `data/raw/` with columns: Open, High, Low, Close, Volume, split_coefficient
- Creates/updates `data/tickers/sectors.csv`
- Downloads SPY data for market context features
- **Incremental updates:** Only downloads new data by default (use `--full` to force redownload)
- **Resume capability:** Use `--resume` to continue from where you left off
- **Adaptive rate limiting:** Automatically adjusts pause time between chunks based on performance

**Output:** Raw CSV files in `data/raw/` (one per ticker) with OHLCV data and split coefficients

**Examples:**
```bash
# Full redownload of all tickers
python src/swing_trade_app.py download --full

# Download specific date range
python src/download_data.py --start-date 2020-01-01 --end-date 2024-12-31

# Resume from checkpoint (skip already downloaded tickers)
python src/download_data.py --resume

# Custom tickers file and output location
python src/download_data.py --tickers-file data/tickers/custom_tickers.csv --raw-folder data/raw_custom

# Skip sector information
python src/download_data.py --no-sectors

# Increase retry attempts for unreliable connections
python src/download_data.py --max-retries 5
```

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

**All Available Arguments:**

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--raw-dir` | `-r` | `data/raw` | Input directory containing raw CSV files |
| `--clean-dir` | `-c` | `data/clean` | Output directory where cleaned Parquet files will be written |
| `--resume` | | `False` | Skip files that have already been cleaned (resume from previous run) |
| `--workers` | `-w` | `4` | Number of parallel workers for processing |
| `--verbose` | `-v` | `False` | Show detailed cleaning steps (DEBUG logs) |

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
**Purpose:** Compute technical indicators (labels are now calculated during training)

**Command:**
```bash
python src/swing_trade_app.py features
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/features.yaml` | Feature configuration file path |
| `--input-dir` | `data/clean` | Input directory containing cleaned Parquet files |
| `--output-dir` | `data/features_labeled` | Output directory for feature Parquet files |
| `--feature-set` | `None` | Feature set name (e.g., 'v1', 'v2'). If specified, automatically sets config and output-dir. Default: uses explicit paths |
| `--full` / `--force-full` | `False` | Force full recomputation (ignores cache, recomputes all tickers) |

**What it does:**
- **Parallel processing:** Uses joblib for multi-core feature computation
- **Feature validation:** Automatically checks for infinities, excessive NaNs, and constant values
- **Robust error handling:** Continues processing other tickers if one fails
- **Caching:** Skips up-to-date files automatically (use `--full` to force recompute)
- **Performance optimized:** Uses efficient DataFrame concatenation (no fragmentation warnings)
- Loads cleaned data from `data/clean/`
- Computes **57 technical indicators** across multiple categories
- **Data leakage prevention:** Forward-looking features removed from registry
- **Note:** Labels are no longer created during feature engineering. They are calculated on-the-fly during training based on the horizon and return threshold specified in the training step.
- Saves feature data to `data/features_labeled/`
- **Summary statistics:** Reports features computed, validation issues, processing time

**Output:** Feature Parquet files in `data/features_labeled/` (one per ticker)

**Key Features Computed (57 total):**

**Price Features (3):**
- **price**: Raw closing price (for filtering, not ML input)
- **price_log**: Log-transformed closing price (ln(close)) - compresses price differences
- **price_vs_ma200**: Price normalized to 200-day MA (close / SMA200) - long-term context

**Return Features (6):**
- **daily_return**: Daily return % clipped to ±20% - daily momentum
- **gap_pct**: Gap % (open - prev_close) / prev_close, clipped to ±20% - overnight momentum
- **weekly_return_5d**: 5-day return % clipped to ±30% - weekly momentum
- **monthly_return_21d**: 21-day return % clipped to ±50% - monthly momentum
- **quarterly_return_63d**: 63-day return % clipped to ±100% - quarterly momentum
- **ytd_return**: Year-to-Date return % clipped to (-1, +2) - YTD performance

**52-Week Features (3):**
- **dist_52w_high**: Distance to 52-week high clipped to (-1, 0.5) - breakout potential
- **dist_52w_low**: Distance to 52-week low clipped to (-0.5, 2) - support level
- **pos_52w**: Position within 52-week range (0=low, 1=high) - relative strength

**Moving Average Features (8):**
- **sma20_ratio**: Price / SMA20 clipped to [0.5, 1.5] - short-term trend
- **sma50_ratio**: Price / SMA50 clipped to [0.5, 1.5] - medium-term trend
- **sma200_ratio**: Price / SMA200 clipped to [0.5, 2.0] - long-term trend
- **sma20_sma50_ratio**: SMA20 / SMA50 clipped to [0.8, 1.2] - short/medium crossover
- **sma50_sma200_ratio**: SMA50 / SMA200 clipped to [0.6, 1.4] - Golden Cross/Death Cross
- **sma50_slope**: 5-day change in SMA50 / close clipped to [-0.1, 0.1] - medium-term momentum
- **sma200_slope**: 10-day change in SMA200 / close clipped to [-0.1, 0.1] - long-term momentum
- **kama_slope**: KAMA (Kaufman Adaptive Moving Average) slope normalized by price, clipped to [-0.1, 0.1] - adaptive trend strength, works better in choppy tickers

**Volatility Features (8):**
- **volatility_5d**: 5-day rolling std of returns clipped to [0, 0.15] - short-term volatility
- **volatility_21d**: 21-day rolling std of returns clipped to [0, 0.15] - medium-term volatility
- **volatility_ratio**: Volatility ratio (vol5/vol21) clipped to [0, 2] - identifies volatility expansion/compression regimes, helps differentiate choppy vs trending markets
- **atr14_normalized**: ATR14 / close clipped to [0, 0.2] - true range volatility (accounts for gaps)
- **bollinger_band_width**: Log-normalized BB width (log1p((upper-lower)/mid)) - volatility compression/expansion, predicts breakouts
- **ttm_squeeze_on**: Binary flag (0/1) - TTM Squeeze condition (BB inside KC) - volatility contraction, #1 breakout indicator
- **ttm_squeeze_momentum**: Normalized momentum (close - SMA20) / close - momentum direction during squeeze
- **volatility_of_volatility**: Measures instability of volatility itself (low=stable regime, high=chaotic regime), normalized to [0, 3] - tells model if volatility indicators are reliable

**Volume Features (5):**
- **log_volume**: Log-transformed volume (log1p(volume)) - normalized volume
- **log_avg_volume_20d**: Log-transformed 20-day average volume - smoothed volume baseline
- **relative_volume**: Log-transformed volume ratio (log1p(volume/vol_avg20)) clipped to [0,10] - unusual volume activity
- **chaikin_money_flow**: CMF (20-period) clipped to [-1, 1] - accumulation vs distribution, combines price with volume
- **obv_momentum**: OBV rate of change (10-day pct change) clipped to [-0.5, 0.5] - volume acceleration, works with breakouts and squeezes

**Momentum Features (9):**
- **rsi14**: RSI14 centered to [-1, +1] range: (rsi-50)/50 - overbought/oversold
- **macd_histogram_normalized**: MACD histogram / close - momentum acceleration/deceleration, most predictive part of MACD
- **ppo_histogram**: PPO (Percentage Price Oscillator) histogram clipped to [-0.2, 0.2] - percentage-based momentum acceleration/deceleration, scale-invariant and cross-ticker comparable
- **dpo**: DPO (Detrended Price Oscillator, 20-period) normalized by price, clipped to [-0.2, 0.2] - cyclical indicator that removes long-term trend, highlights short-term cycles, helps detect cycle peaks/troughs and mean-reversion zones
- **roc10**: ROC (Rate of Change) 10-period clipped to [-0.5, 0.5] - short-term momentum velocity, highly predictive in breakouts and pullbacks
- **roc20**: ROC (Rate of Change) 20-period clipped to [-0.7, 0.7] - medium-term momentum velocity, more expressive form of momentum than basic log returns
- **stochastic_k14**: Stochastic %K (14-period) in [0, 1] range - position within trading range, better than RSI in many scenarios
- **cci20**: CCI (Commodity Channel Index, 20-period) normalized with tanh(cci/100) - standardized distance from trend oscillator, captures trend exhaustion & reversion points
- **williams_r14**: Williams %R (14-period) normalized to [0, 1] range - range momentum/reversion oscillator, very sensitive to reversal points, strong complement to RSI and Stoch

**Market Context (1):**
- **beta_spy_252d**: Rolling beta vs SPY (252-day) normalized to [0, 1] - market correlation

**Candlestick Features (3):**
- **candle_body_pct**: Body / range in [0, 1] - candle strength
- **candle_upper_wick_pct**: Upper wick / range in [0, 1] - selling pressure at highs
- **candle_lower_wick_pct**: Lower wick / range in [0, 1] - buying pressure at lows

**Price Action Features (4):**
- **higher_high_10d**: Binary flag (0/1) - current close > previous 10-day max - bullish momentum
- **higher_low_10d**: Binary flag (0/1) - current close > previous 10-day min - trend continuation
- **donchian_position**: Position within Donchian channel (20-period) in [0, 1] range - breakout structure
- **donchian_breakout**: Binary flag (0/1) - close > donchian_high_20 - breakout detection

**Trend Features (8):**
- **trend_residual**: Deviation from linear trend (50-day regression) clipped to [-0.2, 0.2] - noise vs trend
- **adx14**: ADX (14-period) normalized to [0, 1] range (adx/100) - trend strength independent of direction, fills gap in trend strength measurement
- **aroon_up**: Aroon Up (25-period) normalized to [0, 1] range - days since highest high, uptrend maturity (fresh/maturing/exhausted)
- **aroon_down**: Aroon Down (25-period) normalized to [0, 1] range - days since lowest low, downtrend maturity (starting/ending)
- **aroon_oscillator**: Aroon Oscillator (25-period) normalized to [0, 1] range - trend dominance indicator (aroon_up - aroon_down), net trend pressure, helps identify early trend reversals
- **fractal_dimension_index**: Measures price path roughness (1.0-1.3=smooth/trending, 1.6-1.8=choppy/noisy), normalized to [0, 1] - helps identify trend-friendly vs whipsaw environments
- **hurst_exponent**: Quantifies return persistence (H>0.5=trending/persistent, H<0.5=mean-reverting, H≈0.5=random walk), in [0, 1] - tells model if momentum should be trusted
- **price_curvature**: Second derivative of trend (acceleration), positive=trend bending up, negative=trend bending down, normalized to [-0.05, 0.05] - helps catch early reversals and blow-off moves

**Feature Organization:**
- Features are organized by category: price, returns, 52-week, moving averages, volatility, volume, momentum, market context, candlestick, price action, and trend
- All 57 features are currently enabled in `config/features.yaml`
- See `FEATURE_GUIDE.md` for complete feature documentation including:
  - Detailed calculation methods for each feature
  - Normalization techniques used
  - Feature characteristics and value propositions
  - Complete reference guide for all 37 features
- See `FEATURE_REDUNDANCY_GUIDE.md` for feature selection recommendations
- See `FEATURE_ROADMAP.md` for complete feature roadmap and future additions
- See `features/metadata.py` for feature metadata and categories

**Performance Notes:**
- Feature computation is optimized with efficient DataFrame operations
- Processing time scales with number of features and tickers (expect ~1-2 minutes for 493 tickers with 57 features)

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

# Train model with custom name
python src/swing_trade_app.py train --model-output models/my_experiment_v1.pkl

# Train model with feature set (auto-named)
python src/swing_trade_app.py train --feature-set v2

# Train model with feature set and custom name
python src/swing_trade_app.py train --feature-set v2 --model-output models/custom_v2_model.pkl

# Train model with custom date range (exclude older data)
python src/swing_trade_app.py train --train-start 2020-01-01 --train-end 2022-12-31

# Train model with specific date range
python src/swing_trade_app.py train --train-start 2018-01-01 --train-end 2021-12-31 --val-end 2022-12-31
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tune` | `False` | Perform hyperparameter tuning (RandomizedSearchCV) - **Recommended** |
| `--n-iter` | `20` | Number of iterations for hyperparameter search |
| `--cv` | `False` | Use cross-validation for hyperparameter tuning (slower but more robust) |
| `--cv-folds` | `None` (auto: 3, or 2 if `--fast`) | Number of CV folds. Lower = faster but less robust |
| `--no-early-stop` | `False` | Disable early stopping (train for full n_estimators) |
| `--plots` | `False` | Generate training curves and feature importance plots (requires matplotlib) |
| `--diagnostics` | `False` | Compute and print SHAP diagnostics (requires shap library) |
| `--fast` | `False` | Use faster hyperparameter tuning (reduced search space, fewer CV folds). ~3-5x faster but slightly less optimal |
| `--imbalance-multiplier` | `1.0` | Multiplier for class imbalance handling. Increase to 2.0-3.0 to favor positive class more. Higher = more trades predicted |
| `--train-start` | `None` (use all available) | Training data start date (YYYY-MM-DD). Default: None (use all available data from the beginning). Use to exclude older data (e.g., '2020-01-01' to start from 2020) |
| `--train-end` | `None` (defaults to `2022-12-31`) | Training data end date (YYYY-MM-DD). Use more recent dates (e.g., 2020-12-31) for recent market patterns |
| `--val-end` | `None` (defaults to `2023-12-31`) | Validation data end date (YYYY-MM-DD) |
| `--horizon` | `None` (auto-detected) | Trade horizon in trading days (e.g., 5, 30). Used for on-the-fly label calculation during training |
| `--return-threshold` | `None` | Return threshold for label calculation (as decimal, e.g., 0.05 for 5%). Used for on-the-fly label calculation during training |
| `--label-col` | `None` (auto-detected) | Label column name (e.g., 'label_5d', 'label_30d'). Auto-detected if not specified. **Note:** Labels are now calculated during training, not during feature engineering |
| `--feature-set` | `None` (defaults to 'v1') | Feature set name (e.g., 'v1', 'v2'). If specified, automatically sets data directory and train config |
| `--model-output` | `None` (auto-named) | Custom model output file path (e.g., 'models/my_custom_model.pkl' or 'my_custom_model.pkl'). If not specified, uses default naming based on feature set or 'xgb_classifier_selected_features.pkl' |

**What it does:**
- **Enhanced data splitting:** Train/Validation/Test split by date:
  - Train: up to 2022-12-31
  - Validation: 2023-01-01 to 2023-12-31 (for early stopping)
  - Test: 2024-01-01 onwards
- Loads all feature data from `data/features_labeled/`
- **Calculates labels on-the-fly:** Creates binary labels based on `--horizon` (trading days) and `--return-threshold` before data preparation. Labels are computed using future returns, not pre-computed during feature engineering.
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
- Trained model file: `models/xgb_classifier_selected_features.pkl` (default) or custom path if `--model-output` specified (includes model, scaler, features, metadata)
- Training metadata: `models/training_metadata.json`
- Training curves plot: `models/xgb_training_curves.png` (if `--plots` used)
- Feature importance chart: `models/feature_importance_chart.png` (if `--plots` used)
- Console metrics: Comprehensive evaluation metrics including feature normalization summary

**Model Naming:**
- **Default:** `models/xgb_classifier_selected_features.pkl`
- **With feature set:** `models/xgb_classifier_selected_features_{feature_set}.pkl` (e.g., `xgb_classifier_selected_features_v2.pkl`)
- **Custom name:** Use `--model-output` to specify a custom path (e.g., `--model-output models/my_experiment_v1.pkl` or `--model-output my_experiment_v1.pkl`)

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
python src/swing_trade_app.py backtest --horizon 5 --return-threshold 0.05 --strategy model --model-threshold 0.5 --position-size 1000
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--horizon` | *Required* | Trade window in days (must match features) |
| `--return-threshold` | *Required* | Return threshold (e.g., 0.05 for 5%, must match features) |
| `--strategy` | `model` | Backtest strategy: `model` (use trained model), `oracle` (perfect hindsight), `rsi` (RSI < 30) |
| `--model` | `models/xgb_classifier_selected_features.pkl` | Path to trained model (required for model strategy) |
| `--model-threshold` | `0.5` | Probability threshold for model signals (0.0-1.0) |
| `--position-size` | `1000.0` | Position size per trade in dollars |
| `--stop-loss` | `None` (auto: 2:1 risk-reward) | Stop-loss threshold as decimal (e.g., -0.075 for -7.5%). If not specified and return-threshold is provided, uses 2:1 risk-reward (return_threshold / 2). **DEPRECATED:** Use `--stop-loss-mode` and related args for adaptive stops. |
| `--stop-loss-mode` | `None` (auto: constant) | Stop-loss mode: `constant` (fixed percentage) or `adaptive_atr` (ATR-based per-trade stops) |
| `--atr-stop-k` | `1.8` | ATR multiplier for adaptive stops. Only used when `--stop-loss-mode=adaptive_atr`. Formula: `stop_pct = clamp(atr_stop_k * atr14_normalized, atr_stop_min_pct, atr_stop_max_pct)` |
| `--atr-stop-min-pct` | `0.04` | Minimum stop distance for adaptive stops (e.g., 0.04 = 4%). Only used when `--stop-loss-mode=adaptive_atr` |
| `--atr-stop-max-pct` | `0.10` | Maximum stop distance for adaptive stops (e.g., 0.10 = 10%). Only used when `--stop-loss-mode=adaptive_atr` |
| `--output` | `None` | Optional CSV file to save trades |
| `--data-dir` | `data/features_labeled` | Directory containing feature Parquet files |
| `--tickers-file` | `data/tickers/sp500_tickers.csv` | Path to CSV file with ticker symbols |
| `--test-start-date` | `2024-01-01` | Start date for test data (YYYY-MM-DD). Only data after this date will be backtested. Default: 2024-01-01 (to avoid data leakage from training/validation) |

**What it does:**
- Loads trained model, scaler, and feature data
- **Applies feature scaling:** Automatically scales features during prediction using the saved scaler
- Generates entry signals based on strategy
- Simulates trades with configurable exit logic:
  - Exit at return threshold (if reached)
  - Exit at stop-loss (2:1 risk-reward by default, configurable)
  - Exit at time horizon (if neither threshold reached)
- **Stop-loss modes:**
  - **Constant mode** (default): Uses a fixed stop-loss percentage for all trades (e.g., -7.5%)
  - **Adaptive ATR mode**: Calculates per-trade stop-loss based on ATR (Average True Range) at entry time
    - Formula: `stop_pct = clamp(atr_stop_k * atr14_normalized, atr_stop_min_pct, atr_stop_max_pct)`
    - Example: If ATR is 1.5% of price and `atr_stop_k=1.8`, stop = 2.7% → clamped to min 4% → final stop = 4%
    - Adapts to volatility: tighter stops for low-volatility stocks, wider stops for high-volatility stocks
- Prevents overlapping trades (only one position per ticker at a time)
- Calculates returns and P&L for each trade
- Tracks exit reasons (target_reached, stop_loss, time_limit, end_of_data)
- Aggregates performance metrics

**Stop-Loss Examples:**
```bash
# Constant stop-loss (traditional, 7.5% fixed)
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --stop-loss -0.075

# Adaptive ATR-based stops (default settings: k=1.8, min=4%, max=10%)
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --stop-loss-mode adaptive_atr

# Adaptive ATR with tighter stops (k=1.5, min=3%, max=8%) - Good for lower volatility environments or more conservative trading
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --stop-loss-mode adaptive_atr --atr-stop-k 1.5 --atr-stop-min-pct 0.03 --atr-stop-max-pct 0.08

# Adaptive ATR with wider stops (k=2.2, min=5%, max=12%) - Good for higher volatility stocks or more aggressive risk tolerance
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --stop-loss-mode adaptive_atr --atr-stop-k 2.2 --atr-stop-min-pct 0.05 --atr-stop-max-pct 0.12

# Adaptive ATR with very tight range (k=1.8, min=4%, max=6%) - Forces stops to be between 4-6% regardless of ATR
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --stop-loss-mode adaptive_atr --atr-stop-k 1.8 --atr-stop-min-pct 0.04 --atr-stop-max-pct 0.06

# Adaptive ATR with higher multiplier (k=2.5) - Gives more room for volatile stocks while still respecting min/max bounds
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --stop-loss-mode adaptive_atr --atr-stop-k 2.5 --atr-stop-min-pct 0.04 --atr-stop-max-pct 0.10
```

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
- **Adaptive Stop-Loss Statistics** (when using `--stop-loss-mode=adaptive_atr`):
  - Average stop distance used
  - Minimum and maximum stop distances
  - Stop distance distribution (buckets: 0-4%, 4-5%, 5-7%, 7-10%, >10%)

**Choosing Adaptive Stop-Loss Settings:**

| Setting | ATR K | Min % | Max % | Use Case |
|---------|-------|-------|-------|----------|
| **Conservative** | 1.5 | 3% | 8% | Lower volatility stocks, tighter risk control, shorter horizons |
| **Default** | 1.8 | 4% | 10% | Balanced approach, works well for most swing trades |
| **Aggressive** | 2.2 | 5% | 12% | Higher volatility stocks, more room for price movement |
| **Very Aggressive** | 2.5 | 5% | 15% | Very volatile stocks, longer horizons, higher risk tolerance |

**Guidelines:**
- **Lower ATR K (1.5-1.8)**: Tighter stops, better for low-volatility stocks or conservative traders
- **Higher ATR K (2.0-2.5)**: Wider stops, better for high-volatility stocks or aggressive traders
- **Tighter min/max range**: Forces stops into narrower band (e.g., 4-6%), less adaptation
- **Wider min/max range**: Allows more adaptation (e.g., 3-12%), more responsive to volatility
- **Start with defaults** (k=1.8, min=4%, max=10%) and adjust based on backtest results

---

### **Step 5.5: Analyze Stop-Loss Trades** (Optional)
**Purpose:** Analyze patterns in stop-loss trades to identify risk factors and improve entry filters

**Command:**
```bash
# Constant stop-loss (traditional)
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075

# Adaptive ATR-based stops (default settings)
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr

# Adaptive ATR with custom settings
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 2.0 --atr-stop-min-pct 0.05 --atr-stop-max-pct 0.12
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--horizon` | `30` | Trade window in days (for running backtest if trades-csv not provided, must match features) |
| `--return-threshold` | `0.15` | Return threshold (for running backtest if trades-csv not provided, must match features) |
| `--model-threshold` | `0.85` | Model probability threshold (for running backtest if trades-csv not provided) |
| `--stop-loss` | `None` (auto: 2:1 risk-reward) | Stop-loss threshold (e.g., -0.075 for -7.5%, for running backtest if trades-csv not provided). **DEPRECATED:** Use `--stop-loss-mode` and related args for adaptive stops. |
| `--stop-loss-mode` | `None` (auto: constant) | Stop-loss mode: `constant` (fixed) or `adaptive_atr` (ATR-based per-trade stops) |
| `--atr-stop-k` | `1.8` | ATR multiplier for adaptive stops. Only used when `--stop-loss-mode=adaptive_atr` |
| `--atr-stop-min-pct` | `0.04` | Minimum stop distance for adaptive stops (e.g., 0.04 = 4%). Only used when `--stop-loss-mode=adaptive_atr` |
| `--atr-stop-max-pct` | `0.10` | Maximum stop distance for adaptive stops (e.g., 0.10 = 10%). Only used when `--stop-loss-mode=adaptive_atr` |
| `--trades-csv` | `None` | CSV file with existing backtest trades (if not provided, will run backtest) |
| `--use-validation` | `True` | Use validation data (2023) for analysis instead of test data |
| `--train-end` | `2022-12-31` | Training data end date (YYYY-MM-DD) |
| `--val-end` | `2023-12-31` | Validation data end date (YYYY-MM-DD) |
| `--output` | `None` | Optional JSON file to save analysis results |

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
# Constant stop-loss with default filters (traditional)
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075

# Adaptive ATR-based stops with default filters (recommended)
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr

# Adaptive ATR with tighter stops (conservative: k=1.5, min=3%, max=8%)
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 1.5 --atr-stop-min-pct 0.03 --atr-stop-max-pct 0.08

# Adaptive ATR with wider stops (aggressive: k=2.2, min=5%, max=12%)
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 2.2 --atr-stop-min-pct 0.05 --atr-stop-max-pct 0.12

# Disable timing filters with adaptive stops
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --stop-loss-mode adaptive_atr --no-timing-filters

# Add custom filters with adaptive stops
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --stop-loss-mode adaptive_atr --atr-stop-k 1.8 --custom-filter rsi_slope ">" 0.5 --custom-filter market_correlation_20d ">" 0.6

# Conservative adaptive stops with custom filters
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 1.5 --atr-stop-min-pct 0.03 --atr-stop-max-pct 0.07 --custom-filter daily_return ">" -0.02 --custom-filter higher_low_10d ">" 0.43
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--horizon` | *Required* | Trade window in days (must match features) |
| `--return-threshold` | *Required* | Return threshold (e.g., 0.15 for 15%, must match features) |
| `--model-threshold` | `0.80` | Model probability threshold |
| `--stop-loss` | `None` (auto: 2:1 risk-reward) | Stop-loss threshold (e.g., -0.075 for -7.5%) |
| `--position-size` | `1000.0` | Position size per trade in dollars |
| `--test-start-date` | `2024-01-01` | Test data start date (YYYY-MM-DD) |
| `--test-end-date` | `None` (all available) | Test data end date (YYYY-MM-DD) |
| `--use-default-filters` | `True` | Use default filters from stop-loss analysis |
| `--no-timing-filters` | `False` | Disable timing filters (Monday/Tuesday, October) |
| `--custom-filter` | `None` | Add custom filter (can be used multiple times). Format: `--custom-filter FEATURE OPERATOR THRESHOLD` (e.g., `--custom-filter rsi_slope ">" 0.5`) |

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
python src/swing_trade_app.py identify --min-probability 0.6 --top-n 20 --output current_opportunities.csv

# With recommended filters (reduces false positives)
python src/identify_trades.py --min-probability 0.6 --top-n 20 --use-recommended-filters --output current_opportunities.csv

# With custom filters
python src/identify_trades.py --min-probability 0.6 --top-n 20 --custom-filter candle_body_pct ">" -10 --custom-filter close_position_in_range ">" 0.40 --custom-filter weekly_rsi_14w "<" 42 --output current_opportunities.csv
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-probability` | `0.5` | Minimum prediction probability to consider (0.0-1.0) |
| `--top-n` | `20` | Maximum number of opportunities to return |
| `--output` | `None` | Optional CSV file to save results |
| `--model` | `models/xgb_classifier_selected_features.pkl` | Path to trained model pickle file |
| `--data-dir` | `data/features_labeled` | Directory containing feature Parquet files |
| `--tickers-file` | `data/tickers/sp500_tickers.csv` | Path to CSV file with ticker symbols |
| `--use-recommended-filters` | `False` | Use recommended filters from stop-loss analysis |
| `--custom-filter` | `None` | Add custom filter (can be used multiple times). Format: `--custom-filter FEATURE OPERATOR THRESHOLD` (e.g., `--custom-filter candle_body_pct ">" -10`). Operators: `>`, `<`, `>=`, `<=` |

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
# Constant stop-loss (baseline comparison)
python src/compare_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075

# Adaptive ATR-based stops (default settings)
python src/compare_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr

# Adaptive ATR with conservative settings (tighter stops)
python src/compare_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 1.5 --atr-stop-min-pct 0.03 --atr-stop-max-pct 0.08

# Adaptive ATR with aggressive settings (wider stops)
python src/compare_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 2.2 --atr-stop-min-pct 0.05 --atr-stop-max-pct 0.12
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--horizon` | *Required* | Trade window in days (must match features) |
| `--return-threshold` | *Required* | Return threshold (e.g., 0.15 for 15%, must match features) |
| `--model-threshold` | `0.80` | Model probability threshold |
| `--stop-loss` | `None` (auto: 2:1 risk-reward) | Stop-loss threshold (e.g., -0.075 for -7.5%). **DEPRECATED:** Use `--stop-loss-mode` and related args for adaptive stops. |
| `--stop-loss-mode` | `None` (auto: constant) | Stop-loss mode: `constant` (fixed) or `adaptive_atr` (ATR-based per-trade stops) |
| `--atr-stop-k` | `1.8` | ATR multiplier for adaptive stops. Only used when `--stop-loss-mode=adaptive_atr` |
| `--atr-stop-min-pct` | `0.04` | Minimum stop distance for adaptive stops (e.g., 0.04 = 4%). Only used when `--stop-loss-mode=adaptive_atr` |
| `--atr-stop-max-pct` | `0.10` | Maximum stop distance for adaptive stops (e.g., 0.10 = 10%). Only used when `--stop-loss-mode=adaptive_atr` |
| `--position-size` | `1000.0` | Position size per trade in dollars |
| `--test-start-date` | `2024-01-01` | Test data start date (YYYY-MM-DD) |
| `--test-end-date` | `None` (all available) | Test data end date (YYYY-MM-DD) |

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
python src/swing_trade_app.py full-pipeline --horizon 5 --return-threshold 0.05 --position-size 1000 --min-probability 0.5 --top-n 20
```

**All Available Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--horizon` | *Required* | Trade window in days |
| `--return-threshold` | *Required* | Return threshold (e.g., 0.05 for 5%) |
| `--position-size` | `1000.0` | Position size per trade in dollars |
| `--model-threshold` | `0.5` | Model probability threshold |
| `--min-probability` | `0.5` | Minimum prediction probability for trade identification |
| `--top-n` | `20` | Maximum number of opportunities to return |
| `--full-download` | `False` | Force full data redownload |
| `--full-features` | `False` | Force full feature recomputation |
| `--skip-download` | `False` | Skip data download step |
| `--skip-train` | `False` | Skip model training step |

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
1. Build features (Step 3) - features are independent of horizon/threshold
2. Train model (Step 4) - specify horizon and return-threshold here (labels calculated on-the-fly)
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

- **`config/features.yaml`**: Enable/disable technical indicators (all 57 features enabled by default)
- **`config/train_features.yaml`**: Select features for model training (all 57 features enabled by default)
- **`config/trading_config.yaml`**: Default trading parameters
- **`data/tickers/sp500_tickers.csv`**: List of tickers to analyze

## Documentation Files

- **`FEATURE_GUIDE.md`**: **Complete feature documentation** - Comprehensive guide to all 37 features including:
  - Detailed calculation methods for each feature
  - Normalization techniques and ranges
  - Feature characteristics and value propositions
  - Complete reference guide organized by category
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
python src/swing_trade_app.py features
python src/swing_trade_app.py train --horizon 30 --return-threshold 0.05
python src/swing_trade_app.py backtest --horizon 5 --return-threshold 0.05

# Optional: Improve performance with filters
# Constant stop-loss (traditional)
# Constant stop-loss examples (traditional)
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075
python src/compare_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss -0.075

# Adaptive ATR-based stops (default: k=1.8, min=4%, max=10%)
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr
python src/compare_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr

# Adaptive ATR with conservative settings (tighter stops: k=1.5, min=3%, max=8%)
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 1.5 --atr-stop-min-pct 0.03 --atr-stop-max-pct 0.08
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 1.5 --atr-stop-min-pct 0.03 --atr-stop-max-pct 0.08

# Adaptive ATR with aggressive settings (wider stops: k=2.2, min=5%, max=12%)
python src/analyze_stop_losses.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 2.2 --atr-stop-min-pct 0.05 --atr-stop-max-pct 0.12
python src/apply_entry_filters.py --horizon 30 --return-threshold 0.15 --model-threshold 0.80 --stop-loss-mode adaptive_atr --atr-stop-k 2.2 --atr-stop-min-pct 0.05 --atr-stop-max-pct 0.12

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
- ✅ **57 features** across 11 categories (price, returns, 52-week, moving averages, volatility, volume, momentum, market context, candlestick, price action, trend)
- ✅ Performance optimized (efficient DataFrame operations, no fragmentation)
- ✅ Feature validation (infinities, NaNs, constant values)
- ✅ Data leakage prevention (forward-looking features removed)
- ✅ Standardized column name handling
- ✅ Parallel processing with joblib
- ✅ Comprehensive summary statistics
- ✅ Feature metadata system with categories
- ✅ Ownership features removed (not consistent and cannot be trusted)

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
- ✅ **Adaptive ATR-based stops:** Per-trade stop-loss calculation based on volatility (Phase 1)
  - Constant mode: Fixed stop-loss percentage for all trades
  - Adaptive ATR mode: Stop-loss adapts to each stock's volatility at entry time
  - Formula: `stop_pct = clamp(atr_stop_k * atr14_normalized, min_pct, max_pct)`
  - Configurable parameters: ATR multiplier (k), min/max stop distances
  - Reporting: Average, min, max stop distances and distribution buckets
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

