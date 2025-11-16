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

### **Step 2: Clean Data** (Optional if data already cleaned)
**Purpose:** Clean and normalize raw stock data

**Command:**
```bash
python src/clean_data.py --raw-dir data/raw --clean-dir data/clean
```

Or using short flags:
```bash
python src/clean_data.py -r data/raw -c data/clean
```

**What it does:**
- Renames columns to lowercase
- Parses and validates dates
- Removes invalid/duplicate rows
- Applies split adjustments to prices
- Converts to Parquet format for efficiency

**Output:** Cleaned Parquet files in `data/clean/` (one per ticker)

**Note:** This step is typically only needed once or when raw data is updated.

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
- Loads cleaned data from `data/clean/`
- Computes 50+ technical indicators (RSI, MACD, Bollinger Bands, candlestick patterns, etc.)
- Creates binary labels: 1 if future return > threshold, else 0
- Saves feature + label data to `data/features_labeled/`

**Output:** Feature Parquet files in `data/features_labeled/` (one per ticker)

**Key Features Computed:**
- Price indicators (SMA, EMA, moving averages)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (ATR, Bollinger Bands)
- Volume indicators (OBV, volume ratios)
- Candlestick patterns (engulfing, hammer, doji, etc.)
- Ichimoku components

---

### **Step 4: Train ML Model**
**Purpose:** Train XGBoost classifier to predict profitable trades

**Command:**
```bash
python src/swing_trade_app.py train
```

**Options:**
- `--diagnostics`: Show feature importance and SHAP diagnostics

**What it does:**
- Loads all feature data from `data/features_labeled/`
- Filters features based on `config/train_features.yaml`
- Splits data into train/test sets (by date: train up to 2022-12-31)
- Trains baseline models (DummyClassifier, LogisticRegression)
- Trains XGBoost classifier with class imbalance handling
- Evaluates performance (ROC AUC, precision, recall)
- Saves model to `models/xgb_classifier_selected_features.pkl`

**Output:** 
- Trained model file: `models/xgb_classifier_selected_features.pkl`
- Console metrics: ROC AUC, classification report

**Metrics Shown:**
- ROC AUC score (higher is better, >0.65 is decent)
- Precision/Recall for both classes
- Feature importance (if --diagnostics used)

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
- Executes steps 1-6 in sequence
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

---

## Quick Reference Commands

```bash
# Complete workflow
python src/swing_trade_app.py full-pipeline --horizon 5 --return-threshold 0.05

# Individual steps
python src/swing_trade_app.py download --full
python src/swing_trade_app.py features --horizon 5 --threshold 0.05
python src/swing_trade_app.py train
python src/swing_trade_app.py backtest --horizon 5 --return-threshold 0.05
python src/swing_trade_app.py identify --min-probability 0.6 --top-n 20
```

