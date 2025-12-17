# Swing Trading ML Application

A comprehensive machine learning application for identifying swing trading patterns in stocks. This application downloads stock data from yfinance, builds feature sets, trains ML models, backtests strategies, and identifies current trading opportunities.

## Features

- **Data Download**: Automated downloading of stock data from yfinance
- **Feature Engineering**: Comprehensive technical indicator features (RSI, MACD, Bollinger Bands, candlestick patterns, etc.)
- **ML Model Training**: XGBoost classifier trained on historical patterns
- **Configurable Parameters**: User-defined trade windows and return thresholds
- **Backtesting**: Enhanced backtesting with multiple strategies and detailed metrics
- **Trade Identification**: Real-time identification of current trading opportunities

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** The project uses `pandas-ta-classic` (compatible with Python 3.11) instead of `pandas-ta` (requires Python 3.12+). The import statement `import pandas_ta as ta` works with both packages.

2. Ensure you have the necessary data directories:
```
data/
  ├── raw/          # Raw OHLCV data
  ├── clean/         # Cleaned data
  ├── features_labeled/  # Features with labels
  └── tickers/       # Ticker lists
```

## Quick Start

### 1. Download Stock Data

Download historical stock data from yfinance:

```bash
python src/swing_trade_app.py download --full
```

Or use the existing download script:
```bash
python src/download_data.py --tickers-file data/tickers/sp500_tickers.csv --start-date 2008-01-01 --raw-folder data/raw --sectors-file data/tickers/sectors.csv
```

### 2. Clean Data (if needed)

If you haven't cleaned the data yet:
```bash
python src/clean_data.py --raw-dir data/raw --clean-dir data/clean
```

### 3. Build Features with Custom Parameters

Build features with your desired trade window and return threshold:

```bash
# Example: 10-day trade window, 5% return threshold
python src/swing_trade_app.py features --horizon 10 --threshold 0.05
```

Or directly:
```bash
python src/feature_pipeline.py \
  --input-dir data/clean \
  --output-dir data/features_labeled \
  --config config/features.yaml \
  --horizon 10 \
  --threshold 0.05
```

### 4. Train Model

Train the ML model on your labeled features:

```bash
python src/swing_trade_app.py train
```

Or with diagnostics:
```bash
python src/swing_trade_app.py train --diagnostics
```

### 5. Run Backtest

Backtest your strategy with configurable parameters:

```bash
# Example: 7-day window, 3% return threshold, $1000 position size
python src/swing_trade_app.py backtest \
  --horizon 7 \
  --return-threshold 0.03 \
  --position-size 1000 \
  --strategy model \
  --model-threshold 0.5
```

Available strategies:
- `model`: Use trained ML model predictions
- `oracle`: Perfect hindsight (uses actual labels)
- `rsi`: RSI oversold strategy (RSI < 30)

### 6. Identify Current Trades

Find current trading opportunities:

```bash
python src/swing_trade_app.py identify \
  --min-probability 0.6 \
  --top-n 20 \
  --output current_opportunities.csv
```

### 7. Run Full Pipeline

Run the complete pipeline in one command:

```bash
python src/swing_trade_app.py full-pipeline \
  --horizon 5 \
  --return-threshold 0.05 \
  --position-size 1000 \
  --min-probability 0.5 \
  --top-n 20
```

## Configuration

### Trading Parameters

Edit `config/trading_config.yaml` to set default parameters:

```yaml
trading:
  horizon: 5              # Trade window in days
  return_threshold: 0.05   # 5% return threshold
  position_size: 1000.0    # Position size per trade

model:
  prediction_threshold: 0.5
  min_probability: 0.5
  top_n_opportunities: 20
```

### Feature Selection

Enable/disable features in `config/features.yaml`:

```yaml
features:
  rsi: 1          # Enable RSI
  macd_line: 0    # Disable MACD line
  ...
```

Select features for training in `config/train_features.yaml`.

## Understanding the Workflow

### 1. Data Pipeline

```
Raw Data (CSV) → Clean Data (Parquet) → Features + Labels (Parquet)
```

- **Raw Data**: Downloaded OHLCV from yfinance
- **Clean Data**: Processed and normalized data
- **Features + Labels**: Technical indicators with forward return labels

### 2. Labeling

Labels are created based on future returns:
- For each day, calculate: `return = (close_{t+horizon} / close_t) - 1`
- Label = 1 if `return > threshold`, else 0

This creates a binary classification problem: will the stock achieve the target return within the trade window?

### 3. Model Training

- XGBoost classifier trained on historical patterns
- Handles class imbalance with `scale_pos_weight`
- Features are selected based on `config/train_features.yaml`

### 4. Backtesting

Backtests simulate trading:
- Entry: When model predicts positive signal (or other strategy)
- Exit: After `horizon` days
- Metrics: Win rate, Sharpe ratio, profit factor, drawdown, etc.

### 5. Trade Identification

Current opportunities are identified by:
- Loading latest features for all tickers
- Running model predictions
- Filtering by probability threshold
- Ranking by confidence

## Command Reference

### Main Application

```bash
python src/swing_trade_app.py [command] [options]
```

**Commands:**
- `download`: Download stock data
- `features`: Build feature set
- `train`: Train ML model
- `backtest`: Run backtest
- `identify`: Identify current trades
- `full-pipeline`: Run complete pipeline

### Individual Scripts

**Download Data:**
```bash
python src/download_data.py \
  --tickers-file data/tickers/sp500_tickers.csv \
  --start-date 2008-01-01 \
  --raw-folder data/raw \
  --sectors-file data/tickers/sectors.csv \
  [--full]
```

**Feature Pipeline:**
```bash
python src/feature_pipeline.py \
  --input-dir data/clean \
  --output-dir data/features_labeled \
  --config config/features.yaml \
  --horizon 5 \
  --threshold 0.0 \
  [--full]
```

**Train Model:**
```bash
python src/train_model.py [--diagnostics]
```

**Enhanced Backtest:**
```bash
python src/enhanced_backtest.py \
  --horizon 5 \
  --return-threshold 0.05 \
  --position-size 1000 \
  --strategy model \
  --model models/xgb_classifier_selected_features.pkl \
  --model-threshold 0.5 \
  [--output trades.csv]
```

**Identify Trades:**
```bash
python src/identify_trades.py \
  --model models/xgb_classifier_selected_features.pkl \
  --min-probability 0.5 \
  --top-n 20 \
  [--output opportunities.csv]
```

## Output Files

- **Models**: `models/xgb_classifier_selected_features.pkl`
- **Backtest Results**: Console output + optional CSV
- **Trade Opportunities**: Console output + optional CSV
- **Logs**: `feature_pipeline.log`

## Performance Metrics

Backtesting provides:
- **Win Rate**: Percentage of profitable trades
- **Average Return**: Mean return per trade
- **Total P&L**: Cumulative profit/loss
- **Sharpe Ratio**: Risk-adjusted return
- **Profit Factor**: Gross profit / Gross loss
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Holding Days**: Mean holding period

## Tips

1. **Start with longer horizons**: 10-20 day windows are more stable
2. **Adjust return thresholds**: Higher thresholds = fewer but higher quality signals
3. **Use multiple strategies**: Compare model vs. oracle vs. RSI
4. **Monitor feature importance**: Use `--diagnostics` to see which features matter
5. **Backtest thoroughly**: Test on different time periods and market conditions
6. **Validate on out-of-sample data**: Don't overfit to training data

## Troubleshooting

**No trading opportunities found:**
- Lower `--min-probability` threshold
- Check that features are up-to-date
- Verify model is trained on matching parameters

**Backtest shows poor performance:**
- Try different trade windows
- Adjust return thresholds
- Check feature quality
- Consider different strategies

**Model training fails:**
- Ensure features are computed
- Check `config/train_features.yaml` has enabled features
- Verify data quality (no excessive NaNs)

## Next Steps

1. Experiment with different feature combinations
2. Try different ML models (Random Forest, Neural Networks)
3. Add more sophisticated entry/exit rules
4. Implement portfolio-level risk management
5. Add real-time data feeds
6. Create a web dashboard for monitoring

## License

This application is provided as-is for educational and research purposes.

