# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Workflow

### 1. Download Data
```bash
python src/swing_trade_app.py download --full
```

### 2. Clean Data (if needed)
```bash
python src/clean_data.py --raw-dir data/raw --clean-dir data/clean
```

### 3. Build Features (with your parameters)
```bash
# Example: 10-day trade window, 5% return threshold
python src/swing_trade_app.py features --horizon 10 --threshold 0.05
```

### 4. Train Model
```bash
python src/swing_trade_app.py train
```

### 5. Backtest
```bash
# Test with same parameters you used for features
python src/swing_trade_app.py backtest --horizon 10 --return-threshold 0.05
```

### 6. Find Current Trades
```bash
python src/swing_trade_app.py identify --min-probability 0.6 --top-n 10
```

## One-Command Pipeline

Run everything at once:

```bash
python src/swing_trade_app.py full-pipeline \
  --horizon 5 \
  --return-threshold 0.05
```

## Key Parameters

- **horizon**: Trade window in days (how long to hold)
- **return-threshold**: Minimum return to consider a win (0.05 = 5%)
- **min-probability**: Minimum model confidence for trade identification (0.0-1.0)
- **position-size**: Dollar amount per trade for backtesting

## Example: Custom Swing Trade Setup

For a 7-day swing trade looking for 3% returns:

```bash
# 1. Build features
python src/swing_trade_app.py features --horizon 7 --threshold 0.03

# 2. Train model
python src/swing_trade_app.py train

# 3. Backtest
python src/swing_trade_app.py backtest \
  --horizon 7 \
  --return-threshold 0.03 \
  --position-size 1000

# 4. Find opportunities
python src/swing_trade_app.py identify \
  --min-probability 0.55 \
  --top-n 15
```

## Tips

- Start with longer horizons (10-20 days) for more stable patterns
- Adjust return thresholds based on market conditions
- Use `--diagnostics` when training to see feature importance
- Compare different strategies: `--strategy model`, `--strategy oracle`, `--strategy rsi`

