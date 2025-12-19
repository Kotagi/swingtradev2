# Swing Trading ML Application

A comprehensive machine learning application for identifying swing trading opportunities in stocks. Features a modern PyQt6 GUI and a complete CLI pipeline for data download, feature engineering, model training, backtesting, and trade identification.

## Features

- **📊 Data Management**: Automated downloading and cleaning of stock data from yfinance
- **🔧 Feature Engineering**: 57+ technical indicators across 11 categories (price, returns, volatility, momentum, volume, etc.)
- **🤖 ML Model Training**: XGBoost classifier with hyperparameter tuning and cross-validation
- **📈 Backtesting**: Enhanced backtesting with multiple strategies, adaptive stop-losses, and comprehensive metrics
- **🎯 Trade Identification**: Real-time identification of current trading opportunities
- **📉 Stop-Loss Analysis**: Advanced analysis of stop-loss patterns with filter recommendations
- **🎛️ Filter Editor**: Interactive filter editor with SHAP-guided feature importance, real-time impact calculation, and preset management
- **🖥️ Modern GUI**: Full-featured PyQt6 interface with dark theme
- **📊 Model Comparison**: Side-by-side comparison of trained models and backtest results
- **🔍 Advanced Analytics**: Trade log analysis, performance metrics, and visualizations
- **🧠 SHAP Explainability**: Model interpretability with SHAP values - view feature importance, compare models, and recompute explanations

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### GUI (Recommended)

Launch the graphical interface:

```bash
python run_gui.py
```

The GUI provides a complete interface for all operations:
- **Dashboard**: System status and recent opportunities
- **Data Management**: Download and clean stock data
- **Feature Engineering**: Build technical indicators
- **Model Training**: Train ML models with hyperparameter tuning and optional SHAP explainability
- **Backtesting**: Test strategies with comprehensive metrics
- **Analysis**: Trade log viewer, performance metrics, model/backtest comparison, filter editor
- **Stop-Loss Analysis**: Analyze stop-loss patterns and generate filter recommendations
- **Filter Editor**: Interactive filter testing with SHAP importance guidance, impact preview, and preset management
- **Model Comparison**: Compare trained models, view SHAP explanations, and recompute SHAP for existing models
- **Trade Identification**: Find current trading opportunities

### CLI Pipeline

Run the complete pipeline via command line:

```bash
# 1. Download stock data
python src/swing_trade_app.py download --full

# 2. Clean data (automatically included in full pipeline)
python src/swing_trade_app.py clean

# 3. Build features
python src/swing_trade_app.py features

# 4. Train model
python src/swing_trade_app.py train --horizon 30 --return-threshold 0.05

# 5. Run backtest
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --strategy model

# 6. Identify current trades
python src/swing_trade_app.py identify --min-probability 0.6 --top-n 20
```

Or run everything at once:

```bash
python src/swing_trade_app.py full-pipeline --horizon 30 --return-threshold 0.15
```

## Project Structure

```
SwingTradeV2/
├── bats/              # Batch scripts for Windows
├── config/            # Configuration files (features, training)
├── data/              # Data directories
│   ├── raw/          # Raw stock data (CSV)
│   ├── clean/        # Cleaned data (Parquet)
│   ├── features_labeled/  # Feature data (Parquet)
│   ├── backtest_results/  # Backtest output files
│   ├── filter_presets/   # Saved filter presets
│   └── tickers/      # Ticker lists
├── features/         # Feature engineering code
├── gui/              # PyQt6 GUI application
│   ├── tabs/        # GUI tabs (dashboard, data, features, training, etc.)
│   ├── widgets/     # Custom widgets (charts, etc.)
│   └── utils/       # GUI utilities
├── info/             # Documentation (see below)
├── models/           # Trained models and metadata
│   └── shap_artifacts/  # SHAP explanation artifacts
├── scripts/          # Utility scripts
├── src/              # Main application code
│   ├── swing_trade_app.py  # Main CLI entry point
│   ├── download_data.py    # Data download
│   ├── clean_data.py       # Data cleaning
│   ├── feature_pipeline.py # Feature engineering
│   ├── train_model.py      # Model training
│   ├── enhanced_backtest.py # Backtesting
│   ├── identify_trades.py   # Trade identification
│   ├── analyze_stop_losses.py # Stop-loss analysis
│   └── shap_service.py     # SHAP explainability service
├── tests/            # Test files
├── utils/             # Utility modules
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

## Documentation

Comprehensive documentation is available in the `/info` folder:

- **[PIPELINE_STEPS.md](info/PIPELINE_STEPS.md)**: Complete guide to all 7 pipeline steps with detailed arguments and examples
- **[FEATURE_GUIDE.md](info/FEATURE_GUIDE.md)**: Comprehensive documentation for all 57 technical indicators
- **[QUICKSTART.md](info/QUICKSTART.md)**: Quick start guide
- **[INSTALL.md](info/INSTALL.md)**: Installation instructions
- **[FEATURE_ROADMAP.md](info/FEATURE_ROADMAP.md)**: Planned features and improvements
- **[FEATURE_REDUNDANCY_GUIDE.md](info/FEATURE_REDUNDANCY_GUIDE.md)**: Guide to feature selection and redundancy

See the `/info` folder for complete documentation.

## Key Features

### 57+ Technical Indicators

The application computes 57 technical indicators across 11 categories:
- **Price Features**: Raw price, log price, price vs MA200
- **Return Features**: Daily, weekly, monthly, quarterly, YTD returns
- **52-Week Features**: Distance to highs/lows, position in range
- **Moving Averages**: SMA/EMA ratios, slopes, crossovers
- **Volatility**: ATR, Bollinger Bands, TTM Squeeze, volatility ratios
- **Volume**: Log volume, relative volume, Chaikin Money Flow, OBV momentum
- **Momentum**: RSI, MACD, PPO, ROC, Stochastic, CCI, Williams %R
- **Market Context**: Beta vs SPY
- **Candlestick**: Body, wick percentages
- **Price Action**: Higher highs/lows, Donchian channels
- **Trend**: Trend residual, ADX, Aroon indicators

### Advanced Backtesting

- **Multiple Strategies**: Model-based, oracle (perfect hindsight), RSI
- **Adaptive Stop-Losses**: ATR-based stops that adapt to volatility
- **Comprehensive Metrics**: Win rate, Sharpe ratio, profit factor, max drawdown, annual return
- **Exit Reason Tracking**: Target reached, stop-loss, time limit, end of data
- **Position Management**: Prevents overlapping positions per ticker

### Stop-Loss Analysis

- **Pattern Identification**: Analyzes stop-loss trades vs winners
- **Feature Analysis**: Identifies which features predict stop-losses
- **Filter Recommendations**: Generates specific thresholds to reduce stop-loss rate
- **Impact Preview**: Estimates improvement before applying filters
- **Preset Management**: Save and load filter presets

### Model Training

- **Hyperparameter Tuning**: RandomizedSearchCV with configurable iterations
- **Cross-Validation**: Optional CV for more robust evaluation
- **Early Stopping**: Prevents overfitting
- **Feature Scaling**: Automatic identification and scaling of features
- **Comprehensive Metrics**: ROC AUC, Precision, Recall, F1, Average Precision
- **Model Registry**: Automatic registration and comparison of trained models
- **SHAP Explainability**: Optional SHAP computation for model interpretability (view feature importance, understand predictions)

## Configuration

### Feature Selection

Edit `config/features.yaml` to enable/disable features:

```yaml
features:
  rsi14: 1          # Enable RSI
  macd_histogram_normalized: 1  # Enable MACD histogram
  # ... etc
```

### Training Features

Edit `config/train_features.yaml` to select features for model training.

### Trading Parameters

Edit `config/trading_config.yaml` for default trading parameters.

## Requirements

- Python 3.11 or 3.12+
- See `requirements.txt` for complete dependency list

Key dependencies:
- `pandas>=1.3`
- `numpy>=1.21`
- `yfinance>=0.2.0`
- `xgboost>=1.5.0`
- `scikit-learn>=1.0.0`
- `PyQt6>=6.6.0` (for GUI)
- `matplotlib>=3.4`
- `pandas-ta-classic` (Python 3.11 compatible)
- `shap>=0.41.0` (optional, for model explainability)

## Examples

### Train a Model for 30-Day Swings with 15% Target

```bash
# Build features (labels calculated during training)
python src/swing_trade_app.py features

# Train model
python src/swing_trade_app.py train \
  --horizon 30 \
  --return-threshold 0.15 \
  --tune --n-iter 30

# Backtest
python src/swing_trade_app.py backtest \
  --horizon 30 \
  --return-threshold 0.15 \
  --strategy model \
  --model-threshold 0.5 \
  --stop-loss-mode adaptive_atr
```

### Analyze Stop-Losses and Apply Filters

```bash
# Analyze stop-loss patterns
python src/analyze_stop_losses.py \
  --horizon 30 \
  --return-threshold 0.15 \
  --model-threshold 0.80 \
  --stop-loss-mode adaptive_atr

# Apply recommended filters
python src/apply_entry_filters.py \
  --horizon 30 \
  --return-threshold 0.15 \
  --model-threshold 0.80 \
  --stop-loss-mode adaptive_atr
```

## License

This application is provided as-is for educational and research purposes.

## Contributing

This is a personal project. For questions or issues, please refer to the documentation in the `/info` folder.
