# SwingTradeV1

A lightweight pipeline to download, clean, backtest, and (soon) model historical stock data for algorithmic trading research.

---

## 📁 Project Structure

```
SwingTradeV1/
├── data/
│   ├── raw/           # Downloaded “as-is” CSVs from data provider
│   ├── clean/         # Post-processed CSVs, ready for analysis
│   └── tickers/       # Ticker lists and sector mappings
├── reports/
│   ├── trade_logs/           # Generated trade logs CSVs
│   ├── performance_metrics/  # Backtest evaluation metrics & reports
│   └── plots/                # Performance charts (PNG)
├── src/
│   ├── download_data.py      # Ingest raw price data and write sector lookup
│   ├── clean_data.py         # Clean, fill, adjust, and save CSVs
│   ├── feature_engineering.py# Compute technical indicator features
│   ├── backtest.py           # Rule-based backtesting engine
│   ├── evaluate.py           # Performance metrics calculator
│   └── plot_performance.py   # Static performance chart generator
├── tests/
│   └── test_pipeline.py      # Integration tests for download + cleaning
├── README.md
└── requirements.txt
```

---

## ⚙️ Prerequisites

- Python 3.8 or higher  
- A free API key from your data provider (e.g. Alpha Vantage, if configured)  
- `git` (for cloning and version control)

---

## 🚀 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/SwingTradev1.git
   cd SwingTradeV1
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Configuration

- **Environment variable**  
  ```bash
  export ALPHA_VANTAGE_API_KEY=your_api_key_here
  ```
  (Or set the equivalent in your OS.)

- **Ticker list**  
  By default, the scripts use the tickers defined in `data/tickers/sp500_tickers.csv`.

---

## 💾 Phase 1 – Data Ingestion & Cleaning

1. **Download raw data**  
   ```bash
   python src/download_data.py      -f data/tickers/sp500_tickers.csv      -s 2008-01-01      -e 2025-05-17      -r data/raw      -o data/tickers/sectors.csv
   ```

2. **Clean data**  
   ```bash
   python src/clean_data.py      -i data/raw      -o data/clean
   ```

3. **Validate**  
   ```bash
   pytest
   ```

---

## 📊 Phase 2 – Rule-Based Backtester

1. **Backtest**  
   ```bash
   python src/backtest.py      -t AAPL MSFT AMZN GOOGL      --clean-dir data/clean      --sectors-file data/tickers/sectors.csv      --momentum-threshold 0.05      --stop-loss-atr-mult 2      --time-exit-days 5      --slippage 0      --commission-per-trade 0      --commission-per-share 0      -i 100000      --max-positions 8      --max-sector-exposure 0.25      -o reports/trade_logs/trade_log.csv
   ```

2. **Evaluate**  
   ```bash
   python src/evaluate.py      -l reports/trade_logs/trade_log.csv      -i 100000      -o reports/performance_metrics/performance_metrics.csv
   ```

3. **Plot**  
   ```bash
   python src/plot_performance.py      -l reports/trade_logs/trade_log.csv      -m reports/performance_metrics/performance_metrics.csv      -o reports/plots
   ```

---

## 🤖 Phase 3 – Feature Engineering & Basic ML

1. **Feature List**  
   - 5-day return, 10-day return  
   - ATR(14), Bollinger Band width  
   - EMA(12/26) crossover value  
   - OBV, RSI(14)

2. **Engineering Script**  
   Build `src/feature_engineering.py` to compute & merge features per ticker/day, and save the feature matrix + future-return label to `data/clean/features.parquet`.

3. **Labeling**  
   Define target = 1 if return over next 5 days > 0%; else 0.

4. **Train/Test Splits**  
   Implement walk-forward splits (e.g. train on 2008–2015, test 2016; slide quarterly).

5. **Model Training**  
   Train an XGBoost classifier on features. Use Optuna to tune basic hyperparameters.

6. **Evaluation**  
   Track classification metrics (accuracy, precision/recall), and compare ML-driven backtest vs rule-only backtest.

7. **Integration**  
   Modify `src/backtest.py` to accept ML signals in place of rule entry and re-run backtest; log results to reports/.

8. **Commit**  
   Commit Phase 3 code & findings.

---

## 🤝 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add some feature"`)  
4. Push to your branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
