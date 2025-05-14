# SwingTradeV1

A lightweight pipeline to download, clean, and validate historical stock data for algorithmic trading research.

---

## 📁 Project Structure

```
SwingTradeV1/
├── data/
│   ├── raw/           # Downloaded “as-is” CSVs from data provider
│   └── clean/         # Post-processed CSVs, ready for analysis
├── src/
│   ├── download_data.py   # Ingest raw price data for a list of tickers
│   └── clean_data.py      # Clean, fill, adjust, and save CSVs
├── tests/
│   └── test_pipeline.py   # Integration tests for download + cleaning
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
   cd SwingTradev1
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
  By default, the scripts use the tickers defined in `src/download_data.py`. You can override on the command line.

---

## 💾 Data Ingestion

Downloads raw CSVs into `data/raw/`.

```bash
python src/download_data.py \
  --tickers AAPL MSFT AMZN GOOGL \
  --output-dir data/raw/
```

- `--tickers`  
  Space-separated list of symbols you want to download.  
- `--output-dir`  
  Where to save the raw files.

---

## 🧹 Data Cleaning

Processes every CSV in `data/raw/` and writes clean versions to `data/clean/`.

```bash
python src/clean_data.py \
  --input-dir data/raw/ \
  --output-dir data/clean/
```

Cleans in three main steps:

1. **Index & type normalization**  
   - Parse date column to `DatetimeIndex`  
   - Enforce numeric dtypes on OHLCV columns  
2. **Dedup & sort**  
   - Drop exact duplicates  
   - Sort by timestamp  
3. **Fill & adjust**  
   - Forward/backfill missing bars  
   - Apply split/dividend adjustment (if available)  

---

## ✅ Testing

Run the integration tests (downloads + cleaning) on a small set of tickers:

```bash
pytest
```

You should see something like:

```
8 passed in 0.70s
```

> **Tip**: If you remove symbols (e.g. TSLA) from your default list, tests will adapt automatically.

---

## 📦 requirements.txt

```text
pandas>=1.3
numpy>=1.21
requests>=2.25
pytest>=7.0
python-dotenv>=0.19
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 🛣️ Next Steps (Phase 1)

- [x] Data ingestion (`download_data.py`)
- [x] Data cleaning (`clean_data.py`)
- [x] Validate on sample tickers (AAPL, MSFT, AMZN, GOOGL)
- [x] Write README instructions
- [x] Commit all Phase 1 code

With Phase 1 complete, you can now move on to:

> **Phase 2: Feature Engineering**  
> Derive technical indicators, rolling statistics, and feature sets for modeling.

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
