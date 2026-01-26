# Swing Trading System Roadmap: Phases 4–9

Below is a detailed checklist for Phases 4 through 9, following Phase 3.  
Each phase is broken into scaffolding, feature implementation, testing, integration, evaluation, and documentation steps.

---

## Phase 4 – Advanced Technical & Volume Features

### 1. Scaffolding
- [ ] Update `features/technical.py` to include new modules or classes if needed
- [ ] Add new feature stubs to `features/technical.py`
- [ ] Update `config/features.yaml` to list Phase 4 features
- [ ] Update `tests/test_features.py` to include tests for new features

### 2. Feature Implementation & Testing
For each feature below:
1. Stub function in `features/technical.py` with docstring  
2. Unit test in `tests/test_features.py`  
3. Implement calculation  
4. Run `pytest`  
5. Register in `features/registry.py`  
6. Smoke test on sample ticker

#### Advanced Technical Features
- [ ] Feature: 20-day & 50-day SMA/EMA variants  
- [ ] Feature: Exponential Moving Average Ribbon measurements  
- [ ] Feature: Standard Deviation of returns (volatility proxy)  
- [ ] Feature: ATR-based dynamic stop-loss levels (multiples: 1.5×, 2.5×)  
- [ ] Feature: Momentum indicators (Rate of Change, ROC)  
- [ ] Feature: Volume Profile (Price Volume trend, PVT)  

#### Volume & Liquidity Features
- [ ] Feature: Relative Volume (current vs. 20-day average)  
- [ ] Feature: On-Balance Volume variants (OBV rolling mean)  
- [ ] Feature: Money Flow Index (MFI)  
- [ ] Feature: Chaikin Money Flow (CMF)  
- [ ] Feature: Volume Weighted Average Price (VWAP)  
- [ ] Feature: Liquidity measures (bid-ask spread, average daily dollar volume)  

### 3. Integration & Pipeline
- [ ] Update `scripts/build_features.py` to include Phase 4 features  
- [ ] Run pipeline on sample tickers, inspect `data/clean/features.parquet`  
- [ ] Add data quality assertions (no NaNs, correct ranges)

### 4. Evaluation
- [ ] Retrain XGBoost model in `scripts/train_model.py` with updated feature set  
- [ ] Compare classification metrics vs Phase 3 baseline  
- [ ] Run `scripts/evaluate_phase3.py` renamed for Phase 4 evaluation  
- [ ] Document performance lift or degradation in `reports/phase4_evaluation.md`

### 5. Documentation & Versioning
- [ ] Update `README.md` with Phase 4 summary  
- [ ] Bump version to `3.1-alpha`  
- [ ] Commit, tag (`v3.1-alpha`), and push changes

---

## Phase 5 – Fundamentals & Catalysts

### 1. Scaffolding
- [ ] Create `features/fundamental.py`  
- [ ] Update `features/registry.py` to import new module  
- [ ] Update `config/features.yaml` for Phase 5  
- [ ] Add tests in `tests/test_features.py`

### 2. Feature Implementation & Testing
For each fundamental feature:
1. Stub + docstring  
2. Unit test with synthetic or sample financial data  
3. Implement and verify  
4. Register + smoke test

- [ ] Price/Earnings (P/E) Ratio  
- [ ] Price/Book (P/B) Ratio  
- [ ] PEG Ratio  
- [ ] EV/EBITDA  
- [ ] Debt-to-Equity Ratio  
- [ ] Free Cash Flow per share  
- [ ] Revenue & Earnings growth rates (QoQ, YoY)  
- [ ] Earnings Surprise magnitude (actual vs consensus)  
- [ ] Dividend Yield & Ex-Dividend Date handling  
- [ ] Institutional ownership % and recent changes  
- [ ] Analyst upgrades/downgrades (count in last 30 days)  
- [ ] Upcoming earnings flag (binary within N days)  

### 3. Integration & Pipeline
- [ ] Extend `build_features.py` to fetch/merge fundamental data sources  
- [ ] Handle missing fundamental data gracefully with logging  
- [ ] Save updated `features.parquet`

### 4. Evaluation
- [ ] Retrain and evaluate ML model (script refactor for Phase 5)  
- [ ] Compare precision/recall vs Phase 4  
- [ ] Generate `reports/phase5_evaluation.md`

### 5. Documentation & Versioning
- [ ] Update documentation  
- [ ] Bump version to `3.2-alpha`  
- [ ] Commit & tag `v3.2-alpha`

---

## Phase 6 – Sentiment & News Features

### 1. Scaffolding
- [ ] Create `features/sentiment.py`  
- [ ] Update registry & config for Phase 6  
- [ ] Add tests in `tests/test_features.py`

### 2. Feature Implementation & Testing
- [ ] News sentiment score (NLP on headlines)  
- [ ] News volume spike indicator  
- [ ] Social media sentiment (Twitter/StockTwits score)  
- [ ] Mention volume trend (change in counts)  
- [ ] Analyst sentiment trend (change in consensus rating)  
- [ ] Options put/call ratio and IV skew  
- [ ] Sentiment momentum (delta over time)  
- [ ] Earnings call tone analysis (optional)  

### 3. Integration & Pipeline
- [ ] Integrate external APIs or local data sources  
- [ ] Add retry/backoff for API calls  
- [ ] Merge sentiment features into pipeline  

### 4. Evaluation
- [ ] Retrain and evaluate (Phase 6)  
- [ ] Document incremental lift in `reports/phase6_evaluation.md`

### 5. Documentation & Versioning
- [ ] Bump version to `3.3-alpha`  
- [ ] Commit & tag

---

## Phase 7 – Macro & Sector Context

### 1. Scaffolding
- [ ] Create `features/macro.py`  
- [ ] Registry & config updates  
- [ ] Tests addition  

### 2. Feature Implementation & Testing
- [ ] Broad market trend (S&P 500 moving average)  
- [ ] VIX level & change  
- [ ] Interest rate trend (10y yield)  
- [ ] CPI/Inflation rate change  
- [ ] Sector ETF returns (1m, 3m)  
- [ ] Market breadth (Advance/Decline line)  
- [ ] New highs vs lows ratio  
- [ ] Currency index trends (USD)  
- [ ] Commodity price trends (oil, gold)  
- [ ] Major macro event flags  

### 3. Integration & Pipeline
- [ ] Fetch macro data (FRED, Quandl)  
- [ ] Merge and align with daily dates  

### 4. Evaluation
- [ ] Retrain & evaluate Phase 7 model  
- [ ] Document in `reports/phase7_evaluation.md`

### 5. Documentation & Versioning
- [ ] Bump version to `3.4-alpha`  
- [ ] Commit & tag

---

## Phase 8 – Alternative Data & Market Structure

### 1. Scaffolding
- [ ] Create `features/altdata.py`  
- [ ] Registry & config updates  
- [ ] Tests for altdata

### 2. Feature Implementation & Testing
- [ ] Insider trade cluster detection  
- [ ] Short interest & days-to-cover  
- [ ] Float and free float percentage  
- [ ] Bid-ask spread  
- [ ] Volume profile support/resistance  
- [ ] Order book imbalance metrics  
- [ ] Web traffic & search trend spikes  
- [ ] App download ranking (if applicable)  
- [ ] Satellite imagery parking lot analysis (optional)  
- [ ] ESG controversy flags  

### 3. Integration & Pipeline
- [ ] Connect to alt-data providers  
- [ ] Handle data delays & missing windows  

### 4. Evaluation
- [ ] Retrain & evaluate Phase 8  
- [ ] Report in `reports/phase8_evaluation.md`

### 5. Documentation & Versioning
- [ ] Bump version to `3.5-alpha`  
- [ ] Commit & tag

---

## Phase 9 – End-to-End Integration & Deployment

### 1. Final Model Consolidation
- [ ] Combine all feature modules  
- [ ] Retrain final “full” ML model  
- [ ] Evaluate on hold-out period

### 2. Robust Backtesting Framework
- [ ] Validate backtest performance with slippage, transaction cost modeling  
- [ ] Stress-test under extreme conditions (2008 crisis, COVID-19 crash)  

### 3. Deployment & Monitoring
- [ ] Package pipeline into CLI or microservice  
- [ ] Deploy on server or cloud  
- [ ] Set up scheduled data refresh & backtest runs  
- [ ] Build dashboards for performance & data quality alerts  

### 4. Documentation & Versioning
- [ ] Update `README.md` with full roadmap summary  
- [ ] Bump version to `3.0-final` or `4.0`  
- [ ] Tag final release  

---

*End of Phases 4–9 Detailed Roadmap*  
