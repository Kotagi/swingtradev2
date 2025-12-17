# Future Features - Requiring Additional Data

This document lists features that require additional data sources beyond daily OHLCV from yfinance.

## Features Requiring Intraday/Tick Data

These features require volume at price levels, which is only available from intraday or tick data:

### Volume Profile (True Implementation)
- `volume_profile_poc` - Point of Control (price level with most volume)
- `volume_profile_vah` - Value Area High (70% of volume above this)
- `volume_profile_val` - Value Area Low (70% of volume below this)
- `price_vs_poc` - Distance from current price to Point of Control

**Required Data:** Intraday tick data or volume at price levels (e.g., 1-minute bars with volume distribution)

**Alternative:** Can approximate with daily data, but accuracy is limited. True volume profile requires knowing how volume was distributed across price levels during the day.

---

## Features Requiring Market/Sector Data

These features require additional market indices or sector information:

### Relative Strength Features
- `relative_strength_spy` - Stock return vs SPY return
- `relative_strength_sector` - Stock return vs sector return
- `rs_rank_20d` - Relative strength rank (0-100) vs market
- `rs_rank_50d` - Relative strength rank over 50 days
- `rs_momentum` - Rate of change of relative strength
- `outperformance_flag` - Binary: outperforming market

**Required Data:** 
- SPY (or other market index) daily OHLCV data
- Sector index data
- Sector mapping (ticker → sector)

**Note:** SPY data can be downloaded from yfinance, so this is feasible to add later.

### Sector/Industry Features
- `sector_momentum` - Sector performance
- `sector_relative_strength` - Stock vs sector
- `industry_group_rank` - Industry group performance rank
- `sector_trend` - Sector trend direction

**Required Data:**
- Sector mapping file (ticker → sector/industry)
- Sector index data (e.g., XLF for financials, XLK for tech)
- Industry group indices

**Note:** Sector mapping can be obtained from various sources (Yahoo Finance, SEC filings, etc.)

---

## Features Requiring Level 2 Data

These features require order book or Level 2 market data:

### Market Microstructure Features
- `bid_ask_spread` - Bid-ask spread (requires L2 data)
- `order_imbalance` - Order flow imbalance (requires L2 data)
- `price_impact` - Price impact of trades (requires L2 data)

**Required Data:** Level 2 order book data (bid/ask prices and sizes)

**Note:** Not available from yfinance. Would require paid data sources (e.g., Polygon.io, Alpaca, Interactive Brokers API).

---

## Implementation Priority (When Data Available)

### High Priority (Easy to Add)
1. **SPY Data for Relative Strength** - Can download from yfinance
   - Add SPY download to data pipeline
   - Calculate relative strength features
   - Expected impact: Medium-High

### Medium Priority (Moderate Effort)
2. **Sector Mapping & Sector Indices** - Requires data collection
   - Create sector mapping file
   - Download sector ETF data (XLF, XLK, etc.)
   - Calculate sector features
   - Expected impact: Medium

### Low Priority (Complex/Expensive)
3. **Intraday Volume Profile** - Requires paid data or complex setup
   - Set up intraday data source
   - Implement volume profile calculations
   - Expected impact: Medium (approximations work reasonably well)

4. **Level 2 Market Data** - Requires paid API
   - Set up paid data source (Polygon, Alpaca, etc.)
   - Implement microstructure features
   - Expected impact: Low-Medium (more relevant for day trading)

---

## Notes

- **Volume Profile Approximations:** We can create simplified volume profile features using daily data, but they won't be as accurate as true volume profile from intraday data.

- **Relative Strength:** This is the easiest to add - just need to download SPY data alongside individual stocks.

- **Sector Features:** Would require maintaining a sector mapping file and downloading sector ETF data.

- **Market Microstructure:** Most relevant for high-frequency trading. Less important for swing trading (5-day horizon).

