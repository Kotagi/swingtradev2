# Phase 1 Feature Analysis - Data Requirements

This document categorizes Phase 1 features by what can be implemented with current data (OHLCV from yfinance) vs what requires additional data sources.

## Current Data Available
- **OHLCV**: Open, High, Low, Close, Volume (daily)
- **Date index**: Datetime index for time-based calculations
- **No additional data**: No intraday, no market indices, no sector data

---

## ✅ **CAN IMPLEMENT NOW** (Current OHLCV Data)

### **1. Multi-Timeframe Features (Weekly Patterns)** - ✅ ALL 14 features
**Status:** ✅ **100% implementable**

All weekly features can be created by resampling daily data to weekly, then forward-filling back to daily.

- ✅ `weekly_return_1w` - Resample to weekly, calculate return
- ✅ `weekly_return_2w` - Resample to weekly, calculate 2-week return
- ✅ `weekly_return_4w` - Resample to weekly, calculate 4-week return
- ✅ `weekly_sma_5w` - Resample to weekly, calculate SMA
- ✅ `weekly_sma_10w` - Resample to weekly, calculate SMA
- ✅ `weekly_sma_20w` - Resample to weekly, calculate SMA
- ✅ `weekly_ema_5w` - Resample to weekly, calculate EMA
- ✅ `weekly_ema_10w` - Resample to weekly, calculate EMA
- ✅ `weekly_rsi_14w` - Resample to weekly, calculate RSI
- ✅ `weekly_macd_histogram` - Resample to weekly, calculate MACD
- ✅ `close_vs_weekly_sma20` - Compare daily close to weekly SMA
- ✅ `weekly_volume_ratio` - Resample volume, calculate ratio
- ✅ `weekly_atr_pct` - Resample to weekly, calculate ATR %
- ✅ `weekly_trend_strength` - Slope of weekly SMA

**Implementation:** Use `pd.resample('W')` to convert daily to weekly, compute indicators, then `ffill()` back to daily.

---

### **2. Volatility Regime Features** - ✅ ALL 10 features
**Status:** ✅ **100% implementable**

All volatility features can be calculated from ATR and Bollinger Bands (which we already compute).

- ✅ `volatility_regime` - ATR percentile over 252 days
- ✅ `volatility_trend` - ATR slope (linear regression)
- ✅ `bb_squeeze` - BB width percentile (low = squeeze)
- ✅ `bb_expansion` - BB width percentile (high = expansion)
- ✅ `atr_ratio_20d` - Current ATR / 20-day ATR average
- ✅ `atr_ratio_252d` - Current ATR / 252-day ATR average
- ✅ `volatility_percentile_20d` - ATR percentile over 20 days
- ✅ `volatility_percentile_252d` - ATR percentile over 252 days
- ✅ `high_volatility_flag` - Binary: ATR > 75th percentile
- ✅ `low_volatility_flag` - Binary: ATR < 25th percentile

**Implementation:** Use existing `atr` and `bb_width` features, add percentile and ratio calculations.

---

### **3. Trend Strength & Quality Features** - ✅ ALL 11 features
**Status:** ✅ **100% implementable**

All trend features can be calculated from price and existing moving averages.

- ✅ `trend_strength_20d` - ADX over 20 days (pandas-ta has ADX)
- ✅ `trend_strength_50d` - ADX over 50 days
- ✅ `trend_consistency` - % of days price above/below MA
- ✅ `ema_alignment` - Check if EMAs are aligned (bullish/bearish)
- ✅ `sma_slope_20d` - Linear regression slope of 20-day SMA
- ✅ `sma_slope_50d` - Linear regression slope of 50-day SMA
- ✅ `ema_slope_20d` - Linear regression slope of 20-day EMA
- ✅ `trend_duration` - Days since last trend change
- ✅ `trend_reversal_signal` - Detect potential reversals
- ✅ `price_vs_all_mas` - Count of MAs price is above

**Implementation:** Use existing EMAs/SMAs, add ADX with different periods, add slope calculations.

---

### **4. Support & Resistance Levels** - ✅ ALL 12 features
**Status:** ✅ **100% implementable**

All support/resistance features can be calculated from rolling min/max of price.

- ✅ `resistance_level_20d` - Rolling 20-day high
- ✅ `resistance_level_50d` - Rolling 50-day high
- ✅ `support_level_20d` - Rolling 20-day low
- ✅ `support_level_50d` - Rolling 50-day low
- ✅ `distance_to_resistance` - % distance to nearest resistance
- ✅ `distance_to_support` - % distance to nearest support
- ✅ `price_near_resistance` - Binary: within 2% of resistance
- ✅ `price_near_support` - Binary: within 2% of support
- ✅ `resistance_touches` - Count of times price touched resistance
- ✅ `support_touches` - Count of times price touched support
- ✅ `pivot_point` - Classic pivot: (High + Low + Close) / 3
- ✅ `fibonacci_levels` - Distance to Fibonacci retracement levels (calculated from swing high/low)

**Implementation:** Use `rolling().max()` and `rolling().min()` for support/resistance, calculate distances and touches.

---

### **5. Volume Profile & Distribution Features** - ⚠️ PARTIAL (8/12 features)
**Status:** ⚠️ **67% implementable** (8 out of 12 features)

Some volume features require intraday data (volume at price levels), but many can be calculated from daily OHLCV.

#### ✅ **Can implement (8 features):**
- ✅ `volume_weighted_price` - VWAP (can approximate from daily OHLCV)
- ✅ `price_vs_vwap` - Distance from price to VWAP
- ✅ `vwap_slope` - VWAP trend direction
- ✅ `volume_climax` - Unusually high volume days (percentile)
- ✅ `volume_dry_up` - Unusually low volume days (percentile)
- ✅ `volume_trend` - Volume moving average slope
- ✅ `volume_breakout` - Volume spike on price breakout
- ✅ `volume_distribution` - Volume concentration metric (can approximate)

#### ❌ **Requires intraday data (4 features):**
- ❌ `volume_profile_poc` - Point of Control (needs volume at each price level)
- ❌ `volume_profile_vah` - Value Area High (needs volume distribution)
- ❌ `volume_profile_val` - Value Area Low (needs volume distribution)
- ❌ `price_vs_poc` - Distance to POC (depends on POC)

**Note:** True volume profile (POC, VAH, VAL) requires intraday tick data or at minimum, volume at price levels. However, we can create approximations using daily high/low/close and volume, though they won't be as accurate.

**Implementation:** 
- VWAP: Can calculate from daily data using typical price: `(High + Low + Close) / 3 * Volume`
- Volume climax/dry up: Use percentile calculations
- Volume trend: Linear regression on volume MA
- Volume breakout: Detect volume spikes on price breakouts

---

## ❌ **REQUIRES ADDITIONAL DATA**

### **Volume Profile (True POC/VAH/VAL)**
**Required:** Intraday tick data or volume at price levels
**Alternative:** Can approximate with daily data, but less accurate

### **Relative Strength Features** (Phase 2, but mentioned here)
**Required:** 
- SPY data (market index)
- Sector index data
- Sector mapping (ticker → sector)

**Can add:** We could download SPY data from yfinance and add relative strength features.

---

## 📊 **Summary**

### **Phase 1 Implementation Status:**

| Category | Total Features | Can Implement | Requires Data | % Ready |
|----------|---------------|---------------|---------------|---------|
| **1. Multi-Timeframe** | 14 | 14 | 0 | 100% ✅ |
| **2. Volatility Regime** | 10 | 10 | 0 | 100% ✅ |
| **3. Trend Strength** | 11 | 11 | 0 | 100% ✅ |
| **4. Support/Resistance** | 12 | 12 | 0 | 100% ✅ |
| **5. Volume Profile** | 12 | 8 | 4 | 67% ⚠️ |
| **TOTAL** | **59** | **55** | **4** | **93%** ✅ |

### **Recommendation:**

**Start with Categories 1-4 (47 features)** - All 100% implementable with current data.

**Then add Category 5 (8 features)** - The implementable volume features.

**Skip for now:** True volume profile POC/VAH/VAL (requires intraday data).

**Total implementable now: 55 features** out of 59 (93%)

---

## 🚀 **Implementation Order**

### **Batch 1: Quick Wins (Can implement today)**
1. Support/Resistance (12 features) - Simplest, just rolling min/max
2. Volatility Regime (10 features) - Use existing ATR/BB
3. Trend Strength (11 features) - Use existing MAs, add ADX

**Total: 33 features**

### **Batch 2: Multi-Timeframe (Requires resampling)**
4. Multi-Timeframe Weekly (14 features) - Need to implement resampling logic

**Total: 14 features**

### **Batch 3: Volume Features**
5. Volume Profile (8 features) - VWAP, volume patterns, etc.

**Total: 8 features**

---

## 📝 **Notes**

1. **VWAP Approximation:** True VWAP requires intraday data, but we can approximate using:
   - Typical Price: `(High + Low + Close) / 3`
   - Volume-weighted: `sum(Typical Price * Volume) / sum(Volume)`
   - This is less accurate than true VWAP but still useful.

2. **Volume Profile Approximation:** We could create a simplified version using:
   - Price range distribution (High-Low)
   - Volume allocation across price ranges
   - But this won't be as accurate as true volume profile from tick data.

3. **Relative Strength:** Could add SPY download to enable relative strength features (Phase 2).

4. **Sector Data:** Would require sector mapping file and sector index downloads (Phase 2/3).

---

## ✅ **Action Items**

**Immediate (Can start now):**
- ✅ Implement Support/Resistance features (12)
- ✅ Implement Volatility Regime features (10)
- ✅ Implement Trend Strength features (11)
- ✅ Implement Volume Profile (approximations) (8)

**Next:**
- Implement Multi-Timeframe features (14) - requires resampling logic

**Future:**
- Consider adding SPY data download for relative strength
- Consider adding sector mapping for sector features

