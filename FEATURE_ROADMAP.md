# Feature Roadmap - Comprehensive Feature List

This document outlines all recommended features to add to improve model performance, organized by priority and category.

## Current Feature Count: ~50 features
## Target: ~100-150 features for optimal ML performance

---

## 🔥 **PRIORITY 1: High-Impact Features** (Implement First)

### **1. Multi-Timeframe Features** (Weekly Patterns)
**Why:** Daily patterns alone miss longer-term trends. Weekly context is crucial for swing trading.

- `weekly_return_1w` - 1-week log return
- `weekly_return_2w` - 2-week log return  
- `weekly_return_4w` - 4-week log return
- `weekly_sma_5w` - 5-week SMA
- `weekly_sma_10w` - 10-week SMA
- `weekly_sma_20w` - 20-week SMA
- `weekly_ema_5w` - 5-week EMA
- `weekly_ema_10w` - 10-week EMA
- `weekly_rsi_14w` - Weekly RSI
- `weekly_macd_histogram` - Weekly MACD histogram
- `close_vs_weekly_sma20` - Price vs 20-week SMA
- `weekly_volume_ratio` - Weekly volume vs average
- `weekly_atr_pct` - Weekly ATR as % of price
- `weekly_trend_strength` - Slope of weekly SMA

**Implementation:** Resample daily data to weekly, compute indicators, then forward-fill back to daily.

---

### **2. Volatility Regime Features**
**Why:** Market behavior changes dramatically in high vs low volatility. Critical for risk management.

- `volatility_regime` - Current ATR percentile (0-100) over 252 days
- `volatility_trend` - ATR slope (increasing/decreasing volatility)
- `bb_squeeze` - Bollinger Band squeeze indicator (low volatility)
- `bb_expansion` - Bollinger Band expansion (high volatility)
- `atr_ratio_20d` - Current ATR / 20-day ATR average
- `atr_ratio_252d` - Current ATR / 252-day ATR average
- `volatility_percentile_20d` - ATR percentile over 20 days
- `volatility_percentile_252d` - ATR percentile over 252 days
- `high_volatility_flag` - Binary: ATR > 75th percentile
- `low_volatility_flag` - Binary: ATR < 25th percentile

---

### **3. Trend Strength & Quality Features**
**Why:** Distinguishes strong trends from weak ones. Better signal quality.

- `trend_strength_20d` - ADX (Average Directional Index) - already have ADX_14, add 20d
- `trend_strength_50d` - ADX over 50 days
- `trend_consistency` - % of days price above/below MA over lookback
- `ema_alignment` - All EMAs aligned (bullish/bearish/neutral)
- `sma_slope_20d` - Slope of 20-day SMA
- `sma_slope_50d` - Slope of 50-day SMA
- `ema_slope_20d` - Slope of 20-day EMA
- `trend_duration` - Days since last trend change
- `trend_reversal_signal` - Potential trend reversal indicator
- `price_vs_all_mas` - Count of MAs price is above

---

### **4. Volume Profile & Distribution Features**
**Why:** Volume confirms price moves. Better volume analysis = better signals.

- `volume_profile_poc` - Point of Control (price level with most volume)
- `volume_profile_vah` - Value Area High
- `volume_profile_val` - Value Area Low
- `price_vs_poc` - Distance from current price to POC
- `volume_weighted_price` - VWAP (Volume Weighted Average Price)
- `price_vs_vwap` - Distance from price to VWAP
- `vwap_slope` - VWAP trend direction
- `volume_distribution` - Volume concentration metric
- `volume_climax` - Unusually high volume days
- `volume_dry_up` - Unusually low volume days
- `volume_trend` - Volume moving average slope
- `volume_breakout` - Volume spike on price breakout

---

### **5. Support & Resistance Levels**
**Why:** Key price levels are where reversals happen. Critical for entry/exit.

- `resistance_level_20d` - Nearest resistance (20-day high)
- `resistance_level_50d` - Nearest resistance (50-day high)
- `support_level_20d` - Nearest support (20-day low)
- `support_level_50d` - Nearest support (50-day low)
- `distance_to_resistance` - % distance to nearest resistance
- `distance_to_support` - % distance to nearest support
- `price_near_resistance` - Binary: within 2% of resistance
- `price_near_support` - Binary: within 2% of support
- `resistance_touches` - Number of times price touched resistance
- `support_touches` - Number of times price touched support
- `pivot_point` - Classic pivot point calculation
- `fibonacci_levels` - Distance to Fibonacci retracement levels

---

## 🎯 **PRIORITY 2: Medium-Impact Features** (Implement Second)

### **6. Advanced Momentum Indicators**
**Why:** More momentum perspectives = better signal confirmation.

- `roc_10d` - Rate of Change (10-day)
- `roc_20d` - Rate of Change (20-day)
- `momentum_10d` - Price momentum (close - close[10])
- `momentum_20d` - Price momentum (close - close[20])
- `williams_r` - Williams %R oscillator
- `cci` - Commodity Channel Index
- `mfi` - Money Flow Index (volume-weighted RSI)
- `tsi` - True Strength Index
- `ultimate_oscillator` - Ultimate Oscillator
- `awesome_oscillator` - Awesome Oscillator
- `momentum_divergence` - RSI/price divergence signal

---

### **7. Price Action Patterns (Advanced)**
**Why:** More pattern recognition = better entry signals.

- `three_white_soldiers` - Bullish 3-candle pattern
- `three_black_crows` - Bearish 3-candle pattern
- `hanging_man` - Reversal pattern
- `inverted_hammer` - Reversal pattern
- `piercing_pattern` - Bullish reversal
- `dark_cloud_cover` - Bearish reversal
- `harami_bullish` - Bullish reversal
- `harami_bearish` - Bearish reversal
- `engulfing_strength` - Strength of engulfing pattern
- `candle_wicks_ratio` - Upper wick / lower wick ratio
- `consecutive_green_days` - Count of consecutive up days
- `consecutive_red_days` - Count of consecutive down days
- `inside_bar` - Inside bar pattern
- `outside_bar` - Outside bar pattern

---

### **8. Market Regime Indicators**
**Why:** Market context matters. Bull/bear/sideways markets behave differently.

- `market_regime` - Bull/Bear/Sideways classification
- `regime_strength` - Strength of current regime
- `regime_duration` - Days in current regime
- `regime_change_probability` - Likelihood of regime change
- `trending_market_flag` - Binary: strong trend vs choppy
- `choppy_market_flag` - Binary: sideways/choppy market
- `bull_market_flag` - Binary: bullish regime
- `bear_market_flag` - Binary: bearish regime

---

### **9. Relative Strength Features**
**Why:** How stock performs vs market/sector matters for swing trading.

- `relative_strength_spy` - Stock return vs SPY return (if SPY data available)
- `relative_strength_sector` - Stock return vs sector return
- `rs_rank_20d` - Relative strength rank (0-100)
- `rs_rank_50d` - Relative strength rank over 50 days
- `rs_momentum` - Rate of change of relative strength
- `outperformance_flag` - Binary: outperforming market

**Note:** Requires market/sector data. May need to add SPY download or sector mapping.

---

### **10. Time-Based Features**
**Why:** Market behavior varies by day of week, month, etc.

- `day_of_week` - Monday=0, Friday=4 (cyclical encoding)
- `day_of_month` - 1-31 (cyclical encoding)
- `month_of_year` - 1-12 (cyclical encoding)
- `quarter` - Q1-Q4 (cyclical encoding)
- `is_month_end` - Binary: last 3 days of month
- `is_quarter_end` - Binary: last week of quarter
- `is_year_end` - Binary: December
- `days_since_earnings` - If earnings data available

---

## 📊 **PRIORITY 3: Advanced Features** (Implement Third)

### **11. Feature Interactions**
**Why:** ML models benefit from explicit interactions. Can create polynomial features.

- `rsi_x_volume` - RSI × Volume ratio
- `rsi_x_atr` - RSI × ATR pct
- `macd_x_volume` - MACD × Volume ratio
- `trend_x_momentum` - Trend strength × Momentum
- `volatility_x_volume` - Volatility × Volume
- `price_position_x_rsi` - Price percentile × RSI

**Implementation:** Can be done in feature pipeline or let XGBoost handle (it finds interactions automatically).

---

### **12. Statistical Features**
**Why:** Statistical properties reveal market structure.

- `price_skewness_20d` - Price distribution skewness
- `price_kurtosis_20d` - Price distribution kurtosis
- `returns_skewness_20d` - Return distribution skewness
- `returns_kurtosis_20d` - Return distribution kurtosis
- `price_autocorrelation` - Price autocorrelation (trend persistence)
- `volume_autocorrelation` - Volume autocorrelation
- `hurst_exponent` - Market efficiency measure (trending vs mean-reverting)

---

### **13. Advanced Volatility Features**
**Why:** More volatility perspectives = better risk assessment.

- `parkinson_volatility` - High-low based volatility
- `garman_klass_volatility` - OHLC-based volatility
- `rogers_satchell_volatility` - OHLC volatility with drift
- `yang_zhang_volatility` - Overnight + intraday volatility
- `volatility_of_volatility` - Volatility clustering measure
- `realized_volatility` - Historical realized volatility

---

### **14. Market Microstructure Features**
**Why:** Order flow and microstructure matter for short-term moves.

- `bid_ask_spread` - If available (requires L2 data)
- `order_imbalance` - If available
- `liquidity_measure` - Volume/volatility ratio
- `price_impact` - If available

**Note:** Most require Level 2 data. May not be feasible with yfinance.

---

### **15. Sector/Industry Features** (If Data Available)
**Why:** Sector rotation and industry trends matter.

- `sector_momentum` - Sector performance
- `sector_relative_strength` - Stock vs sector
- `industry_group_rank` - Industry group performance rank
- `sector_trend` - Sector trend direction

**Implementation:** Requires sector mapping and sector index data.

---

## 🔧 **Implementation Priority Summary**

### **Phase 1 (Immediate - Highest ROI):**
1. Multi-timeframe features (weekly patterns) - ~15 features
2. Volatility regime features - ~10 features
3. Trend strength features - ~10 features
4. Volume profile features - ~12 features
5. Support/resistance features - ~12 features

**Total Phase 1: ~59 new features**

### **Phase 2 (Next - Medium ROI):**
6. Advanced momentum indicators - ~12 features
7. Advanced price action patterns - ~14 features
8. Market regime indicators - ~8 features
9. Relative strength features - ~6 features (if data available)
10. Time-based features - ~8 features

**Total Phase 2: ~48 new features**

### **Phase 3 (Later - Advanced):**
11. Feature interactions - Can be automated
12. Statistical features - ~7 features
13. Advanced volatility - ~6 features
14. Market microstructure - Limited by data
15. Sector features - Limited by data

**Total Phase 3: ~13-20 features (depending on data availability)**

---

## 📈 **Expected Impact**

- **Current:** ~50 features, AUC: 0.6933
- **After Phase 1:** ~109 features, Expected AUC: 0.72-0.75
- **After Phase 2:** ~157 features, Expected AUC: 0.75-0.78
- **After Phase 3:** ~170-180 features, Expected AUC: 0.78-0.82

**Note:** More features don't always mean better performance. Feature selection and quality matter more than quantity. But having a diverse feature set gives the model more signal to work with.

---

## 🚀 **Quick Wins (Can Implement Today)**

If you want to start immediately, these are the easiest to implement:

1. **Weekly SMA/EMA features** - Simple resampling
2. **Volatility percentile features** - Simple percentile calculations
3. **Support/Resistance levels** - Rolling min/max
4. **Time-based features** - Date extraction
5. **Additional momentum indicators** - pandas-ta has most of these

These 5 categories can add ~30-40 features quickly and should provide immediate improvement.

---

## 📝 **Notes**

- All features must avoid lookahead bias (no future information)
- Features should be validated for infinities, NaNs, and constant values
- Consider feature importance after adding to identify redundant features
- Some features may require additional data sources (sector, market indices)
- Feature interactions can be handled by XGBoost automatically, but explicit features may help

---

## 🎯 **Recommended Starting Point**

**Start with Phase 1, specifically:**
1. Multi-timeframe (weekly) features
2. Volatility regime features  
3. Support/resistance levels

These three categories alone should add ~35-40 features and provide significant improvement with relatively straightforward implementation.

