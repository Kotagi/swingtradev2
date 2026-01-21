# New Dawn Feature Set - Implementation Roadmap

**Feature Set:** v3_New_Dawn  
**Total Features:** ~383 features  
**Status:** Implementation roadmap with ML/Filter categorization

## **PROGRESS SUMMARY**

**Completed Blocks:**
- ✅ **BLOCK ML-1:** Foundation - Price & Returns (30/30 features) - COMPLETE
- ✅ **BLOCK ML-2:** Foundation - Volume & Basic Momentum (30/30 features) - COMPLETE
- ✅ **BLOCK ML-3:** Advanced Momentum Indicators (18/18 features) - COMPLETE
- ✅ **BLOCK ML-4:** Volatility Regime Features (18/18 features) - COMPLETE
- ✅ **BLOCK ML-5:** Trend Strength & Quality (22/22 features) - COMPLETE
- ✅ **BLOCK ML-6:** Support & Resistance (18/18 features) - COMPLETE
- ✅ **BLOCK ML-7:** Volume Profile & VWAP (20/20 features) - COMPLETE
- ✅ **BLOCK ML-8:** Multi-Timeframe Weekly Features (20/20 features) - COMPLETE
- ✅ **BLOCK ML-9:** Multi-Timeframe Monthly Features (5/5 features) - COMPLETE
- ✅ **BLOCK ML-10:** Price Action Patterns - Basic (25/25 features) - COMPLETE
- ✅ **BLOCK ML-11:** Price Action - Consecutive Days & Swings (6/6 features) - COMPLETE
- ✅ **BLOCK ML-12:** Market Regime Indicators (SPY-based) (15/15 features) - COMPLETE
- ✅ **BLOCK ML-13:** Market Direction Features (SPY-based) (12/12 features) - COMPLETE
- ✅ **BLOCK ML-14:** Relative Strength (SPY-based) (12/12 features) - COMPLETE
- ✅ **BLOCK ML-15:** Time-Based Features (14/15 features) - COMPLETE (1 skipped: days_since_earnings - to be done during fundamental step)
- ✅ **BLOCK ML-16:** Statistical Features (15/15 features) - COMPLETE (volume_autocorrelation already existed from earlier block)
- ✅ **BLOCK ML-17:** Advanced Volatility Features (12/12 features) - COMPLETE
- ✅ **BLOCK ML-18:** Advanced Technical Indicators (8/8 features) - COMPLETE
- ✅ **BLOCK ML-19:** Market Microstructure Proxies (8/8 features) - COMPLETE
- ✅ **BLOCK ML-20:** Feature Interactions - Core (20/20 features) - COMPLETE
- ✅ **BLOCK ML-21:** Predictive Gain Features (15/15 features) - COMPLETE
- ✅ **BLOCK ML-22:** Accuracy-Enhancing Features (20/20 features) - COMPLETE
- ✅ **BLOCK ML-23:** Market Context (SPY-based) (3/3 features) - COMPLETE
- ✅ **BLOCK ML-24:** Breakout & Channel Features (4/4 features) - COMPLETE
- ✅ **BLOCK ML-25:** Trend Exhaustion & Consolidation (4/4 features) - COMPLETE

**Total Completed:** 371/383 features (96.9%)

**Current Status:** All ML blocks complete! Ready for Filter features or testing.

---

## **LEGEND**

- **M** = Good for Machine Learning (predictive, normalized, non-redundant)
- **L** = Good for Filtering (interpretable, binary flags, screening criteria)
- **M+L** = Good for both ML and Filtering
- **Risk:** 🔴 High | ⚠️ Medium | ✅ Low
- **Complexity:** 1-5 (1=simple, 5=complex)
- **Time:** Estimated hours per block

---

## **SECTION 1: MACHINE LEARNING FEATURES (Can Calculate)**

*Features optimized for ML prediction - normalized, non-redundant, predictive*

---

### **BLOCK ML-1: Foundation - Price & Returns** ✅ **COMPLETE**
**Features: 30 | Risk: ✅ Low | Complexity: 1-2 | Time: 4-6 hours | Dependencies: None | Status: ✅ 30/30 Complete**

**Group 1.1: Price & Normalization (4 features)**
- [x] `price` - Raw closing price **M**
- [x] `price_log` - Log of closing price (ln(close)) **M**
- [x] `price_vs_ma200` - Price normalized to 200-day MA **M**
- [x] `close_position_in_range` - Close position in daily range (close-low)/(high-low) **M**

**Group 1.2: Returns (8 features)**
- [x] `log_return_1d` - 1-day log return: ln(close_t / close_{t-1}) **M**
- [x] `daily_return` - Daily return % **M**
- [x] `gap_pct` - Gap % (open - prev_close) / prev_close **M**
- [x] `weekly_return_5d` - 5-day return % **M**
- [x] `monthly_return_21d` - 21-day return % **M**
- [x] `quarterly_return_63d` - 63-day return % **M**
- [x] `ytd_return` - Year-to-Date return % **M**
- [x] `weekly_return_1w` - 1-week log return **M**

**Group 1.3: 52-Week Position (3 features)**
- [x] `dist_52w_high` - Distance to 52-week high **M**
- [x] `dist_52w_low` - Distance to 52-week low **M**
- [x] `pos_52w` - 52-week position (0=low, 1=high) **M+L**

**Group 1.4: Basic Moving Averages (8 features)**
- [x] `sma20_ratio` - SMA20 ratio (close/SMA20) **M**
- [x] `sma50_ratio` - SMA50 ratio (close/SMA50) **M**
- [x] `sma200_ratio` - SMA200 ratio (close/SMA200) **M**
- [x] `ema20_ratio` - EMA20 ratio (close/EMA20) **M**
- [x] `ema50_ratio` - EMA50 ratio (close/EMA50) **M**
- [x] `ema200_ratio` - EMA200 ratio (close/EMA200) **M**
- [x] `sma20_sma50_ratio` - SMA20/SMA50 ratio **M**
- [x] `sma50_sma200_ratio` - SMA50/SMA200 ratio **M**

**Group 1.5: MA Slopes (4 features)**
- [x] `sma20_slope` - SMA20 slope **M**
- [x] `sma50_slope` - SMA50 slope **M**
- [x] `sma200_slope` - SMA200 slope **M**
- [x] `ema20_slope` - EMA20 slope **M**

**Group 1.6: Basic Volatility (3 features)**
- [x] `volatility_5d` - 5-day volatility (std of returns) **M**
- [x] `volatility_21d` - 21-day volatility (std of returns) **M**
- [x] `atr14_normalized` - Normalized ATR14 (ATR14/close) **M**

**Testing Checkpoint:** Validate foundation features, establish baseline AUC

---

### **BLOCK ML-2: Foundation - Volume & Basic Momentum** ✅ **COMPLETE**
**Features: 30 | Risk: ✅ Low | Complexity: 1-2 | Time: 4-6 hours | Dependencies: Block ML-1 | Status: ✅ 30/30 Complete**

**Group 2.1: Basic Volume (6 features)**
- [x] `log_volume` - Log volume (log1p(volume)) **M**
- [x] `log_avg_volume_20d` - Log average volume 20-day **M**
- [x] `relative_volume` - Relative volume (volume/avg_volume) **M+L**
- [x] `obv_momentum` - OBV rate of change **M**
- [x] `volume_trend` - Volume moving average slope **M**
- [x] `volume_momentum` - Volume momentum (rate of change) **M**

**Group 2.2: RSI Family (4 features)**
- [x] `rsi7` - RSI7: shorter-term momentum **M**
- [x] `rsi14` - RSI14: standard momentum **M+L**
- [x] `rsi21` - RSI21: longer-term momentum **M**
- [x] `rsi_momentum` - RSI momentum (rate of change) **M**

**Group 2.3: MACD Family (4 features)**
- [x] `macd_line` - MACD line **M**
- [x] `macd_signal` - MACD signal line **M**
- [x] `macd_histogram_normalized` - MACD histogram normalized by price **M**
- [x] `macd_momentum` - MACD momentum (rate of change) **M**

**Group 2.4: ROC Family (4 features)**
- [x] `roc5` - ROC5: very short-term momentum **M**
- [x] `roc10` - ROC10: short-term momentum **M**
- [x] `roc20` - ROC20: medium-term momentum **M**
- [x] `momentum_10d` - Price momentum (close - close[10]) **M**

**Group 2.5: Stochastic Family (3 features)**
- [x] `stochastic_k14` - Stochastic %K **M+L**
- [x] `stochastic_d14` - Stochastic %D (signal line) **M**
- [x] `stochastic_oscillator` - Combined K/D oscillator **M**

**Group 2.6: Other Oscillators (9 features)**
- [x] `cci20` - Commodity Channel Index **M+L**
- [x] `williams_r14` - Williams %R **M+L**
- [x] `ppo_histogram` - PPO histogram **M**
- [x] `dpo` - Detrended Price Oscillator **M**
- [x] `kama_slope` - KAMA slope **M**
- [x] `chaikin_money_flow` - Chaikin Money Flow **M+L**
- [x] `mfi` - Money Flow Index (volume-weighted RSI) **M+L**
- [x] `tsi` - True Strength Index **M**
- [x] `trend_residual` - Trend residual (noise vs trend) **M**

**Testing Checkpoint:** Test momentum features, check for improvements over baseline

---

### **BLOCK ML-3: Advanced Momentum Indicators** ✅ **COMPLETE**
**Features: 18 | Risk: ⚠️ Medium | Complexity: 2-3 | Time: 6-8 hours | Dependencies: Block ML-2 | Status: ✅ 18/18 Complete**

**Group 3.1: Additional Momentum Oscillators (4 features)**
- [x] `ultimate_oscillator` - Ultimate Oscillator (combines 3 timeframes) **M**
- [x] `awesome_oscillator` - Awesome Oscillator (5-period SMA - 34-period SMA) **M**
- [x] `momentum_20d` - Price momentum (close - close[20]) **M**
- [x] `momentum_50d` - Price momentum (close - close[50]) **M**

**Group 3.2: Momentum Quality Features (10 features)**
- [x] `momentum_divergence` - RSI/price divergence signal **M**
- [x] `momentum_consistency` - Consistency of momentum across timeframes **M**
- [x] `momentum_acceleration` - Rate of change of momentum **M**
- [x] `momentum_strength` - Strength of momentum signal **M**
- [x] `momentum_persistence` - Momentum persistence measure **M**
- [x] `momentum_exhaustion` - Momentum exhaustion indicator **M**
- [x] `momentum_reversal_signal` - Momentum reversal signal **M**
- [x] `momentum_cross` - Momentum crossover signal **M**
- [x] `momentum_rank` - Momentum rank vs historical **M**
- [x] `momentum_regime` - Momentum regime (strong/weak/neutral) **M**

**Group 3.3: Additional Momentum Variants (4 features)**
- [x] `momentum_5d` - 5-day price momentum **M**
- [x] `momentum_15d` - 15-day price momentum **M**
- [x] `momentum_30d` - 30-day price momentum **M**
- [x] `momentum_vs_price` - Momentum vs price divergence **M**

**Testing Checkpoint:** Validate advanced momentum features, check for overfitting

---

### **BLOCK ML-4: Volatility Regime Features** ✅ **COMPLETE**
**Features: 18 | Risk: ⚠️ Medium | Complexity: 2-3 | Time: 6-8 hours | Dependencies: Block ML-1 | Status: ✅ 18/18 Complete**

**Group 4.1: Volatility Regime Detection (10 features)**
- [x] `volatility_regime` - Current ATR percentile (0-100) over 252 days **M**
- [x] `volatility_trend` - ATR slope (increasing/decreasing volatility) **M**
- [x] `bb_squeeze` - Bollinger Band squeeze indicator (low volatility) **M+L**
- [x] `bb_expansion` - Bollinger Band expansion (high volatility) **M+L**
- [x] `atr_ratio_20d` - Current ATR / 20-day ATR average **M**
- [x] `atr_ratio_252d` - Current ATR / 252-day ATR average **M**
- [x] `volatility_percentile_20d` - ATR percentile over 20 days **M**
- [x] `volatility_percentile_252d` - ATR percentile over 252 days **M**
- [x] `high_volatility_flag` - Binary: ATR > 75th percentile **M+L**
- [x] `low_volatility_flag` - Binary: ATR < 25th percentile **M+L**

**Group 4.2: Advanced Volatility Estimators (5 features)**
- [x] `parkinson_volatility` - High-low based volatility estimator **M**
- [x] `garman_klass_volatility` - OHLC-based volatility estimator **M**
- [x] `rogers_satchell_volatility` - OHLC volatility with drift correction **M**
- [x] `yang_zhang_volatility` - Overnight + intraday volatility **M**
- [x] `volatility_clustering` - Volatility clustering measure **M**

**Group 4.3: Realized Volatility (3 features)**
- [x] `realized_volatility_5d` - 5-day realized volatility **M**
- [x] `realized_volatility_20d` - 20-day realized volatility **M**
- [x] `volatility_regime_change` - Binary: volatility regime changed recently **M+L**

**Testing Checkpoint:** Test volatility regime features, validate regime detection

---

### **BLOCK ML-5: Trend Strength & Quality** ✅ **COMPLETE**
**Features: 22 | Risk: ⚠️ Medium | Complexity: 2-3 | Time: 8-10 hours | Dependencies: Block ML-1, ML-2 | Status: ✅ 22/22 Complete**

**Group 5.1: ADX & Trend Strength (5 features)**
- [x] `adx14` - ADX14 (trend strength) **M+L**
- [x] `adx20` - ADX20: medium-term trend strength **M**
- [x] `adx50` - ADX50: long-term trend strength **M**
- [x] `trend_strength_20d` - ADX over 20 days **M**
- [x] `trend_strength_50d` - ADX over 50 days **M**

**Group 5.2: MA Alignment & Crossovers (6 features)**
- [x] `ema_alignment` - All EMAs aligned (bullish/bearish/neutral): -1/0/1 **M+L**
- [x] `ma_crossover_bullish` - Binary: SMA20 > SMA50 > SMA200 **M+L**
- [x] `ma_crossover_bearish` - Binary: SMA20 < SMA50 < SMA200 **M+L**
- [x] `price_vs_all_mas` - Count of MAs price is above **M+L**
- [x] `sma_slope_20d` - Slope of 20-day SMA **M**
- [x] `sma_slope_50d` - Slope of 50-day SMA **M**

**Group 5.3: EMA Slopes (2 features)**
- [x] `ema_slope_20d` - Slope of 20-day EMA **M**
- [x] `ema_slope_50d` - Slope of 50-day EMA **M**

**Group 5.4: Trend Quality Features (9 features)**
- [x] `trend_consistency` - % of days price above/below MA over lookback **M**
- [x] `trend_duration` - Days since last trend change **M**
- [x] `trend_reversal_signal` - Potential trend reversal indicator **M**
- [x] `trend_acceleration` - Rate of change of trend slope **M**
- [x] `trend_divergence` - Price vs trend divergence **M**
- [x] `trend_pullback_strength` - Strength of pullback to trend **M**
- [x] `trend_breakout_strength` - Strength of trend breakout **M**
- [x] `trend_following_strength` - How well price follows trend **M**
- [x] `trend_regime` - Trend regime classification **M**

**Testing Checkpoint:** Validate trend features, check for improvements

---

### **BLOCK ML-6: Support & Resistance** ✅ **COMPLETE**
**Features: 18 | Risk: 🔴 High | Complexity: 3-4 | Time: 10-12 hours | Dependencies: Block ML-1 | Status: ✅ 18/18 Complete**

**Group 6.1: Support/Resistance Levels (6 features)**
- [x] `resistance_level_20d` - Nearest resistance (20-day high) **M+L**
- [x] `resistance_level_50d` - Nearest resistance (50-day high) **M+L**
- [x] `resistance_level_100d` - Nearest resistance (100-day high) **M+L**
- [x] `support_level_20d` - Nearest support (20-day low) **M+L**
- [x] `support_level_50d` - Nearest support (50-day low) **M+L**
- [x] `support_level_100d` - Nearest support (100-day low) **M+L**

**Group 6.2: Distance to S/R (4 features)**
- [x] `distance_to_resistance` - % distance to nearest resistance **M**
- [x] `distance_to_support` - % distance to nearest support **M**
- [x] `price_near_resistance` - Binary: within 2% of resistance **M+L**
- [x] `price_near_support` - Binary: within 2% of support **M+L**

**Group 6.3: S/R Strength (4 features)**
- [x] `resistance_touches` - Number of times price touched resistance **M**
- [x] `support_touches` - Number of times price touched support **M**
- [x] `support_resistance_strength` - Combined support/resistance strength score **M**
- [x] `donchian_position` - Donchian channel position **M+L**

**Group 6.4: Pivot & Fibonacci (4 features)**
- [x] `pivot_point` - Classic pivot point calculation (H+L+C)/3 **M+L**
- [x] `pivot_resistance_1` - Pivot resistance level 1 **M+L**
- [x] `pivot_support_1` - Pivot support level 1 **M+L**
- [x] `fibonacci_retracement` - Current Fibonacci retracement level **M+L**

**Testing Checkpoint:** **CRITICAL** - Validate no lookahead bias in S/R calculations

---

### **BLOCK ML-7: Volume Profile & VWAP** ✅ **COMPLETE**
**Features: 20 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 10-12 hours | Dependencies: Block ML-2 | Status: ✅ 20/20 Complete**

**Group 7.1: Volume Profile Approximations (6 features)**
- [x] `volume_profile_poc` - Point of Control (daily approximation) **M**
- [x] `volume_profile_vah` - Value Area High (approximation) **M**
- [x] `volume_profile_val` - Value Area Low (approximation) **M**
- [x] `price_vs_poc` - Distance from current price to POC **M**
- [x] `price_vs_vah` - Distance from price to Value Area High **M**
- [x] `price_vs_val` - Distance from price to Value Area Low **M**

**Group 7.2: VWAP Features (5 features)**
- [x] `volume_weighted_price` - VWAP (Volume Weighted Average Price) **M+L**
- [x] `price_vs_vwap` - Distance from price to VWAP **M**
- [x] `vwap_slope` - VWAP trend direction **M**
- [x] `vwap_distance_pct` - % distance from VWAP **M**
- [x] `volume_profile_width` - Width of volume profile **M**

**Group 7.3: Volume Analysis (9 features)**
- [x] `volume_distribution` - Volume concentration metric **M**
- [x] `volume_climax` - Unusually high volume days (volume > 2x average) **M+L**
- [x] `volume_dry_up` - Unusually low volume days (volume < 0.5x average) **M+L**
- [x] `volume_breakout` - Volume spike on price breakout **M+L**
- [x] `volume_divergence` - Volume vs price divergence **M**
- [x] `volume_autocorrelation` - Volume autocorrelation **M**
- [x] `volume_imbalance` - Buy vs sell volume imbalance proxy **M**
- [x] `volume_at_price` - Volume at current price level **M**
- [x] `volatility_ratio` - Volatility ratio: vol5/vol21 **M**

**Testing Checkpoint:** Validate volume profile approximations, check VWAP calculations

---

### **BLOCK ML-8: Multi-Timeframe Weekly Features** ✅ **COMPLETE**
**Features: 20 | Risk: 🔴 High | Complexity: 4-5 | Time: 12-15 hours | Dependencies: Block ML-1, ML-2 | Status: ✅ 20/20 Complete**

**Group 8.1: Weekly Returns (3 features)**
- [x] `weekly_return_2w` - 2-week log return **M**
- [x] `weekly_return_4w` - 4-week log return **M**
- [x] `weekly_volume_ratio` - Weekly volume vs 20-week average **M**

**Group 8.2: Weekly Moving Averages (6 features)**
- [x] `weekly_sma_5w` - 5-week SMA **M**
- [x] `weekly_sma_10w` - 10-week SMA **M**
- [x] `weekly_sma_20w` - 20-week SMA **M**
- [x] `weekly_ema_5w` - 5-week EMA **M**
- [x] `weekly_ema_10w` - 10-week EMA **M**
- [x] `weekly_ema_20w` - 20-week EMA **M**

**Group 8.3: Weekly Indicators (6 features)**
- [x] `weekly_rsi_14w` - Weekly RSI (14-week) **M**
- [x] `weekly_rsi_7w` - Weekly RSI (7-week) **M**
- [x] `weekly_macd_histogram` - Weekly MACD histogram **M**
- [x] `weekly_macd_signal` - Weekly MACD signal line **M**
- [x] `weekly_adx` - Weekly ADX (14-week) **M**
- [x] `weekly_stochastic` - Weekly Stochastic (14-week) **M**

**Group 8.4: Weekly Price Comparisons (5 features)**
- [x] `close_vs_weekly_sma20` - Price vs 20-week SMA ratio **M**
- [x] `close_vs_weekly_ema20` - Price vs 20-week EMA ratio **M**
- [x] `weekly_atr_pct` - Weekly ATR as % of price **M**
- [x] `weekly_trend_strength` - Slope of weekly SMA20 **M**
- [x] `weekly_momentum` - Weekly momentum **M**

**Testing Checkpoint:** **CRITICAL** - Validate weekly resampling and shift operations (no lookahead bias)

---

### **BLOCK ML-9: Multi-Timeframe Monthly Features** ✅ **COMPLETE**
**Features: 5 | Risk: 🔴 High | Complexity: 4-5 | Time: 6-8 hours | Dependencies: Block ML-8 | Status: ✅ 5/5 Complete**

**Group 9.1: Monthly Features (5 features)**
- [x] `monthly_return_1m` - 1-month log return **M**
- [x] `monthly_return_3m` - 3-month log return **M**
- [x] `monthly_sma_3m` - 3-month SMA **M**
- [x] `monthly_sma_6m` - 6-month SMA **M**
- [x] `monthly_rsi` - Monthly RSI **M**

**Testing Checkpoint:** Validate monthly resampling, ensure proper shifting

---

### **BLOCK ML-10: Price Action Patterns - Basic** ✅ **COMPLETE**
**Features: 25 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 12-15 hours | Dependencies: Block ML-1 | Status: ✅ 25/25 Complete**

**Group 10.1: Candlestick Components (4 features)**
- [x] `candle_body_pct` - Candle body % (body/range) in [0, 1] **M+L**
- [x] `candle_upper_wick_pct` - Upper wick % (upper/range) in [0, 1] **M+L**
- [x] `candle_lower_wick_pct` - Lower wick % (lower/range) in [0, 1] **M+L**
- [x] `candle_wicks_ratio` - Upper wick / lower wick ratio **M+L**

**Group 10.2: Reversal Patterns (8 features)**
- [x] `hanging_man` - Reversal pattern (bearish) **M+L**
- [x] `inverted_hammer` - Reversal pattern (bullish) **M+L**
- [x] `piercing_pattern` - Bullish reversal (2-candle) **M+L**
- [x] `dark_cloud_cover` - Bearish reversal (2-candle) **M+L**
- [x] `harami_bullish` - Bullish reversal (2-candle) **M+L**
- [x] `harami_bearish` - Bearish reversal (2-candle) **M+L**
- [x] `engulfing_bullish` - Bullish engulfing pattern **M+L**
- [x] `engulfing_bearish` - Bearish engulfing pattern **M+L**

**Group 10.3: Continuation Patterns (5 features)**
- [x] `three_white_soldiers` - Bullish 3-candle pattern **M+L**
- [x] `three_black_crows` - Bearish 3-candle pattern **M+L**
- [x] `inside_bar` - Inside bar pattern **M+L**
- [x] `outside_bar` - Outside bar pattern **M+L**
- [x] `engulfing_strength` - Strength of engulfing pattern **M**

**Group 10.4: Single Candle Patterns (5 features)**
- [x] `doji` - Doji pattern **M+L**
- [x] `hammer` - Hammer pattern **M+L**
- [x] `shooting_star` - Shooting star pattern **M+L**
- [x] `marubozu` - Marubozu pattern (no wicks) **M+L**
- [x] `spinning_top` - Spinning top pattern **M+L**

**Group 10.5: Pattern Quality (3 features)**
- [x] `pattern_strength` - Overall pattern strength score **M**
- [x] `pattern_confirmation` - Pattern confirmation signal **M**
- [x] `pattern_divergence` - Pattern vs price divergence **M**

**Testing Checkpoint:** Validate pattern detection logic, test on sample data

---

### **BLOCK ML-11: Price Action - Consecutive Days & Swings** ✅ **COMPLETE**
**Features: 6 | Risk: ✅ Low | Complexity: 2 | Time: 4-6 hours | Dependencies: Block ML-1 | Status: ✅ 6/6 Complete**

**Group 11.1: Consecutive Days (2 features)**
- [x] `consecutive_green_days` - Count of consecutive up days **M+L**
- [x] `consecutive_red_days` - Count of consecutive down days **M+L**

**Group 11.2: Price Action (4 features)**
- [x] `higher_high_10d` - Higher high (10-day) binary flag **M+L**
- [x] `higher_low_10d` - Higher low (10-day) binary flag **M+L**
- [x] `swing_low_10d` - Recent swing low (10-day) **M+L**
- [x] `pattern_cluster` - Multiple patterns occurring **M**

**Testing Checkpoint:** Quick validation of price action features

---

### **BLOCK ML-12: Market Regime Indicators (SPY-based)** ✅ **COMPLETE**
**Features: 15 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 10-12 hours | Dependencies: Block ML-1, SPY data | Status: ✅ 15/15 Complete**

**Group 12.1: Market Regime Classification (8 features)**
- [x] `market_regime` - Bull/Bear/Sideways classification (0/1/2) **M+L**
- [x] `regime_strength` - Strength of current regime (0-1) **M**
- [x] `regime_duration` - Days in current regime **M**
- [x] `regime_change_probability` - Likelihood of regime change (0-1) **M**
- [x] `trending_market_flag` - Binary: strong trend vs choppy **M+L**
- [x] `choppy_market_flag` - Binary: sideways/choppy market **M+L**
- [x] `bull_market_flag` - Binary: bullish regime **M+L**
- [x] `bear_market_flag` - Binary: bearish regime **M+L**

**Group 12.2: Market Context (7 features)**
- [x] `market_volatility_regime` - Market volatility regime (low/normal/high) **M**
- [x] `market_momentum_regime` - Market momentum regime (strong/weak/neutral) **M**
- [x] `regime_transition` - Binary: regime transition occurring **M+L**
- [x] `regime_stability` - Regime stability measure **M**
- [x] `market_sentiment` - Market sentiment indicator **M**
- [x] `market_fear_greed` - Market fear/greed index proxy **M**
- [x] `regime_consistency` - Consistency of regime signals **M**

**Testing Checkpoint:** Validate SPY data alignment, test regime classification

---

### **BLOCK ML-13: Market Direction Features (SPY-based)** ✅ **COMPLETE**
**Features: 12 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 8-10 hours | Dependencies: Block ML-12 | Status: ✅ 12/12 Complete**

**Group 13.1: Market Direction (6 features)**
- [x] `market_direction` - Market direction (up/down/sideways) **M+L**
- [x] `market_direction_strength` - Strength of market direction **M**
- [x] `market_trend_alignment` - Stock trend vs market trend alignment **M**
- [x] `market_momentum` - Market momentum indicator **M**
- [x] `market_volatility_context` - Market volatility context **M**
- [x] `market_timing` - Market timing indicator **M**

**Group 13.2: Market Structure (6 features)**
- [x] `market_support_resistance` - Market support/resistance levels **M**
- [x] `market_regime_consistency` - Market regime consistency **M**
- [x] `market_sector_rotation` - Market sector rotation signal **M**
- [x] `market_breadth` - Market breadth indicator (advancing/declining) **M**
- [x] `market_leadership` - Market leadership indicator **M**
- [x] `market_sentiment_alignment` - Stock sentiment vs market sentiment **M**

**Testing Checkpoint:** Validate market direction calculations

---

### **BLOCK ML-14: Relative Strength (SPY-based)** ✅ **COMPLETE**
**Features: 12 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 8-10 hours | Dependencies: Block ML-1, SPY data | Status: ✅ 12/12 Complete**

**Group 14.1: Relative Strength vs SPY (5 features)**
- [x] `relative_strength_spy` - Stock return vs SPY return (20-day) **M**
- [x] `relative_strength_spy_50d` - Stock return vs SPY return (50-day) **M**
- [x] `rs_rank_20d` - Relative strength rank (0-100) over 20 days **M**
- [x] `rs_rank_50d` - Relative strength rank (0-100) over 50 days **M**
- [x] `rs_rank_100d` - Relative strength rank (0-100) over 100 days **M**

**Group 14.2: RS Quality Features (7 features)**
- [x] `rs_momentum` - Rate of change of relative strength **M**
- [x] `outperformance_flag` - Binary: outperforming market **M+L**
- [x] `rs_vs_price` - Relative strength vs price divergence **M**
- [x] `rs_consistency` - Consistency of relative strength **M**
- [x] `rs_regime` - Relative strength regime (strong/weak/neutral) **M**
- [x] `rs_trend` - Relative strength trend direction **M**
- [x] `relative_strength_sector` - Stock vs sector return (if sector data available) **M**

**Testing Checkpoint:** **CRITICAL** - Validate SPY data alignment by date, no future data

---

### **BLOCK ML-15: Time-Based Features** ✅ **COMPLETE** (14/15 features)
**Features: 15 | Risk: ✅ Low | Complexity: 2 | Time: 6-8 hours | Dependencies: None | Status: ✅ 14/15 Complete (1 skipped)**

**Group 15.1: Cyclical Time Features (10 features)**
- [x] `day_of_week` - Day of week (cyclical encoding: sin/cos) **M**
- [x] `day_of_week_sin` - Day of week (sine encoding) **M**
- [x] `day_of_week_cos` - Day of week (cosine encoding) **M**
- [x] `day_of_month` - Day of month (cyclical encoding: sin/cos) **M**
- [x] `day_of_month_sin` - Day of month (sine encoding) **M**
- [x] `day_of_month_cos` - Day of month (cosine encoding) **M**
- [x] `month_of_year` - Month of year (cyclical encoding: sin/cos) **M**
- [x] `month_of_year_sin` - Month of year (sine encoding) **M**
- [x] `month_of_year_cos` - Month of year (cosine encoding) **M**
- [x] `quarter` - Quarter (cyclical encoding) **M**

**Group 15.2: Calendar Flags (4 features)**
- [x] `is_month_end` - Binary: last 3 days of month **M+L**
- [x] `is_quarter_end` - Binary: last week of quarter **M+L**
- [x] `is_year_end` - Binary: December **M+L**
- [x] `trading_day_of_year` - Trading day of year (1-252) **M**

**Group 15.3: Earnings (1 feature)**
- [ ] `days_since_earnings` - Days since last earnings (if available) **M+L** ⚠️ **SKIPPED - To be implemented during fundamental features step**

**Testing Checkpoint:** Validate cyclical encoding, test calendar effects

---

### **BLOCK ML-16: Statistical Features** ✅ **COMPLETE**
**Features: 15 | Risk: ⚠️ Medium | Complexity: 4 | Time: 12-15 hours | Dependencies: Block ML-1 | Status: ✅ 15/15 Complete**

**Group 16.1: Distribution Properties (4 features)**
- [x] `price_skewness_20d` - Price distribution skewness (20-day) **M**
- [x] `price_kurtosis_20d` - Price distribution kurtosis (20-day) **M**
- [x] `returns_skewness_20d` - Return distribution skewness (20-day) **M**
- [x] `returns_kurtosis_20d` - Return distribution kurtosis (20-day) **M**

**Group 16.2: Autocorrelation (4 features)**
- [x] `price_autocorrelation_1d` - Price autocorrelation (1-day lag) **M**
- [x] `price_autocorrelation_5d` - Price autocorrelation (5-day lag) **M**
- [x] `returns_autocorrelation` - Return autocorrelation **M**
- [x] `volume_autocorrelation` - Volume autocorrelation **M** (already existed from earlier block)

**Group 16.3: Market Efficiency (7 features)**
- [x] `price_variance_ratio` - Variance ratio test statistic **M**
- [x] `price_half_life` - Price mean-reversion half-life **M**
- [x] `returns_half_life` - Return mean-reversion half-life **M**
- [x] `price_stationarity` - Price stationarity test (ADF) **M**
- [x] `returns_stationarity` - Return stationarity test (ADF) **M**
- [x] `price_cointegration` - Price cointegration with market (if SPY available) **M**
- [x] `statistical_regime` - Statistical regime (trending/mean-reverting/random) **M**

**Testing Checkpoint:** Validate statistical calculations, check for computational efficiency

---

### **BLOCK ML-17: Advanced Volatility Features** ✅ **COMPLETE**
**Features: 12 | Risk: ⚠️ Medium | Complexity: 4-5 | Time: 12-15 hours | Dependencies: Block ML-4 | Status: ✅ 12/12 Complete**

**Group 17.1: Volatility Properties (4 features)**
- [x] `volatility_skewness` - Volatility distribution skewness **M**
- [x] `volatility_kurtosis` - Volatility distribution kurtosis **M**
- [x] `volatility_autocorrelation` - Volatility autocorrelation **M**
- [x] `volatility_mean_reversion` - Volatility mean-reversion speed **M**

**Group 17.2: Advanced Volatility (8 features)**
- [x] `volatility_of_volatility` - VoV (improved) **M**
- [x] `volatility_regime_persistence` - Volatility regime persistence **M**
- [x] `volatility_shock` - Recent volatility shock indicator **M**
- [x] `volatility_normalized_returns` - Returns normalized by volatility **M**
- [x] `volatility_forecast` - Volatility forecast (GARCH-like) **M**
- [x] `volatility_term_structure` - Volatility term structure (short vs long) **M**
- [x] `volatility_risk_premium` - Volatility risk premium proxy **M**
- [x] `volatility_smoothness` - Volatility smoothness measure **M**

**Testing Checkpoint:** Validate advanced volatility calculations

---

### **BLOCK ML-18: Advanced Technical Indicators** ✅ **COMPLETE**
**Features: 8 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 8-10 hours | Dependencies: Block ML-1, ML-2 | Status: ✅ 8/8 Complete**

**Group 18.1: Advanced Indicators (8 features)**
- [x] `bollinger_band_width` - Bollinger Band Width (log normalized) **M+L**
- [x] `fractal_dimension_index` - Fractal Dimension Index normalized to [0, 1] **M**
- [x] `hurst_exponent` - Hurst Exponent in [0, 1] **M**
- [x] `price_curvature` - Price Curvature (improved) **M**
- [x] `aroon_up` - Aroon Up in [0, 1] **M+L**
- [x] `aroon_down` - Aroon Down in [0, 1] **M+L**
- [x] `aroon_oscillator` - Aroon Oscillator normalized to [0, 1] **M+L**
- [x] `donchian_breakout` - Donchian breakout binary flag **M+L**

**Testing Checkpoint:** Validate advanced indicator calculations

---

### **BLOCK ML-19: Market Microstructure Proxies** ✅ **COMPLETE**
**Features: 8 | Risk: ⚠️ Medium | Complexity: 3 | Time: 6-8 hours | Dependencies: Block ML-1, ML-2 | Status: ✅ 8/8 Complete**

**Group 19.1: Microstructure Proxies (8 features)**
- [x] `liquidity_measure` - Volume/volatility ratio **M**
- [x] `price_impact_proxy` - Price impact proxy (volume/price change) **M**
- [x] `bid_ask_spread_proxy` - Bid-ask spread proxy (high-low/close) **M**
- [x] `order_flow_imbalance` - Order flow imbalance proxy **M**
- [x] `market_depth_proxy` - Market depth proxy **M**
- [x] `price_efficiency` - Price efficiency measure **M**
- [x] `transaction_cost_proxy` - Transaction cost proxy **M**
- [x] `market_quality` - Overall market quality measure **M**

**Testing Checkpoint:** Validate microstructure proxy calculations

---

### **BLOCK ML-20: Feature Interactions - Core** ✅ **COMPLETE**
**Features: 20 | Risk: ✅ Low | Complexity: 2 | Time: 6-8 hours | Dependencies: Blocks ML-1 through ML-19 | Status: ✅ 20/20 Complete**

**Group 20.1: Momentum Interactions (6 features)**
- [x] `rsi_x_volume` - RSI × Volume ratio **M**
- [x] `rsi_x_atr` - RSI × ATR pct **M**
- [x] `macd_x_volume` - MACD × Volume ratio **M**
- [x] `trend_x_momentum` - Trend strength × Momentum **M**
- [x] `volume_x_momentum` - Volume × Momentum **M**
- [x] `rsi_x_trend_strength` - RSI × Trend strength **M**

**Group 20.2: Volatility Interactions (4 features)**
- [x] `volatility_x_volume` - Volatility × Volume **M**
- [x] `volatility_x_trend` - Volatility × Trend strength **M**
- [x] `trend_x_volatility` - Trend strength × Volatility **M**
- [x] `volatility_x_regime_strength` - Volatility × Regime strength **M**

**Group 20.3: Position Interactions (4 features)**
- [x] `price_position_x_rsi` - Price percentile × RSI **M**
- [x] `support_resistance_x_momentum` - Support/resistance distance × Momentum **M**
- [x] `rsi_x_support_distance` - RSI × Distance to support **M**
- [x] `volume_x_breakout` - Volume × Breakout signal **M**

**Group 20.4: Regime Interactions (4 features)**
- [x] `momentum_x_regime` - Momentum × Market regime **M**
- [x] `volume_x_regime` - Volume × Market regime **M**
- [x] `rsi_x_relative_strength` - RSI × Relative strength **M**
- [x] `trend_x_relative_strength` - Trend × Relative strength **M**

**Group 20.5: Time Interactions (2 features)**
- [x] `momentum_x_time` - Momentum × Time feature **M**
- [x] `volume_x_time` - Volume × Time feature **M**

**Testing Checkpoint:** Test interaction features, check for improvements

---

### **BLOCK ML-21: Predictive Gain Features** ✅ **COMPLETE**
**Features: 15 | Risk: ⚠️ Medium | Complexity: 4-5 | Time: 15-20 hours | Dependencies: Blocks ML-1 through ML-20 | Status: ✅ 15/15 Complete**

**Group 21.1: Gain Probability (5 features)**
- [x] `historical_gain_probability` - Historical probability of X% gain in Y days **M**
- [x] `gain_probability_score` - Combined score for gain probability **M**
- [x] `gain_regime` - Gain regime (high/low probability) **M**
- [x] `gain_consistency` - Consistency of gain achievement **M**
- [x] `gain_timing` - Optimal timing for gain **M**

**Group 21.2: Gain Momentum (5 features)**
- [x] `gain_momentum` - Momentum toward target gain **M**
- [x] `gain_acceleration` - Acceleration toward target gain **M**
- [x] `target_distance` - Distance to target gain % **M**
- [x] `gain_momentum_strength` - Strength of gain momentum **M**
- [x] `gain_breakout_signal` - Breakout signal for gain **M**

**Group 21.3: Gain Context (5 features)**
- [x] `gain_risk_ratio` - Gain potential vs risk ratio **M**
- [x] `gain_volume_confirmation` - Volume confirmation for gain **M**
- [x] `gain_trend_alignment` - Trend alignment for gain **M**
- [x] `gain_volatility_context` - Volatility context for gain **M**
- [x] `gain_support_level` - Support level for gain **M**

**Testing Checkpoint:** Validate gain prediction features, test on historical data

---

### **BLOCK ML-22: Accuracy-Enhancing Features** ✅ **COMPLETE**
**Features: 20 | Risk: ⚠️ Medium | Complexity: 4-5 | Time: 15-20 hours | Dependencies: Blocks ML-1 through ML-21 | Status: ✅ 20/20 Complete**

**Group 22.1: Signal Quality (10 features)**
- [x] `signal_quality_score` - Overall signal quality score **M**
- [x] `signal_strength` - Overall signal strength **M**
- [x] `signal_consistency` - Signal consistency across indicators **M**
- [x] `signal_confirmation` - Signal confirmation strength **M**
- [x] `signal_timing` - Optimal signal timing **M**
- [x] `signal_risk_reward` - Signal risk-reward ratio **M**
- [x] `signal_statistical_significance` - Statistical significance of signal **M**
- [x] `signal_historical_success` - Historical success rate of similar signals **M**
- [x] `signal_ensemble_score` - Ensemble score from multiple models **M**
- [x] `false_positive_risk` - False positive risk indicator **M**

**Group 22.2: Signal Context (10 features)**
- [x] `signal_divergence` - Signal divergence indicator **M**
- [x] `signal_volume_confirmation` - Volume confirmation for signal **M**
- [x] `signal_trend_alignment` - Trend alignment for signal **M**
- [x] `signal_volatility_context` - Volatility context for signal **M**
- [x] `signal_regime_alignment` - Regime alignment for signal **M**
- [x] `signal_momentum_strength` - Momentum strength for signal **M**
- [x] `signal_support_resistance` - Support/resistance context for signal **M**
- [x] `signal_relative_strength` - Relative strength for signal **M**
- [x] `signal_multi_timeframe` - Multi-timeframe confirmation **M**
- [x] `signal_pattern_confirmation` - Pattern confirmation for signal **M**

**Testing Checkpoint:** Final validation of accuracy-enhancing features

---

### **BLOCK ML-23: Market Context (SPY-based)** ✅ **COMPLETE**
**Features: 3 | Risk: ⚠️ Medium | Complexity: 3 | Time: 4-6 hours | Dependencies: SPY data | Status: ✅ 3/3 Complete**

**Group 23.1: Market Context (3 features)**
- [x] `mkt_spy_dist_sma200` - SPY Distance from SMA200 (z-score normalized) **M**
- [x] `mkt_spy_sma200_slope` - SPY SMA200 Slope (percentile rank normalized) **M**
- [x] `beta_spy_252d` - Rolling beta vs SPY (252-day) normalized to [0, 1] **M**

**Testing Checkpoint:** Validate SPY data alignment

---

### **BLOCK ML-24: Breakout & Channel Features** ✅ **COMPLETE**
**Features: 4 | Risk: 🔴 High | Complexity: 3 | Time: 6-8 hours | Dependencies: Block ML-1 | Status: ✅ 4/4 Complete**

**Group 24.1: Breakout Features (4 features)**
- [x] `donchian_position` - Donchian position in [0, 1] **M+L** (already existed from ML-6.3)
- [x] `donchian_breakout` - Donchian breakout binary flag **M+L** (already existed from ML-18)
- [x] `ttm_squeeze_on` - TTM Squeeze binary flag **M+L**
- [x] `ttm_squeeze_momentum` - TTM Squeeze momentum **M**

**Testing Checkpoint:** **CRITICAL** - Validate no lookahead bias in breakout detection

---

### **BLOCK ML-25: Trend Exhaustion & Consolidation** ✅ **COMPLETE**
**Features: 4 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 6-8 hours | Dependencies: Block ML-5 | Status: ✅ 4/4 Complete**

**Group 25.1: Trend Quality (4 features)**
- [x] `trend_consolidation` - Binary: price consolidating within trend **M+L**
- [x] `trend_exhaustion` - Trend exhaustion indicator **M**
- [x] `trend_vs_mean_reversion` - Trend vs mean-reversion signal **M**
- [x] `volatility_jump` - Volatility jump detection **M**

**Testing Checkpoint:** Validate trend exhaustion detection

---

**ML FEATURES SUMMARY:**
- **Total ML Blocks:** 25 blocks
- **Total ML Features:** ~330 features
- **Estimated Total Time:** ~250-300 hours
- **Testing Checkpoints:** Every block

---

## **SECTION 2: FILTER FEATURES (Can Calculate)**

*Features optimized for filtering/screening - interpretable, binary flags, screening criteria*

---

### **BLOCK FILTER-1: Binary Flags & Screening Criteria**
**Features: 30 | Risk: ✅ Low | Complexity: 1-2 | Time: 4-6 hours | Dependencies: None**

**Group F1.1: Volatility Flags (2 features)**
- [ ] `high_volatility_flag` - Binary: ATR > 75th percentile **L**
- [ ] `low_volatility_flag` - Binary: ATR < 25th percentile **L**

**Group F1.2: Market Regime Flags (4 features)**
- [ ] `trending_market_flag` - Binary: strong trend vs choppy **L**
- [ ] `choppy_market_flag` - Binary: sideways/choppy market **L**
- [ ] `bull_market_flag` - Binary: bullish regime **L**
- [ ] `bear_market_flag` - Binary: bearish regime **L**

**Group F1.3: Volume Flags (2 features)**
- [ ] `volume_climax` - Unusually high volume days (volume > 2x average) **L**
- [ ] `volume_dry_up` - Unusually low volume days (volume < 0.5x average) **L**

**Group F1.4: Breakout Flags (2 features)**
- [ ] `donchian_breakout` - Donchian breakout binary flag **L**
- [ ] `ttm_squeeze_on` - TTM Squeeze binary flag **L**

**Group F1.5: Support/Resistance Flags (4 features)**
- [ ] `price_near_resistance` - Binary: within 2% of resistance **L**
- [ ] `price_near_support` - Binary: within 2% of support **L**
- [ ] `ma_crossover_bullish` - Binary: SMA20 > SMA50 > SMA200 **L**
- [ ] `ma_crossover_bearish` - Binary: SMA20 < SMA50 < SMA200 **L**

**Group F1.6: Price Action Flags (4 features)**
- [ ] `higher_high_10d` - Higher high (10-day) binary flag **L**
- [ ] `higher_low_10d` - Higher low (10-day) binary flag **L**
- [ ] `regime_transition` - Binary: regime transition occurring **L**
- [ ] `volatility_regime_change` - Binary: volatility regime changed recently **L**

**Group F1.7: Pattern Flags (12 features)**
- [ ] `three_white_soldiers` - Bullish 3-candle pattern **L**
- [ ] `three_black_crows` - Bearish 3-candle pattern **L**
- [ ] `hanging_man` - Reversal pattern (bearish) **L**
- [ ] `inverted_hammer` - Reversal pattern (bullish) **L**
- [ ] `piercing_pattern` - Bullish reversal (2-candle) **L**
- [ ] `dark_cloud_cover` - Bearish reversal (2-candle) **L**
- [ ] `harami_bullish` - Bullish reversal (2-candle) **L**
- [ ] `harami_bearish` - Bearish reversal (2-candle) **L**
- [ ] `engulfing_bullish` - Bullish engulfing pattern **L**
- [ ] `engulfing_bearish` - Bearish engulfing pattern **L**
- [ ] `inside_bar` - Inside bar pattern **L**
- [ ] `outside_bar` - Outside bar pattern **L**

**Testing Checkpoint:** Validate binary flags work correctly

---

### **BLOCK FILTER-2: Interpretable Indicators for Screening**
**Features: 30 | Risk: ✅ Low | Complexity: 1-2 | Time: 4-6 hours | Dependencies: Block ML-1, ML-2**

**Group F2.1: Oscillator Levels (8 features)**
- [ ] `rsi14` - RSI14 (for filtering thresholds) **L**
- [ ] `stochastic_k14` - Stochastic %K (for filtering) **L**
- [ ] `cci20` - Commodity Channel Index **L**
- [ ] `williams_r14` - Williams %R **L**
- [ ] `mfi` - Money Flow Index **L**
- [ ] `chaikin_money_flow` - Chaikin Money Flow **L**
- [ ] `aroon_up` - Aroon Up **L**
- [ ] `aroon_down` - Aroon Down **L**

**Group F2.2: Trend Indicators (6 features)**
- [ ] `adx14` - ADX (trend strength) **L**
- [ ] `adx20` - ADX20 **L**
- [ ] `adx50` - ADX50 **L**
- [ ] `ema_alignment` - All EMAs aligned **L**
- [ ] `bollinger_band_width` - Bollinger Band Width **L**
- [ ] `aroon_oscillator` - Aroon Oscillator **L**

**Group F2.3: Price Position (8 features)**
- [ ] `pos_52w` - 52-week position (0=low, 1=high) **L**
- [ ] `sma20_ratio` - SMA20 ratio (for filtering) **L**
- [ ] `sma50_ratio` - SMA50 ratio (for filtering) **L**
- [ ] `sma200_ratio` - SMA200 ratio (for filtering) **L**
- [ ] `price_vs_ma200` - Price normalized to 200-day MA **L**
- [ ] `close_vs_weekly_sma20` - Price vs 20-week SMA ratio **L**
- [ ] `price_vs_vwap` - Distance from price to VWAP **L**
- [ ] `donchian_position` - Donchian position **L**

**Group F2.4: Volume Indicators (4 features)**
- [ ] `relative_volume` - Relative volume **L**
- [ ] `volume_breakout` - Volume spike on price breakout **L**
- [ ] `obv_momentum` - OBV rate of change **L**
- [ ] `volume_weighted_price` - VWAP **L**

**Group F2.5: Support/Resistance Levels (4 features)**
- [ ] `resistance_level_20d` - Nearest resistance (20-day high) **L**
- [ ] `resistance_level_50d` - Nearest resistance (50-day high) **L**
- [ ] `support_level_20d` - Nearest support (20-day low) **L**
- [ ] `support_level_50d` - Nearest support (50-day low) **L**

**Testing Checkpoint:** Validate filtering indicators

---

### **BLOCK FILTER-3: Composite Scores & Quality Metrics**
**Features: 20 | Risk: ⚠️ Medium | Complexity: 3-4 | Time: 8-10 hours | Dependencies: Blocks ML-1 through ML-22**

**Group F3.1: Quality Scores (10 features)**
- [ ] `signal_quality_score` - Overall signal quality score **L**
- [ ] `signal_strength` - Overall signal strength **L**
- [ ] `pattern_strength` - Overall pattern strength score **L**
- [ ] `trend_following_strength` - How well price follows trend **L**
- [ ] `support_resistance_strength` - Combined support/resistance strength score **L**
- [ ] `regime_strength` - Strength of current regime **L**
- [ ] `momentum_strength` - Strength of momentum signal **L**
- [ ] `gain_probability_score` - Combined score for gain probability **L**
- [ ] `market_quality` - Overall market quality measure **L**
- [ ] `signal_ensemble_score` - Ensemble score from multiple models **L**

**Group F3.2: Risk Metrics (5 features)**
- [ ] `false_positive_risk` - False positive risk indicator **L**
- [ ] `signal_risk_reward` - Signal risk-reward ratio **L**
- [ ] `gain_risk_ratio` - Gain potential vs risk ratio **L**
- [ ] `volatility_regime` - Current ATR percentile **L**
- [ ] `volatility_of_volatility` - VoV **L**

**Group F3.3: Consistency Metrics (5 features)**
- [ ] `signal_consistency` - Signal consistency across indicators **L**
- [ ] `momentum_consistency` - Consistency of momentum across timeframes **L**
- [ ] `trend_consistency` - % of days price above/below MA **L**
- [ ] `rs_consistency` - Consistency of relative strength **L**
- [ ] `regime_consistency` - Consistency of regime signals **L**

**Testing Checkpoint:** Validate composite score calculations

---

### **BLOCK FILTER-4: Multi-Timeframe Indicators for Screening**
**Features: 25 | Risk: ⚠️ Medium | Complexity: 2-3 | Time: 8-10 hours | Dependencies: Block ML-8, ML-9**

**Group F4.1: Weekly Indicators (15 features)**
- [ ] `weekly_return_1w` - 1-week log return **L**
- [ ] `weekly_return_2w` - 2-week log return **L**
- [ ] `weekly_return_4w` - 4-week log return **L**
- [ ] `weekly_sma_5w` - 5-week SMA **L**
- [ ] `weekly_sma_10w` - 10-week SMA **L**
- [ ] `weekly_sma_20w` - 20-week SMA **L**
- [ ] `weekly_ema_5w` - 5-week EMA **L**
- [ ] `weekly_ema_10w` - 10-week EMA **L**
- [ ] `weekly_ema_20w` - 20-week EMA **L**
- [ ] `weekly_rsi_14w` - Weekly RSI (14-week) **L**
- [ ] `weekly_rsi_7w` - Weekly RSI (7-week) **L**
- [ ] `weekly_macd_histogram` - Weekly MACD histogram **L**
- [ ] `weekly_macd_signal` - Weekly MACD signal line **L**
- [ ] `weekly_adx` - Weekly ADX (14-week) **L**
- [ ] `weekly_stochastic` - Weekly Stochastic (14-week) **L**

**Group F4.2: Monthly Indicators (5 features)**
- [ ] `monthly_return_1m` - 1-month log return **L**
- [ ] `monthly_return_3m` - 3-month log return **L**
- [ ] `monthly_sma_3m` - 3-month SMA **L**
- [ ] `monthly_sma_6m` - 6-month SMA **L**
- [ ] `monthly_rsi` - Monthly RSI **L**

**Group F4.3: Weekly Comparisons (5 features)**
- [ ] `close_vs_weekly_sma20` - Price vs 20-week SMA ratio **L**
- [ ] `close_vs_weekly_ema20` - Price vs 20-week EMA ratio **L**
- [ ] `weekly_volume_ratio` - Weekly volume vs 20-week average **L**
- [ ] `weekly_atr_pct` - Weekly ATR as % of price **L**
- [ ] `weekly_trend_strength` - Slope of weekly SMA20 **L**

**Testing Checkpoint:** Validate multi-timeframe indicators for filtering

---

### **BLOCK FILTER-5: Pattern Recognition for Screening**
**Features: 15 | Risk: ✅ Low | Complexity: 2-3 | Time: 6-8 hours | Dependencies: Block ML-10**

**Group F5.1: Single Candle Patterns (5 features)**
- [ ] `doji` - Doji pattern **L**
- [ ] `hammer` - Hammer pattern **L**
- [ ] `shooting_star` - Shooting star pattern **L**
- [ ] `marubozu` - Marubozu pattern (no wicks) **L**
- [ ] `spinning_top` - Spinning top pattern **L**

**Group F5.2: Candlestick Components (4 features)**
- [ ] `candle_body_pct` - Candle body % (body/range) **L**
- [ ] `candle_upper_wick_pct` - Upper wick % **L**
- [ ] `candle_lower_wick_pct` - Lower wick % **L**
- [ ] `candle_wicks_ratio` - Upper wick / lower wick ratio **L**

**Group F5.3: Pattern Quality (3 features)**
- [ ] `engulfing_strength` - Strength of engulfing pattern **L**
- [ ] `pattern_confirmation` - Pattern confirmation signal **L**
- [ ] `pattern_cluster` - Multiple patterns occurring **L**

**Group F5.4: Consecutive Days (2 features)**
- [ ] `consecutive_green_days` - Count of consecutive up days **L**
- [ ] `consecutive_red_days` - Count of consecutive down days **L**

**Group F5.5: Price Action (1 feature)**
- [ ] `swing_low_10d` - Recent swing low (10-day) **L**

**Testing Checkpoint:** Validate pattern recognition for filtering

---

### **BLOCK FILTER-6: Market Context for Screening**
**Features: 15 | Risk: ⚠️ Medium | Complexity: 2-3 | Time: 6-8 hours | Dependencies: Block ML-12, ML-13**

**Group F6.1: Market Regime (8 features)**
- [ ] `market_regime` - Bull/Bear/Sideways classification **L**
- [ ] `regime_strength` - Strength of current regime **L**
- [ ] `regime_duration` - Days in current regime **L**
- [ ] `market_volatility_regime` - Market volatility regime **L**
- [ ] `market_momentum_regime` - Market momentum regime **L**
- [ ] `regime_stability` - Regime stability measure **L**
- [ ] `market_sentiment` - Market sentiment indicator **L**
- [ ] `market_fear_greed` - Market fear/greed index proxy **L**

**Group F6.2: Market Direction (4 features)**
- [ ] `market_direction` - Market direction (up/down/sideways) **L**
- [ ] `market_direction_strength` - Strength of market direction **L**
- [ ] `market_trend_alignment` - Stock trend vs market trend alignment **L**
- [ ] `market_momentum` - Market momentum indicator **L**

**Group F6.3: Market Structure (3 features)**
- [ ] `market_support_resistance` - Market support/resistance levels **L**
- [ ] `market_breadth` - Market breadth indicator **L**
- [ ] `market_leadership` - Market leadership indicator **L**

**Testing Checkpoint:** Validate market context for filtering

---

### **BLOCK FILTER-7: Relative Strength for Screening**
**Features: 12 | Risk: ⚠️ Medium | Complexity: 2-3 | Time: 6-8 hours | Dependencies: Block ML-14**

**Group F7.1: Relative Strength (12 features)**
- [ ] `relative_strength_spy` - Stock return vs SPY return (20-day) **L**
- [ ] `relative_strength_spy_50d` - Stock return vs SPY return (50-day) **L**
- [ ] `rs_rank_20d` - Relative strength rank (0-100) over 20 days **L**
- [ ] `rs_rank_50d` - Relative strength rank (0-100) over 50 days **L**
- [ ] `rs_rank_100d` - Relative strength rank (0-100) over 100 days **L**
- [ ] `rs_momentum` - Rate of change of relative strength **L**
- [ ] `outperformance_flag` - Binary: outperforming market **L**
- [ ] `rs_vs_price` - Relative strength vs price divergence **L**
- [ ] `rs_consistency` - Consistency of relative strength **L**
- [ ] `rs_regime` - Relative strength regime (strong/weak/neutral) **L**
- [ ] `rs_trend` - Relative strength trend direction **L**
- [ ] `relative_strength_sector` - Stock vs sector return (if available) **L**

**Testing Checkpoint:** Validate relative strength for filtering

---

### **BLOCK FILTER-8: Pivot Points & Fibonacci for Screening**
**Features: 3 | Risk: ✅ Low | Complexity: 2 | Time: 3-4 hours | Dependencies: Block ML-6**

**Group F8.1: Pivot & Fibonacci (3 features)**
- [ ] `pivot_point` - Classic pivot point calculation (H+L+C)/3 **L**
- [ ] `pivot_resistance_1` - Pivot resistance level 1 **L**
- [ ] `pivot_support_1` - Pivot support level 1 **L**

**Testing Checkpoint:** Quick validation

---

**FILTER FEATURES SUMMARY:**
- **Total Filter Blocks:** 8 blocks
- **Total Filter Features:** ~150 features
- **Estimated Total Time:** ~50-60 hours
- **Testing Checkpoints:** Every block

---

## **SECTION 3: CANNOT CALCULATE (Need Additional Data)**

*Features that require data sources beyond daily OHLCV*

---

### **CATEGORY: Volume Profile (True Implementation)**
**Required Data:** Intraday tick data or volume at price levels (1-minute bars with volume distribution)

**Features (4 features):**
- [ ] `volume_profile_poc` - Point of Control (true implementation) - **Needs:** Intraday volume at price levels
- [ ] `volume_profile_vah` - Value Area High (true implementation) - **Needs:** Intraday volume at price levels
- [ ] `volume_profile_val` - Value Area Low (true implementation) - **Needs:** Intraday volume at price levels
- [ ] `price_vs_poc` - Distance to POC (true implementation) - **Needs:** Intraday volume at price levels

**Data Source Options:**
- Polygon.io API (paid)
- Alpaca API (paid)
- Interactive Brokers API (paid)
- yfinance intraday data (limited, 1-minute bars available but may have gaps)

**Note:** Current implementation uses daily approximations. True volume profile requires knowing how volume was distributed across price levels during the day.

---

### **CATEGORY: Sector/Industry Features**
**Required Data:** Sector mapping file + Sector ETF data (XLF, XLK, XLE, etc.)

**Features (10 features):**
- [ ] `sector_momentum` - Sector performance (20-day return) - **Needs:** Sector ETF OHLCV data
- [ ] `sector_relative_strength` - Stock vs sector return - **Needs:** Sector ETF OHLCV data + Sector mapping
- [ ] `industry_group_rank` - Industry group performance rank - **Needs:** Industry group indices + Industry mapping
- [ ] `sector_trend` - Sector trend direction - **Needs:** Sector ETF OHLCV data
- [ ] `sector_volatility` - Sector volatility - **Needs:** Sector ETF OHLCV data
- [ ] `sector_beta` - Sector beta vs market - **Needs:** Sector ETF OHLCV data
- [ ] `industry_momentum` - Industry momentum - **Needs:** Industry group indices
- [ ] `sector_rotation` - Sector rotation signal - **Needs:** All sector ETF data
- [ ] `industry_strength` - Industry strength rank - **Needs:** Industry group indices
- [ ] `sector_regime` - Sector regime (outperforming/underperforming) - **Needs:** Sector ETF OHLCV data

**Data Source Options:**
- Sector mapping: Yahoo Finance (free), SEC filings (free), or manual creation
- Sector ETFs: yfinance (free) - XLF (Financials), XLK (Tech), XLE (Energy), etc.
- Industry groups: Can create from sector data or use industry indices

**Implementation Priority:** Medium - Can be added once sector mapping is created

---

### **CATEGORY: Earnings Features**
**Required Data:** Earnings announcement dates

**Features (1 feature):**
- [ ] `days_since_earnings` - Days since last earnings - **Needs:** Earnings announcement dates

**Data Source Options:**
- Alpha Vantage API (free tier available)
- Polygon.io API (paid)
- yfinance (limited earnings data)
- Manual collection from company websites

**Implementation Priority:** Low - Optional feature

---

### **CATEGORY: Market Microstructure (True Implementation)**
**Required Data:** Level 2 order book data (bid/ask prices and sizes)

**Features (8 features):**
- [ ] `bid_ask_spread` - Bid-ask spread (true implementation) - **Needs:** Level 2 order book data
- [ ] `order_imbalance` - Order flow imbalance (true implementation) - **Needs:** Level 2 order book data
- [ ] `price_impact` - Price impact of trades (true implementation) - **Needs:** Level 2 order book data
- [ ] `market_depth` - Market depth (true implementation) - **Needs:** Level 2 order book data
- [ ] `order_flow_imbalance` - Order flow imbalance (true implementation) - **Needs:** Level 2 order book data
- [ ] `price_efficiency` - Price efficiency (true implementation) - **Needs:** Level 2 order book data
- [ ] `transaction_cost` - Transaction cost (true implementation) - **Needs:** Level 2 order book data
- [ ] `market_quality` - Market quality (true implementation) - **Needs:** Level 2 order book data

**Data Source Options:**
- Polygon.io API (paid, Level 2 data)
- Alpaca API (paid, Level 2 data)
- Interactive Brokers API (paid, Level 2 data)
- Not available from yfinance

**Note:** Current implementation uses proxies from daily data. True microstructure features require Level 2 data.

**Implementation Priority:** Low - More relevant for day trading than swing trading

---

### **CATEGORY: Market Breadth (True Implementation)**
**Required Data:** Advancing/declining stocks data

**Features (1 feature):**
- [ ] `market_breadth` - Market breadth indicator (true implementation) - **Needs:** Advancing/declining stocks count

**Data Source Options:**
- Yahoo Finance (free, limited)
- Alpha Vantage API (free tier)
- Manual calculation from market data

**Implementation Priority:** Low - Can approximate from SPY data

---

**CANNOT CALCULATE SUMMARY:**
- **Total Features:** ~24 features
- **Data Requirements:**
  - Intraday data: 4 features (Volume Profile)
  - Sector data: 10 features (Sector/Industry)
  - Earnings data: 1 feature (Earnings)
  - Level 2 data: 8 features (Microstructure)
  - Market breadth: 1 feature (Market Breadth)

---

## **IMPLEMENTATION PRIORITY SUMMARY**

### **Phase 1: Foundation (Blocks ML-1, ML-2)**
- **Features:** 60 features
- **Time:** 8-12 hours
- **Priority:** Highest - Foundation for everything else

### **Phase 2: Core Features (Blocks ML-3, ML-4, ML-5)**
- **Features:** 58 features
- **Time:** 20-26 hours
- **Priority:** High - High-impact features

### **Phase 3: Support/Resistance & Volume (Blocks ML-6, ML-7)**
- **Features:** 38 features
- **Time:** 20-24 hours
- **Priority:** High - Critical for entry/exit

### **Phase 4: Multi-Timeframe (Blocks ML-8, ML-9)**
- **Features:** 25 features
- **Time:** 18-23 hours
- **Priority:** High - Multi-timeframe context

### **Phase 5: Patterns & Market Context (Blocks ML-10, ML-11, ML-12, ML-13)**
- **Features:** 58 features
- **Time:** 40-50 hours
- **Priority:** Medium-High - Pattern recognition and market context

### **Phase 6: Advanced Features (Blocks ML-14, ML-15, ML-16, ML-17, ML-18, ML-19)**
- **Features:** 66 features
- **Time:** 60-75 hours
- **Priority:** Medium - Advanced indicators

### **Phase 7: Interactions & Predictive (Blocks ML-20, ML-21, ML-22)**
- **Features:** 55 features
- **Time:** 36-48 hours
- **Priority:** Medium - Feature interactions and target-specific

### **Phase 8: Remaining ML Features (Blocks ML-23, ML-24, ML-25)**
- **Features:** 11 features
- **Time:** 16-22 hours
- **Priority:** Medium - Final ML features

### **Phase 9: Filter Features (Blocks FILTER-1 through FILTER-8)**
- **Features:** ~150 features
- **Time:** 50-60 hours
- **Priority:** Low - Build after ML features are stable

---

## **TOTAL IMPLEMENTATION ESTIMATE**

- **ML Features:** 25 blocks, ~330 features, ~250-300 hours
- **Filter Features:** 8 blocks, ~150 features, ~50-60 hours
- **Cannot Calculate:** ~24 features (need additional data)
- **Grand Total:** ~504 features (330 ML + 150 Filter + 24 Cannot Calculate)

---

**Last Updated:** 2025-01-19  
**Next Steps:** Begin with Block ML-1 (Foundation - Price & Returns)
