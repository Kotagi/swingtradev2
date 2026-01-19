# New Dawn Feature Set - Implementation Checklist

**Feature Set:** v3_New_Dawn  
**Total Features:** ~383 features  
**Status:** Implementation tracking checklist

---

## **PART I: EXISTING FEATURES (56 features) - KEEP & VERIFY**

### ✅ **1. Price & Normalization Features (3 features)**
- [x] `price` - Raw closing price
- [x] `price_log` - Log of closing price (ln(close))
- [x] `price_vs_ma200` - Price normalized to 200-day MA

### ✅ **2. Return Features (7 features)**
- [x] `daily_return` - Daily return % (v2: no clipping)
- [x] `gap_pct` - Gap % (open - prev_close) / prev_close (v2: no clipping)
- [x] `weekly_return_5d` - 5-day return % (v2: no clipping)
- [x] `monthly_return_21d` - 21-day return % (v2: no clipping)
- [x] `quarterly_return_63d` - 63-day return % (v2: no clipping)
- [x] `ytd_return` - Year-to-Date return % (v2: no clipping)
- [ ] `log_return_1d` - **MISSING FROM REGISTRY** (implemented but not registered)

### ✅ **3. 52-Week Position Features (3 features)**
- [x] `dist_52w_high` - Distance to 52-week high (v2: no clipping)
- [x] `dist_52w_low` - Distance to 52-week low (v2: no clipping)
- [x] `pos_52w` - 52-week position (0=low, 1=high)

### ✅ **4. Moving Average Features (8 features)**
- [x] `sma20_ratio` - SMA20 ratio (close/SMA20) (v2: no clipping)
- [x] `sma50_ratio` - SMA50 ratio (close/SMA50) (v2: no clipping)
- [x] `sma200_ratio` - SMA200 ratio (close/SMA200) (v2: no clipping)
- [x] `sma20_sma50_ratio` - SMA20/SMA50 ratio (v2: no clipping)
- [x] `sma50_sma200_ratio` - SMA50/SMA200 ratio (v2: no clipping)
- [x] `sma50_slope` - SMA50 slope (v2: no clipping)
- [x] `sma200_slope` - SMA200 slope (v2: no clipping)
- [ ] `ema20_ratio` - **MISSING FROM REGISTRY** (mentioned in spec but not registered)

### ✅ **5. Volatility Features (5 features)**
- [x] `volatility_5d` - 5-day volatility (std of returns) (v2: no clipping)
- [x] `volatility_21d` - 21-day volatility (std of returns) (v2: no clipping)
- [x] `volatility_ratio` - Volatility ratio: vol5/vol21 (v2: no clipping)
- [x] `atr14_normalized` - Normalized ATR14 (ATR14/close) (v2: no clipping)
- [x] `volatility_of_volatility` - Volatility-of-Volatility (v2: no clipping)

### ✅ **6. Volume Features (4 features)**
- [x] `log_volume` - Log volume (log1p(volume))
- [x] `log_avg_volume_20d` - Log average volume 20-day
- [x] `relative_volume` - Relative volume (v2: no pre-clipping)
- [x] `obv_momentum` - OBV rate of change (v2: no clipping)

### ✅ **7. Momentum & Oscillator Features (15 features)**
- [x] `rsi14` - RSI14 centered ((rsi-50)/50) in [-1, +1]
- [x] `macd_histogram_normalized` - MACD histogram normalized by price
- [x] `ppo_histogram` - PPO histogram (v2: no clipping)
- [x] `dpo` - Detrended Price Oscillator (v2: no clipping)
- [x] `roc10` - ROC10: short-term momentum (v2: no clipping)
- [x] `roc20` - ROC20: medium-term momentum (v2: no clipping)
- [x] `stochastic_k14` - Stochastic %K in [0, 1]
- [x] `cci20` - Commodity Channel Index normalized
- [x] `williams_r14` - Williams %R normalized to [0, 1]
- [x] `kama_slope` - KAMA Slope (v2: no clipping)
- [x] `aroon_up` - Aroon Up in [0, 1]
- [x] `aroon_down` - Aroon Down in [0, 1]
- [x] `aroon_oscillator` - Aroon Oscillator normalized to [0, 1]
- [x] `chaikin_money_flow` - Chaikin Money Flow in [-1, 1]
- [x] `trend_residual` - Trend residual (v2: no clipping)

### ✅ **8. Candlestick Features (3 features)**
- [x] `candle_body_pct` - Candle body % (body/range) in [0, 1]
- [x] `candle_upper_wick_pct` - Upper wick % (upper/range) in [0, 1]
- [x] `candle_lower_wick_pct` - Lower wick % (lower/range) in [0, 1]
- [ ] `close_position_in_range` - **MISSING FROM REGISTRY** (mentioned in spec but not registered)

### ✅ **9. Price Action Features (3 features)**
- [x] `higher_high_10d` - Higher high (10-day) binary flag
- [x] `higher_low_10d` - Higher low (10-day) binary flag
- [x] `swing_low_10d` - Recent swing low (10-day)

### ✅ **10. Advanced Technical Features (4 features)**
- [x] `bollinger_band_width` - Bollinger Band Width (log normalized)
- [x] `adx14` - ADX (trend strength) normalized to [0, 1]
- [x] `fractal_dimension_index` - Fractal Dimension Index normalized to [0, 1]
- [x] `hurst_exponent` - Hurst Exponent in [0, 1]

### ✅ **11. Market Context Features (3 features)**
- [x] `mkt_spy_dist_sma200` - SPY Distance from SMA200 (z-score normalized)
- [x] `mkt_spy_sma200_slope` - SPY SMA200 Slope (percentile rank normalized)
- [x] `beta_spy_252d` - Rolling beta vs SPY (252-day) normalized to [0, 1]

### ✅ **12. Breakout & Channel Features (4 features)**
- [x] `donchian_position` - Donchian position in [0, 1]
- [x] `donchian_breakout` - Donchian breakout binary flag
- [x] `ttm_squeeze_on` - TTM Squeeze binary flag
- [x] `ttm_squeeze_momentum` - TTM Squeeze momentum

### ✅ **13. Advanced Trend Features (2 features)**
- [x] `price_curvature` - Price Curvature (v2: normalized by price, no clipping)
- [x] `price_vs_ema50` - Price vs EMA50 ratio (if exists)

---

## **PART II: MISSING FEATURES TO ADD (3 features)**

### 🔴 **Missing Features (Implemented but not in registry)**
- [ ] `log_return_1d` - 1-day log return: ln(close_t / close_{t-1})
- [ ] `ema20_ratio` - EMA20 ratio (close/EMA20)
- [ ] `close_position_in_range` - Close position in daily range (close-low)/(high-low) in [0, 1]

---

## **PART III: ENHANCED VARIANTS (Multi-Timeframe) - 15 features**

### 📈 **RSI Variants (2 features)**
- [ ] `rsi7` - RSI7: shorter-term momentum
- [ ] `rsi21` - RSI21: longer-term momentum

### 📈 **ADX Variants (2 features)**
- [ ] `adx20` - ADX20: medium-term trend strength
- [ ] `adx50` - ADX50: long-term trend strength

### 📈 **ROC Variants (1 feature)**
- [ ] `roc5` - ROC5: very short-term momentum

### 📈 **Stochastic Variants (1 feature)**
- [ ] `stochastic_d14` - Stochastic %D (signal line)

### 📈 **EMA Variants (2 features)**
- [ ] `ema50_ratio` - EMA50 ratio (close/EMA50)
- [ ] `ema200_ratio` - EMA200 ratio (close/EMA200)

### 📈 **MACD Variants (2 features)**
- [ ] `macd_line` - MACD line (not just histogram)
- [ ] `macd_signal` - MACD signal line

### 📈 **Additional Momentum (5 features)**
- [ ] `momentum_5d` - 5-day price momentum
- [ ] `momentum_15d` - 15-day price momentum
- [ ] `momentum_30d` - 30-day price momentum
- [ ] `rsi_momentum` - RSI momentum (rate of change)
- [ ] `macd_momentum` - MACD momentum (rate of change)

---

## **PART IV: NEW FEATURES FOR NEW DAWN (~309 features)**

### **CATEGORY 1: MULTI-TIMEFRAME FEATURES (Weekly/Monthly) - 25 features**

#### Weekly Features (20 features)
- [ ] `weekly_return_1w` - 1-week log return
- [ ] `weekly_return_2w` - 2-week log return
- [ ] `weekly_return_4w` - 4-week log return
- [ ] `weekly_sma_5w` - 5-week SMA
- [ ] `weekly_sma_10w` - 10-week SMA
- [ ] `weekly_sma_20w` - 20-week SMA
- [ ] `weekly_ema_5w` - 5-week EMA
- [ ] `weekly_ema_10w` - 10-week EMA
- [ ] `weekly_ema_20w` - 20-week EMA
- [ ] `weekly_rsi_14w` - Weekly RSI (14-week)
- [ ] `weekly_rsi_7w` - Weekly RSI (7-week)
- [ ] `weekly_macd_histogram` - Weekly MACD histogram
- [ ] `weekly_macd_signal` - Weekly MACD signal line
- [ ] `close_vs_weekly_sma20` - Price vs 20-week SMA ratio
- [ ] `close_vs_weekly_ema20` - Price vs 20-week EMA ratio
- [ ] `weekly_volume_ratio` - Weekly volume vs 20-week average
- [ ] `weekly_atr_pct` - Weekly ATR as % of price
- [ ] `weekly_trend_strength` - Slope of weekly SMA20
- [ ] `weekly_adx` - Weekly ADX (14-week)
- [ ] `weekly_stochastic` - Weekly Stochastic (14-week)

#### Monthly Features (5 features)
- [ ] `monthly_return_1m` - 1-month log return
- [ ] `monthly_return_3m` - 3-month log return
- [ ] `monthly_sma_3m` - 3-month SMA
- [ ] `monthly_sma_6m` - 6-month SMA
- [ ] `monthly_rsi` - Monthly RSI

---

### **CATEGORY 2: VOLATILITY REGIME FEATURES - 18 features**

- [ ] `volatility_regime` - Current ATR percentile (0-100) over 252 days
- [ ] `volatility_trend` - ATR slope (increasing/decreasing volatility)
- [ ] `bb_squeeze` - Bollinger Band squeeze indicator (low volatility)
- [ ] `bb_expansion` - Bollinger Band expansion (high volatility)
- [ ] `atr_ratio_20d` - Current ATR / 20-day ATR average
- [ ] `atr_ratio_252d` - Current ATR / 252-day ATR average
- [ ] `volatility_percentile_20d` - ATR percentile over 20 days
- [ ] `volatility_percentile_252d` - ATR percentile over 252 days
- [ ] `high_volatility_flag` - Binary: ATR > 75th percentile
- [ ] `low_volatility_flag` - Binary: ATR < 25th percentile
- [ ] `parkinson_volatility` - High-low based volatility estimator
- [ ] `garman_klass_volatility` - OHLC-based volatility estimator
- [ ] `rogers_satchell_volatility` - OHLC volatility with drift correction
- [ ] `yang_zhang_volatility` - Overnight + intraday volatility
- [ ] `realized_volatility_5d` - 5-day realized volatility
- [ ] `realized_volatility_20d` - 20-day realized volatility
- [ ] `volatility_clustering` - Volatility clustering measure
- [ ] `volatility_regime_change` - Binary: volatility regime changed recently

---

### **CATEGORY 3: TREND STRENGTH & QUALITY FEATURES - 22 features**

- [ ] `trend_strength_20d` - ADX over 20 days
- [ ] `trend_strength_50d` - ADX over 50 days
- [ ] `trend_consistency` - % of days price above/below MA over lookback
- [ ] `ema_alignment` - All EMAs aligned (bullish/bearish/neutral): -1/0/1
- [ ] `sma_slope_20d` - Slope of 20-day SMA
- [ ] `sma_slope_50d` - Slope of 50-day SMA
- [ ] `ema_slope_20d` - Slope of 20-day EMA
- [ ] `ema_slope_50d` - Slope of 50-day EMA
- [ ] `trend_duration` - Days since last trend change
- [ ] `trend_reversal_signal` - Potential trend reversal indicator
- [ ] `price_vs_all_mas` - Count of MAs price is above
- [ ] `ma_crossover_bullish` - Binary: SMA20 > SMA50 > SMA200
- [ ] `ma_crossover_bearish` - Binary: SMA20 < SMA50 < SMA200
- [ ] `trend_acceleration` - Rate of change of trend slope
- [ ] `trend_divergence` - Price vs trend divergence
- [ ] `trend_pullback_strength` - Strength of pullback to trend
- [ ] `trend_breakout_strength` - Strength of trend breakout
- [ ] `trend_consolidation` - Binary: price consolidating within trend
- [ ] `trend_exhaustion` - Trend exhaustion indicator
- [ ] `trend_following_strength` - How well price follows trend
- [ ] `trend_vs_mean_reversion` - Trend vs mean-reversion signal
- [ ] `trend_regime` - Trend regime classification

---

### **CATEGORY 4: VOLUME PROFILE & DISTRIBUTION FEATURES - 20 features**

- [ ] `volume_profile_poc` - Point of Control (price level with most volume)
- [ ] `volume_profile_vah` - Value Area High (70% volume above)
- [ ] `volume_profile_val` - Value Area Low (70% volume below)
- [ ] `price_vs_poc` - Distance from current price to POC
- [ ] `price_vs_vah` - Distance from price to Value Area High
- [ ] `price_vs_val` - Distance from price to Value Area Low
- [ ] `volume_weighted_price` - VWAP (Volume Weighted Average Price)
- [ ] `price_vs_vwap` - Distance from price to VWAP
- [ ] `vwap_slope` - VWAP trend direction
- [ ] `vwap_distance_pct` - % distance from VWAP
- [ ] `volume_distribution` - Volume concentration metric
- [ ] `volume_climax` - Unusually high volume days (volume > 2x average)
- [ ] `volume_dry_up` - Unusually low volume days (volume < 0.5x average)
- [ ] `volume_trend` - Volume moving average slope
- [ ] `volume_breakout` - Volume spike on price breakout
- [ ] `volume_divergence` - Volume vs price divergence
- [ ] `volume_momentum` - Volume momentum (rate of change)
- [ ] `volume_profile_width` - Width of volume profile
- [ ] `volume_imbalance` - Buy vs sell volume imbalance
- [ ] `volume_at_price` - Volume at current price level

---

### **CATEGORY 5: SUPPORT & RESISTANCE FEATURES - 18 features**

- [ ] `resistance_level_20d` - Nearest resistance (20-day high)
- [ ] `resistance_level_50d` - Nearest resistance (50-day high)
- [ ] `resistance_level_100d` - Nearest resistance (100-day high)
- [ ] `support_level_20d` - Nearest support (20-day low)
- [ ] `support_level_50d` - Nearest support (50-day low)
- [ ] `support_level_100d` - Nearest support (100-day low)
- [ ] `distance_to_resistance` - % distance to nearest resistance
- [ ] `distance_to_support` - % distance to nearest support
- [ ] `price_near_resistance` - Binary: within 2% of resistance
- [ ] `price_near_support` - Binary: within 2% of support
- [ ] `resistance_touches` - Number of times price touched resistance
- [ ] `support_touches` - Number of times price touched support
- [ ] `pivot_point` - Classic pivot point calculation (H+L+C)/3
- [ ] `pivot_resistance_1` - Pivot resistance level 1
- [ ] `pivot_support_1` - Pivot support level 1
- [ ] `fibonacci_levels` - Distance to Fibonacci retracement levels
- [ ] `fibonacci_retracement` - Current Fibonacci retracement level
- [ ] `support_resistance_strength` - Combined support/resistance strength score

---

### **CATEGORY 6: ADVANCED MOMENTUM INDICATORS - 18 features**

- [ ] `momentum_10d` - Price momentum (close - close[10])
- [ ] `momentum_20d` - Price momentum (close - close[20])
- [ ] `momentum_50d` - Price momentum (close - close[50])
- [ ] `mfi` - Money Flow Index (volume-weighted RSI)
- [ ] `tsi` - True Strength Index
- [ ] `ultimate_oscillator` - Ultimate Oscillator (combines 3 timeframes)
- [ ] `awesome_oscillator` - Awesome Oscillator (5-period SMA - 34-period SMA)
- [ ] `momentum_divergence` - RSI/price divergence signal
- [ ] `momentum_consistency` - Consistency of momentum across timeframes
- [ ] `momentum_acceleration` - Rate of change of momentum
- [ ] `momentum_strength` - Strength of momentum signal
- [ ] `momentum_regime` - Momentum regime (strong/weak/neutral)
- [ ] `momentum_vs_price` - Momentum vs price divergence
- [ ] `momentum_persistence` - Momentum persistence measure
- [ ] `momentum_exhaustion` - Momentum exhaustion indicator
- [ ] `momentum_reversal_signal` - Momentum reversal signal
- [ ] `momentum_cross` - Momentum crossover signal
- [ ] `momentum_rank` - Momentum rank vs historical

---

### **CATEGORY 7: PRICE ACTION PATTERNS - 25 features**

- [ ] `three_white_soldiers` - Bullish 3-candle pattern
- [ ] `three_black_crows` - Bearish 3-candle pattern
- [ ] `hanging_man` - Reversal pattern (bearish)
- [ ] `inverted_hammer` - Reversal pattern (bullish)
- [ ] `piercing_pattern` - Bullish reversal (2-candle)
- [ ] `dark_cloud_cover` - Bearish reversal (2-candle)
- [ ] `harami_bullish` - Bullish reversal (2-candle)
- [ ] `harami_bearish` - Bearish reversal (2-candle)
- [ ] `engulfing_bullish` - Bullish engulfing pattern
- [ ] `engulfing_bearish` - Bearish engulfing pattern
- [ ] `engulfing_strength` - Strength of engulfing pattern
- [ ] `candle_wicks_ratio` - Upper wick / lower wick ratio
- [ ] `consecutive_green_days` - Count of consecutive up days
- [ ] `consecutive_red_days` - Count of consecutive down days
- [ ] `inside_bar` - Inside bar pattern
- [ ] `outside_bar` - Outside bar pattern
- [ ] `doji` - Doji pattern
- [ ] `hammer` - Hammer pattern
- [ ] `shooting_star` - Shooting star pattern
- [ ] `marubozu` - Marubozu pattern (no wicks)
- [ ] `spinning_top` - Spinning top pattern
- [ ] `pattern_strength` - Overall pattern strength score
- [ ] `pattern_confirmation` - Pattern confirmation signal
- [ ] `pattern_divergence` - Pattern vs price divergence
- [ ] `pattern_cluster` - Multiple patterns occurring

---

### **CATEGORY 8: MARKET REGIME INDICATORS - 15 features**

- [ ] `market_regime` - Bull/Bear/Sideways classification (0/1/2)
- [ ] `regime_strength` - Strength of current regime (0-1)
- [ ] `regime_duration` - Days in current regime
- [ ] `regime_change_probability` - Likelihood of regime change (0-1)
- [ ] `trending_market_flag` - Binary: strong trend vs choppy
- [ ] `choppy_market_flag` - Binary: sideways/choppy market
- [ ] `bull_market_flag` - Binary: bullish regime
- [ ] `bear_market_flag` - Binary: bearish regime
- [ ] `market_volatility_regime` - Market volatility regime (low/normal/high)
- [ ] `market_momentum_regime` - Market momentum regime (strong/weak/neutral)
- [ ] `regime_transition` - Binary: regime transition occurring
- [ ] `regime_stability` - Regime stability measure
- [ ] `market_sentiment` - Market sentiment indicator
- [ ] `market_fear_greed` - Market fear/greed index proxy
- [ ] `regime_consistency` - Consistency of regime signals

---

### **CATEGORY 9: RELATIVE STRENGTH FEATURES - 12 features**

- [ ] `relative_strength_spy` - Stock return vs SPY return (20-day)
- [ ] `relative_strength_spy_50d` - Stock return vs SPY return (50-day)
- [ ] `relative_strength_sector` - Stock return vs sector return
- [ ] `rs_rank_20d` - Relative strength rank (0-100) over 20 days
- [ ] `rs_rank_50d` - Relative strength rank (0-100) over 50 days
- [ ] `rs_rank_100d` - Relative strength rank (0-100) over 100 days
- [ ] `rs_momentum` - Rate of change of relative strength
- [ ] `outperformance_flag` - Binary: outperforming market
- [ ] `rs_vs_price` - Relative strength vs price divergence
- [ ] `rs_consistency` - Consistency of relative strength
- [ ] `rs_regime` - Relative strength regime (strong/weak/neutral)
- [ ] `rs_trend` - Relative strength trend direction

---

### **CATEGORY 10: TIME-BASED FEATURES - 15 features**

- [ ] `day_of_week` - Monday=0, Friday=4 (cyclical encoding: sin/cos)
- [ ] `day_of_week_sin` - Day of week (sine encoding)
- [ ] `day_of_week_cos` - Day of week (cosine encoding)
- [ ] `day_of_month` - 1-31 (cyclical encoding: sin/cos)
- [ ] `day_of_month_sin` - Day of month (sine encoding)
- [ ] `day_of_month_cos` - Day of month (cosine encoding)
- [ ] `month_of_year` - 1-12 (cyclical encoding: sin/cos)
- [ ] `month_of_year_sin` - Month of year (sine encoding)
- [ ] `month_of_year_cos` - Month of year (cosine encoding)
- [ ] `quarter` - Q1-Q4 (cyclical encoding)
- [ ] `is_month_end` - Binary: last 3 days of month
- [ ] `is_quarter_end` - Binary: last week of quarter
- [ ] `is_year_end` - Binary: December
- [ ] `days_since_earnings` - Days since last earnings (if available)
- [ ] `trading_day_of_year` - Trading day of year (1-252)

---

### **CATEGORY 11: FEATURE INTERACTIONS - 20 features**

- [ ] `rsi_x_volume` - RSI × Volume ratio
- [ ] `rsi_x_atr` - RSI × ATR pct
- [ ] `macd_x_volume` - MACD × Volume ratio
- [ ] `trend_x_momentum` - Trend strength × Momentum
- [ ] `volatility_x_volume` - Volatility × Volume
- [ ] `price_position_x_rsi` - Price percentile × RSI
- [ ] `rsi_x_trend_strength` - RSI × Trend strength
- [ ] `volume_x_momentum` - Volume × Momentum
- [ ] `volatility_x_trend` - Volatility × Trend strength
- [ ] `support_resistance_x_momentum` - Support/resistance distance × Momentum
- [ ] `rsi_x_support_distance` - RSI × Distance to support
- [ ] `volume_x_breakout` - Volume × Breakout signal
- [ ] `trend_x_volatility` - Trend strength × Volatility
- [ ] `momentum_x_regime` - Momentum × Market regime
- [ ] `volume_x_regime` - Volume × Market regime
- [ ] `rsi_x_relative_strength` - RSI × Relative strength
- [ ] `trend_x_relative_strength` - Trend × Relative strength
- [ ] `volatility_x_regime_strength` - Volatility × Regime strength
- [ ] `momentum_x_time` - Momentum × Time feature
- [ ] `volume_x_time` - Volume × Time feature

---

### **CATEGORY 12: STATISTICAL FEATURES - 15 features**

- [ ] `price_skewness_20d` - Price distribution skewness (20-day)
- [ ] `price_kurtosis_20d` - Price distribution kurtosis (20-day)
- [ ] `returns_skewness_20d` - Return distribution skewness (20-day)
- [ ] `returns_kurtosis_20d` - Return distribution kurtosis (20-day)
- [ ] `price_autocorrelation_1d` - Price autocorrelation (1-day lag)
- [ ] `price_autocorrelation_5d` - Price autocorrelation (5-day lag)
- [ ] `volume_autocorrelation` - Volume autocorrelation
- [ ] `returns_autocorrelation` - Return autocorrelation
- [ ] `price_variance_ratio` - Variance ratio test statistic
- [ ] `price_half_life` - Price mean-reversion half-life
- [ ] `returns_half_life` - Return mean-reversion half-life
- [ ] `price_stationarity` - Price stationarity test (ADF)
- [ ] `returns_stationarity` - Return stationarity test (ADF)
- [ ] `price_cointegration` - Price cointegration with market (if SPY available)
- [ ] `statistical_regime` - Statistical regime (trending/mean-reverting/random)

---

### **CATEGORY 13: ADVANCED VOLATILITY FEATURES - 12 features**

- [ ] `volatility_skewness` - Volatility distribution skewness
- [ ] `volatility_kurtosis` - Volatility distribution kurtosis
- [ ] `volatility_autocorrelation` - Volatility autocorrelation
- [ ] `volatility_mean_reversion` - Volatility mean-reversion speed
- [ ] `volatility_regime_persistence` - Volatility regime persistence
- [ ] `volatility_shock` - Recent volatility shock indicator
- [ ] `volatility_normalized_returns` - Returns normalized by volatility
- [ ] `volatility_forecast` - Volatility forecast (GARCH-like)
- [ ] `volatility_term_structure` - Volatility term structure (short vs long)
- [ ] `volatility_risk_premium` - Volatility risk premium proxy
- [ ] `volatility_jump` - Volatility jump detection
- [ ] `volatility_smoothness` - Volatility smoothness measure

---

### **CATEGORY 14: MARKET MICROSTRUCTURE FEATURES - 8 features**

- [ ] `liquidity_measure` - Volume/volatility ratio
- [ ] `price_impact_proxy` - Price impact proxy (volume/price change)
- [ ] `bid_ask_spread_proxy` - Bid-ask spread proxy (high-low/close)
- [ ] `order_flow_imbalance` - Order flow imbalance proxy
- [ ] `market_depth_proxy` - Market depth proxy
- [ ] `price_efficiency` - Price efficiency measure
- [ ] `transaction_cost_proxy` - Transaction cost proxy
- [ ] `market_quality` - Overall market quality measure

---

### **CATEGORY 15: SECTOR/INDUSTRY FEATURES - 10 features**

- [ ] `sector_momentum` - Sector performance (20-day return)
- [ ] `sector_relative_strength` - Stock vs sector return
- [ ] `industry_group_rank` - Industry group performance rank
- [ ] `sector_trend` - Sector trend direction
- [ ] `sector_volatility` - Sector volatility
- [ ] `sector_beta` - Sector beta vs market
- [ ] `industry_momentum` - Industry momentum
- [ ] `sector_rotation` - Sector rotation signal
- [ ] `industry_strength` - Industry strength rank
- [ ] `sector_regime` - Sector regime (outperforming/underperforming)

---

### **CATEGORY 16: PREDICTIVE FEATURES FOR PERCENT GAIN - 15 features**

- [ ] `historical_gain_probability` - Historical probability of X% gain in Y days
- [ ] `gain_momentum` - Momentum toward target gain
- [ ] `gain_acceleration` - Acceleration toward target gain
- [ ] `target_distance` - Distance to target gain %
- [ ] `gain_probability_score` - Combined score for gain probability
- [ ] `gain_timing` - Optimal timing for gain
- [ ] `gain_risk_ratio` - Gain potential vs risk ratio
- [ ] `gain_consistency` - Consistency of gain achievement
- [ ] `gain_regime` - Gain regime (high/low probability)
- [ ] `gain_momentum_strength` - Strength of gain momentum
- [ ] `gain_volume_confirmation` - Volume confirmation for gain
- [ ] `gain_trend_alignment` - Trend alignment for gain
- [ ] `gain_volatility_context` - Volatility context for gain
- [ ] `gain_support_level` - Support level for gain
- [ ] `gain_breakout_signal` - Breakout signal for gain

---

### **CATEGORY 17: MARKET DIRECTION FEATURES - 12 features**

- [ ] `market_direction` - Market direction (up/down/sideways)
- [ ] `market_direction_strength` - Strength of market direction
- [ ] `market_trend_alignment` - Stock trend vs market trend alignment
- [ ] `market_momentum` - Market momentum indicator
- [ ] `market_volatility_context` - Market volatility context
- [ ] `market_support_resistance` - Market support/resistance levels
- [ ] `market_regime_consistency` - Market regime consistency
- [ ] `market_sector_rotation` - Market sector rotation signal
- [ ] `market_breadth` - Market breadth indicator (advancing/declining)
- [ ] `market_leadership` - Market leadership indicator
- [ ] `market_sentiment_alignment` - Stock sentiment vs market sentiment
- [ ] `market_timing` - Market timing indicator

---

### **CATEGORY 18: ACCURACY-ENHANCING FEATURES - 20 features**

- [ ] `signal_quality_score` - Overall signal quality score
- [ ] `false_positive_risk` - False positive risk indicator
- [ ] `signal_confirmation` - Signal confirmation strength
- [ ] `signal_divergence` - Signal divergence indicator
- [ ] `signal_consistency` - Signal consistency across indicators
- [ ] `signal_strength` - Overall signal strength
- [ ] `signal_timing` - Optimal signal timing
- [ ] `signal_risk_reward` - Signal risk-reward ratio
- [ ] `signal_volume_confirmation` - Volume confirmation for signal
- [ ] `signal_trend_alignment` - Trend alignment for signal
- [ ] `signal_volatility_context` - Volatility context for signal
- [ ] `signal_regime_alignment` - Regime alignment for signal
- [ ] `signal_momentum_strength` - Momentum strength for signal
- [ ] `signal_support_resistance` - Support/resistance context for signal
- [ ] `signal_relative_strength` - Relative strength for signal
- [ ] `signal_multi_timeframe` - Multi-timeframe confirmation
- [ ] `signal_pattern_confirmation` - Pattern confirmation for signal
- [ ] `signal_statistical_significance` - Statistical significance of signal
- [ ] `signal_historical_success` - Historical success rate of similar signals
- [ ] `signal_ensemble_score` - Ensemble score from multiple models

---

## **SUMMARY**

### **Feature Count Breakdown:**
- **Existing Features (Keep):** 56 features ✅
- **Missing Features (Add):** 3 features 🔴
- **Enhanced Variants:** 15 features 📈
- **New Features:** 309 features 🆕
- **TOTAL:** ~383 features

### **Implementation Priority:**
1. **Phase 1:** Add missing 3 features + Enhanced variants (18 features)
2. **Phase 2:** Multi-timeframe, Volatility Regime, Trend Strength, Volume Profile, Support/Resistance (~100 features)
3. **Phase 3:** Advanced Momentum, Price Patterns, Market Regime, Relative Strength, Time-Based (~100 features)
4. **Phase 4:** Feature Interactions, Statistical, Advanced Volatility, Microstructure, Sector (~90 features)
5. **Phase 5:** Predictive Gain Features, Market Direction, Accuracy-Enhancing (~47 features)

### **Status Legend:**
- ✅ = Already implemented (verify in registry)
- 🔴 = Missing (implemented but not in registry)
- 📈 = Enhanced variant (multi-timeframe version)
- 🆕 = New feature to implement
- [ ] = Not yet implemented
- [x] = Implemented and verified

---

**Last Updated:** 2025-01-19  
**Next Review:** After Phase 1 completion
