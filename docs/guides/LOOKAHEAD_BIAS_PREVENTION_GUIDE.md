# Lookahead Bias Prevention Guide

## **CRITICAL: Every feature must be checked for lookahead bias during planning and implementation**

Lookahead bias occurs when a feature uses information that wouldn't be available at the time of prediction. This creates unrealistic performance in backtesting and will fail in live trading.

---

## **🔴 COMMON LOOKAHEAD BIAS PATTERNS TO AVOID**

### **1. Using Current Bar in Rolling Calculations**
❌ **WRONG:**
```python
# Uses current close in the max calculation
resistance = close.rolling(20).max()  # Includes current bar!
distance = (close - resistance) / close  # Lookahead bias!
```

✅ **CORRECT:**
```python
# Use prior period's max, shifted by 1
prior_resistance = close.rolling(20).max().shift(1)  # Excludes current bar
distance = (close - prior_resistance) / close  # No lookahead bias
```

### **2. Using Future Data in Comparisons**
❌ **WRONG:**
```python
# Compares current price to future high
breakout = close > close.rolling(20).max()  # Uses current bar in max!
```

✅ **CORRECT:**
```python
# Compare to prior period's max, shifted
prior_high = close.rolling(20).max().shift(1)  # Prior period only
breakout = (close > prior_high).astype(int)  # No lookahead bias
```

### **3. Using Current Bar in Percentile Ranks**
❌ **WRONG:**
```python
# Includes current value in percentile calculation
percentile = series.rolling(252).apply(lambda x: (x[-1] >= x).sum() / len(x))
```

✅ **CORRECT:**
```python
# Calculate percentile of prior period, then shift
prior_percentile = series.rolling(252).apply(
    lambda x: (x[-2] >= x[:-1]).sum() / len(x[:-1]) if len(x) > 1 else np.nan
).shift(1)
```

### **4. Using Future Returns in Calculations**
❌ **WRONG:**
```python
# Uses future return to calculate feature
future_return = close.shift(-5) / close - 1  # Future data!
feature = some_calculation(future_return)  # Lookahead bias!
```

✅ **CORRECT:**
```python
# Only use past returns
past_return = close / close.shift(5) - 1  # Past data only
feature = some_calculation(past_return)  # No lookahead bias
```

### **5. Using Current Bar in Pattern Detection**
❌ **WRONG:**
```python
# Pattern uses current bar's close in pattern detection
pattern = detect_pattern(high, low, close, open)  # If close is used in detection
```

✅ **CORRECT:**
```python
# Pattern detection should only use OHLC of current bar (that's available)
# But comparisons should use prior bars
pattern = detect_pattern(high, low, close, open)  # OK - current bar OHLC is available
# But if comparing to prior patterns:
prior_pattern = pattern.shift(1)  # Use prior pattern
```

### **6. Multi-Timeframe Resampling Issues**
❌ **WRONG:**
```python
# Resample and forward-fill, but includes current week
weekly = df.resample('W').last()  # Last bar of week includes current day
weekly_features = calculate_features(weekly)
daily_features = weekly_features.reindex(df.index, method='ffill')  # Forward-fill includes future
```

✅ **CORRECT:**
```python
# Resample to end of prior week, then forward-fill
weekly = df.resample('W').last()  # Last bar of week
weekly_features = calculate_features(weekly)
# Shift by 1 to use prior week's data
weekly_features_shifted = weekly_features.shift(1)
daily_features = weekly_features_shifted.reindex(df.index, method='ffill')
```

---

## **✅ SAFE PATTERNS TO USE**

### **1. Rolling Windows (Exclude Current Bar)**
```python
# Always shift rolling calculations by 1
sma = close.rolling(20).mean().shift(1)  # Prior 20-day average
max_high = high.rolling(20).max().shift(1)  # Prior 20-day high
```

### **2. Percent Change (Automatically Safe)**
```python
# pct_change() automatically uses past data
daily_return = close.pct_change(1)  # close_t / close_{t-1} - 1 (safe)
weekly_return = close.pct_change(5)  # close_t / close_{t-5} - 1 (safe)
```

### **3. Shift Operations**
```python
# Use shift() to reference past data
prev_close = close.shift(1)  # Previous day's close
gap = (open - prev_close) / prev_close  # Safe - uses prior close
```

### **4. Current Bar OHLC (Safe)**
```python
# Current bar's OHLC is available at end of day
candle_body = abs(close - open) / (high - low)  # Safe - all current bar
upper_wick = (high - max(open, close)) / (high - low)  # Safe
```

### **5. Expanding Windows (Use with Caution)**
```python
# Expanding windows include current bar - shift by 1
expanding_max = close.expanding().max().shift(1)  # Prior expanding max
```

---

## **📋 LOOKAHEAD BIAS CHECKLIST FOR EACH FEATURE**

Before implementing any feature, verify:

- [ ] **No current bar in rolling calculations** - All rolling windows use `.shift(1)` or exclude current bar
- [ ] **No future data** - No `.shift(-n)` operations (negative shifts)
- [ ] **Breakout/Resistance calculations** - Use prior period's high/low, shifted by 1
- [ ] **Percentile ranks** - Calculate on prior period, exclude current value
- [ ] **Multi-timeframe** - Weekly/monthly features shifted to use prior period
- [ ] **Pattern detection** - Only uses current bar OHLC (available), not future bars
- [ ] **Comparisons** - Always compare current to prior period's values
- [ ] **SPY/Market data** - Aligned by date, no future market data
- [ ] **Volume features** - Use prior period's volume averages
- [ ] **Support/Resistance** - Use prior period's highs/lows, shifted

---

## **🚨 HIGH-RISK FEATURES (Need Extra Attention)**

### **1. Support & Resistance Levels**
**Risk:** Using current bar's high/low in resistance/support calculation

**Safe Implementation:**
```python
# Resistance: Prior period's high, shifted
resistance_20d = high.rolling(20).max().shift(1)  # Prior 20-day high
distance_to_resistance = (resistance_20d - close) / close  # Safe

# Support: Prior period's low, shifted
support_20d = low.rolling(20).min().shift(1)  # Prior 20-day low
distance_to_support = (close - support_20d) / close  # Safe
```

### **2. Breakout Features**
**Risk:** Including current bar in breakout detection

**Safe Implementation:**
```python
# Donchian breakout: Prior period's high, shifted
prior_20d_high = close.rolling(20).max().shift(1)  # Prior period only
breakout = (close > prior_20d_high).astype(int)  # Safe
```

### **3. Percentile Ranks**
**Risk:** Including current value in percentile calculation

**Safe Implementation:**
```python
# Calculate percentile of prior period
def safe_percentile_rank(series, window):
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i < window:
            result.iloc[i] = np.nan
            continue
        # Use prior window (exclude current)
        window_data = series.iloc[i-window:i].values
        current_val = series.iloc[i-1]  # Prior value
        rank = (window_data <= current_val).sum()
        result.iloc[i] = rank / len(window_data) * 100
    return result
```

### **4. Multi-Timeframe Features (Weekly/Monthly)**
**Risk:** Forward-filling includes current week/month's data

**Safe Implementation:**
```python
# Resample to weekly
weekly_df = df.resample('W').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Calculate weekly features
weekly_features = calculate_weekly_features(weekly_df)

# Shift by 1 to use prior week's data
weekly_features_prior = weekly_features.shift(1)

# Forward-fill to daily (now uses prior week's data)
daily_weekly_features = weekly_features_prior.reindex(
    df.index, 
    method='ffill'
)
```

### **5. Volume Profile (Approximations)**
**Risk:** Using current day's volume in profile calculation

**Safe Implementation:**
```python
# Volume profile uses prior period's data
# POC: Price level with most volume in prior period
volume_profile = calculate_volume_profile(
    price=close.shift(1),  # Prior prices
    volume=volume.shift(1),  # Prior volumes
    window=20
)
poc = volume_profile['poc'].shift(1)  # Prior period's POC
```

### **6. VWAP**
**Risk:** Including current bar in VWAP calculation

**Safe Implementation:**
```python
# VWAP: Cumulative (price * volume) / cumulative volume
# But for feature, use prior period's VWAP
vwap = (close * volume).cumsum() / volume.cumsum()
vwap_prior = vwap.shift(1)  # Prior period's VWAP
price_vs_vwap = (close - vwap_prior) / vwap_prior  # Safe
```

### **7. Relative Strength (SPY)**
**Risk:** SPY data not aligned by date, or using future SPY data

**Safe Implementation:**
```python
# Align SPY data by date
spy_data = _load_spy_data()
spy_close = spy_data['Close'].reindex(df.index, method='ffill')

# Calculate returns on same dates
stock_return = close.pct_change(20)  # 20-day return
spy_return = spy_close.pct_change(20)  # SPY 20-day return (same dates)

# Relative strength: stock return - SPY return
relative_strength = stock_return - spy_return  # Safe - same dates
```

### **8. Pattern Detection**
**Risk:** Using future bars to confirm patterns

**Safe Implementation:**
```python
# Pattern detection uses only current bar OHLC (available)
def detect_engulfing(open, high, low, close):
    # Current bar
    body_today = abs(close - open)
    body_yesterday = abs(close.shift(1) - open.shift(1))
    
    # Bullish engulfing: today's body > yesterday's, and today closes above yesterday's high
    bullish = (body_today > body_yesterday) & (close > high.shift(1))
    
    return bullish.astype(int)  # Safe - only uses current and prior bar
```

### **9. Statistical Features (Skewness, Kurtosis)**
**Risk:** Including current value in distribution calculation

**Safe Implementation:**
```python
# Calculate skewness/kurtosis of prior period
def safe_skewness(series, window):
    result = pd.Series(index=series.index, dtype=float)
    for i in range(len(series)):
        if i < window:
            result.iloc[i] = np.nan
            continue
        # Use prior window (exclude current)
        window_data = series.iloc[i-window:i].values
        result.iloc[i] = pd.Series(window_data).skew()
    return result
```

### **10. Trend Reversal Signals**
**Risk:** Using future data to detect reversals

**Safe Implementation:**
```python
# Trend reversal: Detect when trend changes direction
# Use prior period's trend, compare to current
sma20_prior = close.rolling(20).mean().shift(1)  # Prior SMA
sma20_current = close.rolling(20).mean()  # Current SMA (includes current bar - OK for trend)

# But for reversal signal, compare prior trend to current trend
trend_prior = (sma20_prior > sma20_prior.shift(1)).astype(int)  # Prior trend direction
trend_current = (sma20_current > sma20_current.shift(1)).astype(int)  # Current trend

# Reversal: Trend direction changed
reversal = (trend_prior != trend_current).astype(int)  # Safe
```

---

## **🔍 VALIDATION TESTS FOR EACH FEATURE**

After implementing a feature, test for lookahead bias:

### **Test 1: Time Travel Test**
```python
# If you could travel back in time, would you have this information?
# Feature at time t should only use data from t-1 and earlier
feature_t = calculate_feature(df.iloc[:t])  # Should only use data up to t-1
assert feature_t is not np.nan  # Should be calculable
```

### **Test 2: Shift Test**
```python
# Feature should not change if you shift the entire dataset forward
feature_original = calculate_feature(df)
feature_shifted = calculate_feature(df.shift(1))

# Features should shift together (no future data leakage)
assert feature_original.iloc[1] == feature_shifted.iloc[0]  # Should align
```

### **Test 3: Rolling Window Test**
```python
# Rolling calculations should exclude current bar
rolling_max = high.rolling(20).max()
rolling_max_shifted = high.rolling(20).max().shift(1)

# Current bar should not be in shifted version
assert rolling_max.iloc[20] != rolling_max_shifted.iloc[20]  # Different values
assert rolling_max.iloc[20] == rolling_max_shifted.iloc[21]  # Aligned correctly
```

### **Test 4: Breakout Test**
```python
# Breakout should only trigger after price exceeds prior period's high
prior_high = close.rolling(20).max().shift(1)
breakout = (close > prior_high).astype(int)

# On breakout day, close should be > prior_high
breakout_days = breakout[breakout == 1].index
for day in breakout_days:
    assert close.loc[day] > prior_high.loc[day]  # Should be true
    # Prior high should be from prior period only
    prior_period_high = close.loc[day-pd.Timedelta(days=20):day-pd.Timedelta(days=1)].max()
    assert prior_high.loc[day] == prior_period_high  # Should match
```

---

## **📝 FEATURE IMPLEMENTATION TEMPLATE**

Use this template for each feature to ensure no lookahead bias:

```python
def feature_my_feature(df: DataFrame) -> Series:
    """
    Feature description.
    
    LOOKAHEAD BIAS CHECK:
    - [ ] No current bar in rolling calculations (uses .shift(1))
    - [ ] No future data (no .shift(-n))
    - [ ] Breakout/resistance uses prior period's values
    - [ ] Percentile ranks exclude current value
    - [ ] Multi-timeframe features shifted to prior period
    
    Calculation:
    1. [Step 1 - describe, note any shifts]
    2. [Step 2 - describe, note any shifts]
    3. [Final calculation]
    
    Args:
        df: Input DataFrame with OHLCV columns.
    
    Returns:
        Series named 'my_feature' with no lookahead bias.
    """
    close = _get_close_series(df)
    
    # Step 1: Calculate prior period's value (shifted)
    prior_value = close.rolling(20).mean().shift(1)  # Prior 20-day average
    
    # Step 2: Compare current to prior
    feature = (close - prior_value) / prior_value  # Safe - no lookahead bias
    
    feature.name = "my_feature"
    return feature
```

---

## **🎯 PHASE-BY-PHASE LOOKAHEAD BIAS RISKS**

### **Phase 1: Foundation (30 features)**
**Low Risk** - Basic price/returns/MA features
- ✅ Returns: `pct_change()` is automatically safe
- ✅ MAs: Use `.shift(1)` for prior period averages
- ✅ 52-week: Rolling max/min, then shift

### **Phase 2: Momentum (35 features)**
**Low-Medium Risk** - Oscillators generally safe
- ✅ RSI/MACD: Use standard implementations (no lookahead)
- ⚠️ Momentum divergence: Compare current to prior, not future
- ⚠️ Momentum reversal: Use prior trend, not future

### **Phase 3: Trend & Volatility (50 features)**
**Medium Risk** - Trend detection needs care
- ⚠️ Trend reversal: Use prior trend direction
- ⚠️ Trend exhaustion: Compare to prior period's trend
- ✅ Volatility: Rolling calculations, shift by 1

### **Phase 4: Volume & S/R (45 features)**
**High Risk** - Support/resistance critical
- 🔴 Support/Resistance: MUST use prior period's high/low, shifted
- 🔴 Breakouts: MUST use prior period's high, shifted
- ⚠️ Volume profile: Use prior period's data
- ⚠️ VWAP: Use prior period's VWAP

### **Phase 5: Multi-timeframe (50 features)**
**High Risk** - Resampling can introduce lookahead
- 🔴 Weekly/Monthly: MUST shift by 1 after resampling
- 🔴 Forward-fill: Only after shifting to prior period

### **Phase 6: Market Context (45 features)**
**Medium Risk** - SPY alignment critical
- ⚠️ SPY data: Must align by date, no future SPY data
- ⚠️ Market regime: Use prior period's market state

### **Phase 7: Statistical (50 features)**
**Medium Risk** - Distribution calculations
- ⚠️ Skewness/Kurtosis: Exclude current value
- ⚠️ Autocorrelation: Use prior period's data

### **Phase 8: Interactions (20 features)**
**Low Risk** - Multiplications of safe features
- ✅ Safe if base features are safe

### **Phase 9: Predictive (47 features)**
**High Risk** - Target-specific features
- 🔴 Historical gain probability: Use prior period's statistics
- 🔴 Gain momentum: Compare to prior target progress

### **Phase 10: RS & Sector (22 features)**
**Medium Risk** - Data alignment
- ⚠️ Relative strength: Align SPY dates correctly
- ⚠️ Sector: Align sector ETF dates correctly

---

## **✅ FINAL VALIDATION CHECKLIST**

Before marking any feature as complete:

- [ ] **Code Review:** Reviewed for `.shift(-n)` (negative shifts)
- [ ] **Code Review:** Reviewed for rolling windows without `.shift(1)`
- [ ] **Code Review:** Reviewed for current bar in max/min calculations
- [ ] **Test:** Time travel test passes
- [ ] **Test:** Shift test passes
- [ ] **Test:** Breakout test passes (if applicable)
- [ ] **Documentation:** Docstring includes lookahead bias check notes
- [ ] **Validation:** Feature calculated on sample data, values make sense
- [ ] **Edge Cases:** Handles NaN values correctly
- [ ] **Edge Cases:** Handles insufficient data (min_periods)

---

## **🚫 ABSOLUTE RULES (Never Violate)**

1. **NEVER use `.shift(-n)`** (negative shift = future data)
2. **NEVER include current bar in rolling max/min for breakouts/resistance**
3. **NEVER forward-fill multi-timeframe features without shifting first**
4. **NEVER use future returns in any calculation**
5. **NEVER align data by index position - always by date**
6. **NEVER use current value in percentile rank calculation**
7. **ALWAYS shift rolling calculations by 1 for comparisons**
8. **ALWAYS use prior period's high/low for support/resistance**
9. **ALWAYS shift multi-timeframe features by 1 period**
10. **ALWAYS validate with time travel test**

---

**Remember: If you're not 100% sure, shift it by 1. It's better to be conservative than to have lookahead bias.**
