# Technical Calculations Optimization Brainstorm
## Current Performance: 12 minutes per ticker (371 features)

**Goal**: Reduce computation time while maintaining data integrity and precision

**Date**: 2026-01-24

---

## 🔍 Current Bottleneck Analysis

### 1. **Redundant Intermediate Calculations** (Estimated: 40-50% of time)

#### Most Common Redundant Operations:

**Moving Averages (computed 50-100+ times each):**
- `SMA20`: `close.rolling(20).mean()` - appears in ~50+ features
- `SMA50`: `close.rolling(50).mean()` - appears in ~40+ features  
- `SMA200`: `close.rolling(200).mean()` - appears in ~30+ features
- `EMA20`: `close.ewm(span=20).mean()` - appears in ~30+ features
- `EMA50`: `close.ewm(span=50).mean()` - appears in ~25+ features
- `EMA200`: `close.ewm(span=200).mean()` - appears in ~20+ features
- `EMA12`, `EMA26`: Used for MACD family (~10+ features)

**RSI Calculations (computed 30+ times each):**
- `RSI14`: Complex calculation with gain/loss averaging - appears in ~30+ features
- `RSI7`: Similar calculation - appears in ~15+ features
- `RSI21`: Similar calculation - appears in ~10+ features

**Volatility Measures (computed 25+ times each):**
- `ATR14`: True Range calculation + rolling mean - appears in ~25+ features
- `volatility_5d`: `returns.rolling(5).std()` - appears in ~20+ features
- `volatility_21d`: `returns.rolling(21).std()` - appears in ~25+ features

**Price Series Extraction (computed 371 times):**
- `_get_close_series(df)` - called in EVERY feature function
- `_get_high_series(df)` - called in ~150+ features
- `_get_low_series(df)` - called in ~150+ features
- `_get_volume_series(df)` - called in ~100+ features

**Returns Calculations (computed 50+ times):**
- `close.pct_change()` - daily returns - appears in ~50+ features
- `np.log(close / close.shift(1))` - log returns - appears in ~30+ features

**52-Week Calculations (computed 10+ times):**
- `close.rolling(252).max()` - 52-week high - appears in ~10+ features
- `close.rolling(252).min()` - 52-week low - appears in ~10+ features

**True Range (computed 20+ times):**
- `max(high - low, abs(high - prev_close), abs(low - prev_close))` - appears in ~20+ features

### 2. **Expensive Resampling Operations** (Estimated: 10-15% of time)

**Weekly Resampling:**
- Multiple features resample to weekly: `close.resample('W-FRI').last()`
- Then compute indicators, then reindex back to daily
- Examples: `weekly_sma_5w`, `weekly_ema_5w`, `weekly_rsi`, etc.
- **Problem**: Resampling is expensive, and many features do it independently

**Monthly Resampling:**
- Similar pattern for monthly features: `close.resample('ME').last()`
- Examples: `monthly_sma_3m`, `monthly_sma_6m`, `monthly_rsi`
- **Problem**: Same resampling done multiple times

**Optimization Opportunity**: Pre-compute resampled series once, reuse for all weekly/monthly features

### 3. **Nested Feature Calls** (Estimated: 5-10% of time)

Some features call other feature functions:
- `feature_top_features_ensemble()` calls:
  - `feature_volatility_forecast()`
  - `feature_volatility_21d()`
  - `feature_volume_imbalance()`
  - `feature_gain_probability_score()`
  - `feature_momentum_rank()`
- **Problem**: These features are computed twice (once standalone, once for ensemble)
- **Solution**: Cache feature results or pass pre-computed features

### 4. **Python Loops in Hot Paths** (Estimated: 5-10% of time)

Some features use Python loops instead of vectorized operations:
- `feature_volatility_risk_premium()`: Has a for loop for EWMA calculation
- `feature_hurst_exponent()`: Complex rolling calculations with loops
- `feature_fractal_dimension_index()`: Window-based calculations
- **Problem**: Python loops are slow compared to vectorized pandas/numpy

### 5. **Validation Overhead** (Estimated: 3-5% of time)

- Each feature validated individually (371 validation calls)
- Each validation checks: NaN count, infinity count, constant values, variance
- **Problem**: Could batch validate all features at once

### 6. **Memory Allocation** (Estimated: 2-3% of time)

- 371 Series objects created individually
- Then concatenated into DataFrame
- **Problem**: Multiple memory allocations, could be more efficient

---

## 💡 Optimization Strategies (Brainstorming)

### **STRATEGY 1: Shared Intermediate Cache** ⭐⭐⭐⭐⭐
**Impact**: 40-50% speedup | **Effort**: Medium | **Risk**: Low | **Priority**: HIGHEST

**Concept**: Pre-compute all commonly used intermediates once per ticker, pass to features

**Implementation Approach**:
```python
def compute_shared_intermediates(df: DataFrame) -> Dict[str, Series]:
    """Pre-compute all commonly used intermediates once"""
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    # Pre-compute returns (used everywhere)
    returns_1d = close.pct_change()
    log_returns_1d = np.log(close / close.shift(1))
    
    intermediates = {
        # Base series (avoid repeated column lookups)
        '_close': close,
        '_high': high,
        '_low': low,
        '_volume': volume,
        '_returns_1d': returns_1d,
        '_log_returns_1d': log_returns_1d,
        
        # Moving Averages (most common - computed 50-100+ times)
        '_sma20': close.rolling(20, min_periods=1).mean(),
        '_sma50': close.rolling(50, min_periods=1).mean(),
        '_sma200': close.rolling(200, min_periods=1).mean(),
        '_ema20': close.ewm(span=20, adjust=False).mean(),
        '_ema50': close.ewm(span=50, adjust=False).mean(),
        '_ema200': close.ewm(span=200, adjust=False).mean(),
        '_ema12': close.ewm(span=12, adjust=False).mean(),
        '_ema26': close.ewm(span=26, adjust=False).mean(),
        
        # Volatility (very common)
        '_volatility_5d': returns_1d.rolling(5, min_periods=1).std(),
        '_volatility_21d': returns_1d.rolling(21, min_periods=1).std(),
        
        # True Range & ATR
        '_tr': _compute_true_range(high, low, close),
        '_atr14': _compute_atr(high, low, close, 14),
        
        # RSI (expensive to compute)
        '_rsi7': _compute_rsi(close, 7),
        '_rsi14': _compute_rsi(close, 14),
        '_rsi21': _compute_rsi(close, 21),
        
        # 52-week extremes
        '_high_52w': close.rolling(252, min_periods=1).max(),
        '_low_52w': close.rolling(252, min_periods=1).min(),
        
        # Volume averages
        '_volume_avg_20d': volume.rolling(20, min_periods=1).mean(),
        
        # Resampled series (expensive operations)
        '_weekly_close': close.resample('W-FRI').last(),
        '_monthly_close': close.resample('ME').last(),
    }
    return intermediates

# Modify feature functions to accept optional intermediates
def feature_sma20_ratio(df: DataFrame, intermediates: Optional[Dict] = None) -> Series:
    if intermediates:
        close = intermediates['_close']
        sma20 = intermediates['_sma20']
    else:
        close = _get_close_series(df)
        sma20 = close.rolling(20, min_periods=1).mean()
    
    ratio = close / sma20
    ratio.name = "sma20_ratio"
    return ratio
```

**Migration Strategy**:
1. Add `intermediates` parameter to all feature functions (optional, backward compatible)
2. Update high-impact features first (SMAs, RSIs, volatility - ~100 features)
3. Gradually migrate remaining features
4. Remove redundant calculations from migrated features

**Data Integrity Considerations**:
- ✅ Same calculations, just cached - no precision loss
- ✅ Can validate intermediates once instead of 50+ times
- ✅ Easier to debug (single source of truth)

**Precision Considerations**:
- ✅ Using same pandas operations, just cached
- ✅ Floating point precision identical
- ✅ Can add validation to ensure intermediates match original calculations

---

### **STRATEGY 2: Batch Resampling** ⭐⭐⭐⭐
**Impact**: 10-15% speedup | **Effort**: Low-Medium | **Risk**: Low | **Priority**: HIGH

**Concept**: Pre-compute resampled series once, reuse for all weekly/monthly features

**Current Problem**:
- `feature_weekly_sma_5w()`: Resamples to weekly, computes SMA, reindexes
- `feature_weekly_ema_5w()`: Resamples to weekly AGAIN, computes EMA, reindexes
- `feature_weekly_rsi()`: Resamples to weekly AGAIN, computes RSI, reindexes
- Same resampling done 10+ times independently

**Solution**:
```python
# In shared_intermediates:
weekly_close = close.resample('W-FRI').last()
monthly_close = close.resample('ME').last()

# Features use pre-resampled data:
def feature_weekly_sma_5w(df: DataFrame, intermediates: Optional[Dict] = None) -> Series:
    if intermediates:
        weekly_close = intermediates['_weekly_close']
        close = intermediates['_close']
    else:
        close = _get_close_series(df)
        weekly_close = close.resample('W-FRI').last()
    
    weekly_sma = weekly_close.rolling(5, min_periods=1).mean()
    daily_sma = weekly_sma.reindex(close.index, method='ffill')
    return daily_sma / close
```

**Data Integrity**: ✅ Same resampling logic, just cached

---

### **STRATEGY 3: Vectorized Feature Groups** ⭐⭐⭐
**Impact**: 10-15% speedup | **Effort**: Medium | **Risk**: Medium | **Priority**: MEDIUM

**Concept**: Compute similar features together in batches

**Examples**:

**SMA Ratio Family** (8 features):
```python
def compute_sma_ratios_vectorized(close, intermediates):
    """Compute all SMA ratios at once"""
    sma20 = intermediates['_sma20']
    sma50 = intermediates['_sma50']
    sma200 = intermediates['_sma200']
    
    return {
        'sma20_ratio': close / sma20,
        'sma50_ratio': close / sma50,
        'sma200_ratio': close / sma200,
        'sma20_sma50_ratio': sma20 / sma50,
        'sma50_sma200_ratio': sma50 / sma200,
        # ... etc
    }
```

**RSI Family** (4 features):
```python
def compute_rsi_family_vectorized(close, intermediates):
    """Compute all RSI features at once"""
    rsi7 = intermediates['_rsi7']
    rsi14 = intermediates['_rsi14']
    rsi21 = intermediates['_rsi21']
    
    return {
        'rsi7': rsi7,
        'rsi14': rsi14,
        'rsi21': rsi21,
        'rsi_momentum': rsi14 - rsi14.shift(5),
    }
```

**Volatility Family** (10+ features):
```python
def compute_volatility_family_vectorized(close, intermediates):
    """Compute all volatility features at once"""
    returns = intermediates['_returns_1d']
    vol_5d = intermediates['_volatility_5d']
    vol_21d = intermediates['_volatility_21d']
    
    return {
        'volatility_5d': vol_5d,
        'volatility_21d': vol_21d,
        'volatility_regime': (vol_21d > vol_21d.rolling(252).quantile(0.75)).astype(float),
        # ... etc
    }
```

**Data Integrity**: ✅ Same calculations, just grouped

---

### **STRATEGY 4: Replace Python Loops with Vectorized Operations** ⭐⭐⭐
**Impact**: 5-10% speedup (for affected features) | **Effort**: Medium | **Risk**: Medium | **Priority**: MEDIUM

**Problem Features**:

**1. `feature_volatility_risk_premium()`** - Has Python loop:
```python
# Current (slow):
for i in range(1, len(df)):
    forecast_var.iloc[i] = alpha * forecast_var.iloc[i-1] + (1 - alpha) * squared_returns.iloc[i]

# Optimized (vectorized):
forecast_var = squared_returns.ewm(alpha=alpha, adjust=False).mean()
```

**2. `feature_hurst_exponent()`** - Complex rolling calculations
- Could potentially use numba JIT or vectorize better

**3. `feature_fractal_dimension_index()`** - Window-based calculations
- Could use rolling apply with optimized functions

**Data Integrity**: ⚠️ Need careful testing to ensure vectorized version matches loop version

**Precision**: ⚠️ EWMA with `adjust=False` should match, but need validation

---

### **STRATEGY 5: Batch Validation** ⭐⭐
**Impact**: 3-5% speedup | **Effort**: Low | **Risk**: Very Low | **Priority**: LOW

**Concept**: Validate all features at once instead of individually

**Current**:
```python
for name, func in enabled_features.items():
    feature_series = func(df)
    is_valid, issues = validate_feature(feature_series, name)  # 371 calls
```

**Optimized**:
```python
# Collect all features first
feature_dict = {}
for name, func in enabled_features.items():
    feature_dict[name] = func(df)

# Batch validate
feature_df = pd.DataFrame(feature_dict)
validation_issues = validate_features_batch(feature_df)  # 1 call
```

**Data Integrity**: ✅ Same validation logic, just batched

---

### **STRATEGY 6: Feature Dependency Graph & Parallel Groups** ⭐⭐⭐
**Impact**: 15-25% speedup | **Effort**: High | **Risk**: Medium | **Priority**: MEDIUM (if needed)

**Concept**: Group independent features, compute in parallel

**Dependency Levels**:
- **Level 0**: Base features (price, returns) - no dependencies
- **Level 1**: Simple indicators (SMAs, EMAs) - depends on Level 0
- **Level 2**: Composite indicators (RSI, MACD) - depends on Level 1
- **Level 3**: Advanced features (ensembles, complex indicators) - depends on Level 2

**Implementation**:
```python
# Compute Level 0 in parallel
level_0_features = ['price', 'price_log', 'daily_return', 'gap_pct']
results_0 = Parallel(n_jobs=-1)(
    delayed(compute_feature)(name, func, df, intermediates)
    for name, func in level_0_features.items()
)

# Then Level 1 (depends on Level 0)
level_1_features = ['sma20_ratio', 'sma50_ratio', ...]
results_1 = Parallel(n_jobs=-1)(
    delayed(compute_feature)(name, func, df, intermediates, results_0)
    for name, func in level_1_features.items()
)
```

**Data Integrity**: ✅ Same calculations, just parallelized

**Considerations**:
- Overhead of parallelization might not be worth it for small groups
- Need to ensure intermediates are thread-safe
- Memory usage increases with parallel workers

---

### **STRATEGY 7: Numba JIT for Hot Paths** ⭐⭐
**Impact**: 20-40% speedup for affected features | **Effort**: High | **Risk**: High | **Priority**: LOW (only if needed)

**Concept**: JIT-compile numeric-heavy functions

**Candidates**:
- `_compute_rsi()` - Complex gain/loss averaging
- `_compute_hurst_exponent()` - Complex rolling calculations
- `_compute_fractal_dimension()` - Window-based calculations
- Custom rolling correlation functions

**Implementation**:
```python
from numba import jit

@jit(nopython=True)
def compute_rsi_numba(close_array, period):
    # Rewrite in numba-compatible code
    ...

def feature_rsi14(df: DataFrame, intermediates: Optional[Dict] = None) -> Series:
    if intermediates:
        close = intermediates['_close']
    else:
        close = _get_close_series(df)
    
    rsi_values = compute_rsi_numba(close.values, 14)
    return pd.Series(rsi_values, index=close.index, name='rsi14')
```

**Data Integrity**: ⚠️ Need careful validation - numba can have different floating point behavior

**Precision**: ⚠️ Numba uses different math libraries, might have slight precision differences

---

### **STRATEGY 8: Conditional Feature Computation** ⭐
**Impact**: Variable (depends on feature usage) | **Effort**: Low | **Risk**: Low | **Priority**: LOW

**Concept**: Skip expensive features if they're not used downstream

**Example**: If `feature_top_features_ensemble()` is enabled, but individual features it uses are not needed elsewhere, could skip computing them standalone

**Data Integrity**: ✅ No impact (features still computed when needed)

---

### **STRATEGY 9: Memory-Efficient Concatenation** ⭐
**Impact**: 2-3% speedup | **Effort**: Low | **Risk**: Very Low | **Priority**: LOW

**Concept**: Build DataFrame more efficiently

**Current**:
```python
feature_dict = {}
for name, func in enabled_features.items():
    feature_dict[name] = func(df)  # 371 Series objects
feature_df = pd.DataFrame(feature_dict)  # Large allocation
```

**Optimized**:
```python
feature_df = pd.DataFrame(index=df.index)
for name, func in enabled_features.items():
    feature_df[name] = func(df)  # Incremental allocation
```

**Data Integrity**: ✅ Same result

---

### **STRATEGY 10: Disable Debug Logging During Computation** ⭐
**Impact**: 2-3% speedup | **Effort**: Very Low | **Risk**: Very Low | **Priority**: LOW

**Concept**: Set log level to INFO during feature computation

**Current**: `logger.debug(f"{name} computed")` called 371 times per ticker

**Optimized**: Only log errors and warnings during computation

**Data Integrity**: ✅ No impact

---

## 📊 Expected Combined Impact

### **Phase 1: Quick Wins** (1-2 days implementation)
1. ✅ **Shared Intermediate Cache** (Strategy 1): 40-50% speedup
2. ✅ **Batch Resampling** (Strategy 2): 10-15% speedup
3. ✅ **Batch Validation** (Strategy 5): 3-5% speedup
4. ✅ **Disable Debug Logging** (Strategy 10): 2-3% speedup

**Combined Phase 1**: ~50-70% speedup
- **Current**: 12 minutes per ticker
- **After Phase 1**: **3.6-6 minutes per ticker**

### **Phase 2: Medium Impact** (3-5 days implementation)
5. ✅ **Vectorized Feature Groups** (Strategy 3): 10-15% additional speedup
6. ✅ **Replace Python Loops** (Strategy 4): 5-10% additional speedup
7. ✅ **Memory-Efficient Concatenation** (Strategy 9): 2-3% additional speedup

**Combined Phase 2**: Additional 15-25% speedup
- **After Phase 2**: **2.7-4.5 minutes per ticker**

### **Phase 3: Advanced** (1-2 weeks, if needed)
8. ✅ **Feature Dependency Graph** (Strategy 6): 15-25% additional speedup (if parallelization overhead is worth it)
9. ✅ **Numba JIT** (Strategy 7): 20-40% for affected features (if precision acceptable)

**Combined Phase 3**: Additional 15-30% speedup
- **After Phase 3**: **1.9-3.2 minutes per ticker**

---

## 🛡️ Data Integrity & Precision Safeguards

### **For All Strategies**:

1. **Validation Testing**:
   - Compare optimized outputs vs original outputs
   - Use `np.allclose()` with appropriate tolerance (e.g., `rtol=1e-10, atol=1e-10`)
   - Check for NaN/infinity differences
   - Verify edge cases (empty data, single row, etc.)

2. **Intermediate Validation**:
   - Validate shared intermediates match original calculations
   - Add unit tests for intermediate computation
   - Log warnings if intermediates don't match

3. **Backward Compatibility**:
   - Make optimizations optional (feature flag)
   - Keep original code paths available
   - Gradual migration (migrate features in batches)

4. **Precision Monitoring**:
   - Track floating point differences
   - Ensure same pandas operations used (just cached)
   - Document any precision trade-offs

5. **Regression Testing**:
   - Run full feature suite before/after
   - Compare model outputs (if features used in training)
   - Monitor for any downstream issues

---

## 🎯 Recommended Implementation Order

### **Week 1: Phase 1 (Target: 50-70% speedup)**
1. **Day 1-2**: Implement Shared Intermediate Cache (Strategy 1)
   - Create `compute_shared_intermediates()` function
   - Add `intermediates` parameter to 50 highest-impact features
   - Test and validate
   
2. **Day 2-3**: Implement Batch Resampling (Strategy 2)
   - Add resampled series to intermediates
   - Update weekly/monthly features to use cached resampled data
   
3. **Day 3**: Implement Batch Validation (Strategy 5)
   - Create `validate_features_batch()` function
   - Update `apply_features()` to use batch validation
   
4. **Day 3**: Disable Debug Logging (Strategy 10)
   - Set log level during computation

**Expected Result**: 12 minutes → **3.6-6 minutes** (50-70% faster)

### **Week 2: Phase 2 (Target: Additional 15-25% speedup)**
5. **Day 4-5**: Vectorized Feature Groups (Strategy 3)
   - Group SMA ratios, RSI family, volatility family
   - Implement vectorized computation functions
   
6. **Day 5-6**: Replace Python Loops (Strategy 4)
   - Identify features with loops
   - Replace with vectorized operations
   - Validate precision

**Expected Result**: 3.6-6 minutes → **2.7-4.5 minutes** (additional 15-25% faster)

### **Week 3+: Phase 3 (If Needed)**
7. **If still not fast enough**: Consider parallelization (Strategy 6)
8. **If precision acceptable**: Consider Numba JIT (Strategy 7)

---

## ❓ Questions to Answer Before Implementation

1. **What's the actual bottleneck?**
   - Profile first! Time each feature function
   - Identify slowest 20 features
   - Measure I/O vs computation time

2. **How many redundant calculations?**
   - Count: How many times is SMA20 computed?
   - Count: How many times is RSI14 computed?
   - This will validate our estimates

3. **What's the typical use case?**
   - Full rebuilds (all tickers)?
   - Incremental updates (new tickers only)?
   - This affects which optimizations are most valuable

4. **What's the target time?**
   - < 5 minutes per ticker?
   - < 2 minutes per ticker?
   - This determines how aggressive to be

5. **Precision requirements?**
   - Can we tolerate tiny floating point differences?
   - Or must be bit-for-bit identical?
   - This affects which strategies are acceptable

---

## 📝 Notes

- **Data Integrity**: All optimizations maintain same calculations, just cached/reorganized
- **Precision**: Using same pandas operations, floating point precision should be identical
- **Testing**: Each optimization tested independently before combining
- **Rollback**: Each optimization can be disabled via feature flag
- **Gradual Migration**: Migrate features in batches, not all at once

---

## 🚀 Next Steps

1. **Profile current implementation** (1-2 hours)
   - Time each feature
   - Identify slowest features
   - Count redundant calculations
   - Measure I/O vs computation

2. **Start with Strategy 1** (Shared Intermediate Cache)
   - Highest impact, medium effort
   - Expected 40-50% speedup
   - Low risk to data integrity

3. **Validate and measure**
   - Compare outputs
   - Measure actual speedup
   - Fix any issues

4. **Continue with Phase 1**
   - Batch resampling
   - Batch validation
   - Disable debug logging

5. **Evaluate and decide on Phase 2**
   - If Phase 1 achieves target, may not need Phase 2
   - If not, proceed with vectorization and loop replacement
