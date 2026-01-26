# Performance Debugging & Optimization Brainstorm
## Current Performance: 15 minutes per ticker (WORSE than 12 minutes!)

**Date**: 2026-01-24

**Problem**: After implementing Strategy 1 (Shared Intermediate Cache), performance got WORSE (15 min vs 12 min original)

---

## 📊 PROFILING RESULTS (AAPL, 4544 rows, 61 features)

**Total Time**: 9.726s
- Data loading: 0.082s (0.8%)
- **Intermediates: 0.026s (0.3%)** ✅ NOT THE PROBLEM!
- **Feature computation: 9.618s (98.9%)** ⚠️ THIS IS THE BOTTLENECK
- Signature inspection: 0.001s (0.0%)

**Top 5 Slowest Features** (84% of total time):
1. `hurst_exponent`: 3.333s (34.7%)
2. `trend_residual`: 1.894s (19.7%)
3. `aroon_oscillator`: 1.126s (11.7%)
4. `mkt_spy_sma200_slope`: 0.981s (10.2%)
5. `fractal_dimension_index`: 0.743s (7.7%)

**Key Finding**: These slowest features DON'T use intermediates! They're doing expensive computations that can't be cached.

**Features using intermediates**: 13/61 (21.3%)

---

## 🔍 Potential Causes of Slowdown

### 1. **Intermediates Computation Overhead** ✅ RESOLVED - NOT THE PROBLEM
**Hypothesis**: Computing all intermediates upfront might be slower than computing them on-demand

**Possible Issues**:
- Computing 27 intermediates upfront might be expensive
- Some intermediates might not be used by most features
- Resampling operations (`_weekly_close`, `_monthly_close`) are expensive and might not be needed
- Computing all RSIs, SMAs, EMAs upfront might be wasteful if only a few are used

**Investigation Needed**:
- Profile: How long does `compute_shared_intermediates()` take?
- Check: Which intermediates are actually being used?
- Measure: Time to compute intermediates vs time saved by reusing them

**Potential Fixes**:
- **Lazy computation**: Only compute intermediates when first needed
- **Selective computation**: Only compute intermediates for features that are actually enabled
- **Remove unused intermediates**: Don't compute resampled series if no weekly/monthly features are enabled
- **Cache intermediates**: Compute once, reuse across multiple feature calls

---

### 2. **Function Signature Inspection Overhead** ⚠️ MEDIUM PRIORITY
**Hypothesis**: Checking function signatures for every feature call adds overhead

**Current Code**:
```python
for name, func in enabled_features.items():
    sig = inspect.signature(func)  # Called 371 times!
    params = sig.parameters
    if 'spy_data' in params:
        kwargs['spy_data'] = spy_data
    if 'intermediates' in params:
        kwargs['intermediates'] = intermediates
```

**Problem**: `inspect.signature()` is called 371 times per ticker, which has overhead

**Potential Fixes**:
- **Cache signatures**: Compute signatures once at startup, not per ticker
- **Use function attributes**: Store signature info as function attributes
- **Registry metadata**: Store parameter info in feature registry

---

### 3. **Dictionary Lookup Overhead** ⚠️ LOW PRIORITY
**Hypothesis**: Looking up intermediates in dictionary might add overhead

**Current Code**:
```python
if intermediates:
    close = intermediates['_close']  # Dictionary lookup
    sma20 = intermediates['_sma20']  # Dictionary lookup
```

**Potential Fixes**:
- **Direct attribute access**: Use object attributes instead of dict (but requires refactoring)
- **Unpack intermediates**: Pass as separate arguments (but breaks backward compatibility)
- **Keep as-is**: Dictionary lookup is very fast, probably not the issue

---

### 4. **Conditional Logic Overhead** ⚠️ LOW PRIORITY
**Hypothesis**: `if intermediates:` checks in every feature function add overhead

**Current Pattern**:
```python
def feature_sma20_ratio(df: DataFrame, intermediates: Optional[dict] = None) -> Series:
    if intermediates:
        close = intermediates['_close']
        sma20 = intermediates['_sma20']
    else:
        close = _get_close_series(df)
        sma20 = close.rolling(window=20, min_periods=1).mean()
```

**Potential Fixes**:
- **Always pass intermediates**: Remove the `if intermediates:` check, always compute intermediates
- **Two separate functions**: One optimized, one original (but breaks backward compatibility)
- **Keep as-is**: Conditional check is very fast, probably not the issue

---

### 5. **Resampling Operations Are Expensive** ⚠️ HIGH PRIORITY
**Hypothesis**: Computing `_weekly_close` and `_monthly_close` in intermediates is expensive and might not be needed

**Current Code**:
```python
'_weekly_close': close.resample('W-FRI').last() if isinstance(close.index, pd.DatetimeIndex) else close,
'_monthly_close': close.resample('ME').last() if isinstance(close.index, pd.DatetimeIndex) else close,
```

**Problem**: Resampling is expensive, and if no weekly/monthly features are enabled, this is wasted computation

**Potential Fixes**:
- **Lazy resampling**: Only compute when a weekly/monthly feature is called
- **Check if needed**: Only compute if weekly/monthly features are in enabled_features
- **Remove from intermediates**: Compute on-demand in features that need them

---

### 6. **Slow Features Don't Use Intermediates** ⚠️⚠️⚠️ CRITICAL PRIORITY
**Hypothesis**: The slowest features (84% of time) don't use intermediates, so optimization doesn't help them

**Profiling Results**:
- `hurst_exponent`: 3.333s - doesn't use intermediates
- `trend_residual`: 1.894s - doesn't use intermediates
- `aroon_oscillator`: 1.126s - doesn't use intermediates
- `mkt_spy_sma200_slope`: 0.981s - doesn't use intermediates
- `fractal_dimension_index`: 0.743s - doesn't use intermediates

**Problem**: These features are inherently slow (complex rolling calculations, loops, etc.) and can't benefit from cached intermediates

**Potential Fixes**:
- **Optimize these specific features**: Use numba, vectorization, or algorithmic improvements
- **Profile each slow feature**: Identify what makes them slow
- **Consider disabling slow features**: If they're not important, disable them
- **Parallelize slow features**: Compute them in parallel if independent

### 7. **Not Enough Features Using Intermediates** ⚠️ MEDIUM PRIORITY
**Hypothesis**: Only 13 features use intermediates, so the benefit is minimal

**Current Status**:
- 13 features migrated to use intermediates
- 48 features still don't use intermediates
- Most features still compute everything from scratch

**Impact**: If only 13/61 features benefit, and intermediates computation takes time, net result could be slower

**Potential Fixes**:
- **Migrate more features**: Prioritize high-impact features (those called most often)
- **Profile to find slowest features**: Focus optimization on features that take the most time
- **Batch migrate**: Migrate all SMA/EMA/RSI/volatility features at once

---

### 7. **Intermediate Computation Redundancy** ⚠️ MEDIUM PRIORITY
**Hypothesis**: Some intermediates might be computing things that aren't needed

**Example**: Computing `_rsi7`, `_rsi14`, `_rsi21` even if only `rsi14` is used

**Potential Fixes**:
- **Selective computation**: Only compute intermediates for enabled features
- **Lazy computation**: Compute RSI only when first RSI feature is called
- **Feature dependency analysis**: Compute only what's needed

---

## 🚀 Optimization Strategies

### **STRATEGY A: Profile First!** ⭐⭐⭐⭐⭐
**Priority**: CRITICAL - Do this first!

**Action**: Add timing/profiling to identify actual bottlenecks

**Implementation**:
```python
import time
import cProfile
import pstats

# In apply_features():
intermediate_start = time.perf_counter()
intermediates = compute_shared_intermediates(df)
intermediate_time = time.perf_counter() - intermediate_start
logger.info(f"Intermediates computation: {intermediate_time:.2f}s")

# Time each feature
feature_times = {}
for name, func in enabled_features.items():
    start = time.perf_counter()
    # ... compute feature ...
    elapsed = time.perf_counter() - start
    feature_times[name] = elapsed

# Report slowest features
slowest = sorted(feature_times.items(), key=lambda x: x[1], reverse=True)[:20]
logger.info("Slowest 20 features:")
for name, elapsed in slowest:
    logger.info(f"  {name}: {elapsed:.3f}s")
```

**Expected Outcome**: Identify where time is actually being spent

---

### **STRATEGY B: Lazy Intermediate Computation** ⭐⭐⭐⭐
**Priority**: HIGH

**Concept**: Don't compute all intermediates upfront. Compute them on-demand when first needed.

**Implementation**:
```python
class LazyIntermediates:
    def __init__(self, df):
        self.df = df
        self._cache = {}
    
    def __getitem__(self, key):
        if key not in self._cache:
            # Compute on-demand
            if key == '_close':
                self._cache[key] = _get_close_series(self.df)
            elif key == '_sma20':
                self._cache[key] = self['_close'].rolling(20, min_periods=1).mean()
            # ... etc
        return self._cache[key]

# In apply_features():
intermediates = LazyIntermediates(df)  # No computation yet!
```

**Benefits**:
- Only computes what's actually used
- No upfront cost
- Still caches for reuse

**Drawbacks**:
- Slightly more complex
- First access to each intermediate has computation cost

---

### **STRATEGY C: Selective Intermediate Computation** ⭐⭐⭐⭐
**Priority**: HIGH

**Concept**: Only compute intermediates that are needed by enabled features

**Implementation**:
```python
def compute_shared_intermediates(df: DataFrame, needed_keys: set = None) -> dict:
    """Only compute intermediates that are needed"""
    if needed_keys is None:
        needed_keys = set()  # Compute all if not specified
    
    intermediates = {}
    
    # Always need base series
    close = _get_close_series(df)
    intermediates['_close'] = close
    
    # Only compute if needed
    if '_sma20' in needed_keys or not needed_keys:
        intermediates['_sma20'] = close.rolling(20, min_periods=1).mean()
    
    # ... etc
    return intermediates

# In apply_features():
# Analyze which intermediates are needed
needed_intermediates = analyze_intermediate_dependencies(enabled_features)
intermediates = compute_shared_intermediates(df, needed_intermediates)
```

**Benefits**:
- Skips unnecessary computations
- Faster if only a few features use intermediates

---

### **STRATEGY D: Cache Function Signatures** ⭐⭐⭐
**Priority**: MEDIUM

**Concept**: Compute function signatures once, not per ticker

**Implementation**:
```python
# At module level or in registry:
FEATURE_SIGNATURES = {}
for name, func in FEATURE_REGISTRY.items():
    FEATURE_SIGNATURES[name] = {
        'has_spy_data': 'spy_data' in inspect.signature(func).parameters,
        'has_intermediates': 'intermediates' in inspect.signature(func).parameters,
    }

# In apply_features():
for name, func in enabled_features.items():
    sig_info = FEATURE_SIGNATURES.get(name, {})
    if sig_info.get('has_intermediates'):
        kwargs['intermediates'] = intermediates
```

**Benefits**:
- Eliminates 371 `inspect.signature()` calls per ticker
- Small but measurable speedup

---

### **STRATEGY E: Remove Expensive Unused Intermediates** ⭐⭐⭐⭐
**Priority**: HIGH

**Concept**: Don't compute resampled series unless needed

**Implementation**:
```python
def compute_shared_intermediates(df: DataFrame, compute_resampled: bool = False) -> dict:
    intermediates = {
        # ... base intermediates ...
    }
    
    # Only compute resampled if needed
    if compute_resampled:
        close = intermediates['_close']
        if isinstance(close.index, pd.DatetimeIndex):
            intermediates['_weekly_close'] = close.resample('W-FRI').last()
            intermediates['_monthly_close'] = close.resample('ME').last()
    
    return intermediates

# In apply_features():
# Check if any weekly/monthly features are enabled
has_weekly_monthly = any('weekly' in name or 'monthly' in name 
                         for name in enabled_features.keys())
intermediates = compute_shared_intermediates(df, compute_resampled=has_weekly_monthly)
```

**Benefits**:
- Skips expensive resampling if not needed
- Could save significant time

---

### **STRATEGY F: Optimize Slowest Features Directly** ⭐⭐⭐⭐⭐
**Priority**: CRITICAL

**Concept**: The 5 slowest features account for 84% of time. Optimize them directly.

**Target Features**:
1. `hurst_exponent` (3.333s) - Complex rolling calculations
2. `trend_residual` (1.894s) - Rolling regression
3. `aroon_oscillator` (1.126s) - Window-based calculations
4. `mkt_spy_sma200_slope` (0.981s) - SPY data + rolling calculations
5. `fractal_dimension_index` (0.743s) - Complex window calculations

**Optimization Strategies**:
- **Profile each feature**: Use line_profiler to find slow lines
- **Vectorize loops**: Replace Python loops with numpy/pandas operations
- **Use numba JIT**: JIT-compile numeric-heavy portions
- **Algorithmic improvements**: Use more efficient algorithms
- **Cache intermediate results**: If features compute similar things, cache them

**Expected Impact**: 50-70% speedup if these 5 features are optimized

### **STRATEGY G: Migrate More High-Impact Features** ⭐⭐⭐
**Priority**: CRITICAL

**Concept**: Only 13 features use intermediates. Need to migrate more to see benefit.

**High-Impact Features to Migrate Next**:
1. **MACD features** (use EMA12, EMA26)
2. **Volume features** (use `_volume_avg_20d`)
3. **All features using `_returns_1d`** (many features compute this)
4. **All features using `_log_returns_1d`**
5. **Features using `_tr` or `_atr14`**

**Implementation**: Same pattern as before, but migrate in batches

**Expected Impact**: If 50+ features use intermediates, should see significant speedup

---

### **STRATEGY G: Optimize Intermediate Computation Itself** ⭐⭐⭐
**Priority**: MEDIUM

**Concept**: Some intermediate computations might be inefficient

**Potential Optimizations**:
1. **Vectorize RSI computation**: Current RSI computation might be slow
2. **Optimize True Range**: Current TR computation uses `pd.concat().max()` which might be slow
3. **Batch compute similar operations**: Compute all SMAs in one pass

**Example**:
```python
# Current: Individual computations
'_sma20': close.rolling(20, min_periods=1).mean(),
'_sma50': close.rolling(50, min_periods=1).mean(),
'_sma200': close.rolling(200, min_periods=1).mean(),

# Optimized: Could potentially batch, but pandas rolling is already optimized
```

**Note**: Pandas rolling operations are already highly optimized, so gains might be minimal

---

### **STRATEGY H: Parallel Intermediate Computation** ⭐⭐
**Priority**: LOW (probably not worth it)

**Concept**: Compute independent intermediates in parallel

**Problem**: Most intermediates depend on `_close`, so can't parallelize much

**Only candidates**:
- Volume averages (independent of price)
- Resampled series (independent of other intermediates)

**Expected Impact**: Minimal, probably not worth the complexity

---

## 📊 Diagnostic Steps (Do First!)

### Step 1: Add Timing Instrumentation
Add timing to `apply_features()` to see where time is spent:

```python
def apply_features(...):
    total_start = time.perf_counter()
    
    # Time intermediates
    intermediate_start = time.perf_counter()
    intermediates = compute_shared_intermediates(df)
    intermediate_time = time.perf_counter() - intermediate_start
    logger.info(f"Intermediates: {intermediate_time:.2f}s")
    
    # Time feature computation
    feature_start = time.perf_counter()
    feature_times = {}
    for name, func in enabled_features.items():
        start = time.perf_counter()
        # ... compute feature ...
        elapsed = time.perf_counter() - start
        feature_times[name] = elapsed
    
    feature_time = time.perf_counter() - feature_start
    logger.info(f"Features: {feature_time:.2f}s")
    logger.info(f"Total: {time.perf_counter() - total_start:.2f}s")
    
    # Report slowest
    slowest = sorted(feature_times.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Slowest 10 features:")
    for name, elapsed in slowest:
        logger.info(f"  {name}: {elapsed:.3f}s ({elapsed/feature_time*100:.1f}%)")
```

### Step 2: Profile Intermediate Computation
Time each intermediate:

```python
def compute_shared_intermediates(df: DataFrame) -> dict:
    times = {}
    intermediates = {}
    
    start = time.perf_counter()
    close = _get_close_series(df)
    intermediates['_close'] = close
    times['_close'] = time.perf_counter() - start
    
    start = time.perf_counter()
    intermediates['_sma20'] = close.rolling(20, min_periods=1).mean()
    times['_sma20'] = time.perf_counter() - start
    
    # ... etc
    
    # Log slowest intermediates
    slowest = sorted(times.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Slowest intermediates:")
    for key, elapsed in slowest:
        logger.info(f"  {key}: {elapsed:.3f}s")
    
    return intermediates
```

### Step 3: Count Intermediate Usage
Track which intermediates are actually used:

```python
class UsageTrackingIntermediates(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_count = {}
    
    def __getitem__(self, key):
        self.usage_count[key] = self.usage_count.get(key, 0) + 1
        return super().__getitem__(key)
    
    def get_unused(self):
        all_keys = set(self.keys())
        used_keys = set(self.usage_count.keys())
        return all_keys - used_keys
```

---

## 🎯 Recommended Action Plan

### **Immediate (Do Now)**:
1. ✅ **Add timing instrumentation** to see where time is spent
2. ✅ **Profile intermediate computation** to see if it's the bottleneck
3. ✅ **Count intermediate usage** to see which are actually used

### **Short Term (This Week)**:
4. ✅ **Remove unused intermediates** (especially resampled series if not needed)
5. ✅ **Cache function signatures** (easy win)
6. ✅ **Migrate more high-impact features** (MACD, volume, returns-based features)

### **Medium Term (Next Week)**:
7. ✅ **Implement lazy intermediate computation** if profiling shows it helps
8. ✅ **Selective intermediate computation** based on enabled features
9. ✅ **Optimize slow intermediate computations** if any are identified

---

## ❓ Key Questions to Answer

1. **How long does `compute_shared_intermediates()` take?**
   - If > 1 minute, that's the problem
   - If < 10 seconds, intermediates aren't the issue

2. **Which intermediates are actually used?**
   - If many are unused, remove them
   - If only a few are used, consider lazy computation

3. **Which features are slowest?**
   - Focus optimization on slowest features
   - Migrate slowest features to use intermediates first

4. **Is signature inspection taking time?**
   - If yes, cache signatures
   - If no, not worth optimizing

5. **Are resampled series being computed but not used?**
   - If yes, remove them or make lazy
   - If no, keep them

---

## 📝 Notes

- **15 minutes is worse than 12 minutes** - something is definitely wrong
- **Most likely culprit**: Intermediates computation taking too long, or not enough features using them
- **Need profiling data** to make informed decisions
- **Don't optimize blindly** - profile first, then optimize based on data
