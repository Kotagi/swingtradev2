# Feature Building Optimization Game Plan
## For 371 Features Across 500 Tickers

**Current Problem**: Building 371 features for 500 tickers takes hours (estimated 3-6+ hours)

**Goal**: Reduce to manageable time (< 1 hour for full rebuild, < 10 minutes for incremental)

**Last Updated**: 2025-01-22

---

## Current Architecture Analysis

### Current Flow (Per Ticker):
1. Load Parquet file (PyArrow) ✅ Already optimized
2. **SEQUENTIAL**: Compute 371 features one-by-one (BOTTLENECK #1)
3. Validate each feature individually (BOTTLENECK #2)
4. Concatenate all features
5. Write Parquet file (PyArrow) ✅ Already optimized

### Current Parallelization:
- ✅ Parallel across tickers (Joblib multiprocessing)
- ✅ SPY data pre-loaded and shared
- ❌ Sequential within each ticker (371 features computed one-by-one)

### Key Bottlenecks Identified:

1. **Sequential Feature Computation** (Biggest bottleneck)
   - 371 features computed one-by-one per ticker
   - Each feature is a separate function call
   - No shared intermediate calculations
   - Estimated: 60-70% of total time

2. **Redundant Calculations**
   - SMA20 computed ~50+ times across different features
   - SMA50 computed ~40+ times
   - RSI14 computed ~30+ times
   - ATR computed ~25+ times
   - Close/High/Low series extracted hundreds of times
   - Estimated: 20-30% of total time wasted

3. **Validation Overhead**
   - Each feature validated individually (NaN, inf, variance checks)
   - 371 validation calls per ticker
   - Estimated: 5-10% of total time

4. **Memory Allocation**
   - Each feature creates new Series
   - 371 Series objects created, then concatenated
   - Estimated: 3-5% of total time

---

## Optimization Strategy: 3-Phase Approach

### **PHASE 1: Quick Wins (1-2 days, 30-50% speedup)**

#### 1.1 Shared Intermediate Calculations Cache ⭐ **HIGHEST PRIORITY**
**Impact**: 30-40% speedup | **Effort**: Medium (1 day) | **Risk**: Low

**Problem**: Same calculations repeated hundreds of times
- SMA20, SMA50, SMA200 computed 50+ times each
- RSI14, RSI7, RSI21 computed 30+ times each
- ATR, volatility measures computed 25+ times each
- Close/High/Low series extracted 371 times

**Solution**: Pre-compute common intermediates once per ticker, cache in dict

**Implementation**:
```python
def compute_shared_intermediates(df: DataFrame) -> Dict[str, Series]:
    """Pre-compute all commonly used intermediates once"""
    close = _get_close_series(df)
    high = _get_high_series(df)
    low = _get_low_series(df)
    volume = _get_volume_series(df)
    
    intermediates = {
        # Price series
        '_close': close,
        '_high': high,
        '_low': low,
        '_volume': volume,
        
        # Moving averages (most common)
        '_sma20': close.rolling(20).mean(),
        '_sma50': close.rolling(50).mean(),
        '_sma200': close.rolling(200).mean(),
        '_ema20': close.ewm(span=20).mean(),
        '_ema50': close.ewm(span=50).mean(),
        '_ema200': close.ewm(span=200).mean(),
        
        # Volatility (very common)
        '_atr14': compute_atr(high, low, close, 14),
        '_volatility_21d': close.pct_change().rolling(21).std(),
        
        # Returns (common)
        '_returns_1d': close.pct_change(),
        '_log_returns_1d': np.log(close / close.shift(1)),
        
        # RSI (common)
        '_rsi14': compute_rsi(close, 14),
        '_rsi7': compute_rsi(close, 7),
        '_rsi21': compute_rsi(close, 21),
        
        # True Range
        '_tr': compute_true_range(high, low, close),
        
        # Volume averages
        '_volume_avg_20d': volume.rolling(20).mean(),
    }
    return intermediates

def apply_features_optimized(df, enabled_features, intermediates):
    """Use cached intermediates instead of recomputing"""
    for name, func in enabled_features.items():
        # Pass intermediates to feature function
        feature_series = func(df, intermediates=intermediates)
        ...
```

**Migration Strategy**:
- Phase 1a: Add intermediates parameter to feature functions (optional, backward compatible)
- Phase 1b: Update 50 most-used features first (SMAs, RSIs, volatility)
- Phase 1c: Gradually migrate remaining features
- Phase 1d: Remove redundant calculations from migrated features

**Expected Speedup**: 30-40% overall

---

#### 1.2 Batch Feature Validation ⭐ **HIGH PRIORITY**
**Impact**: 5-10% speedup | **Effort**: Low (2-3 hours) | **Risk**: Very Low

**Problem**: 371 individual validation calls per ticker

**Solution**: Validate all features at once after computation

**Implementation**:
```python
def validate_features_batch(feature_dict: Dict[str, Series]) -> Dict[str, List[str]]:
    """Validate all features in one pass"""
    issues = {}
    
    # Vectorized NaN check
    feature_df = pd.DataFrame(feature_dict)
    nan_counts = feature_df.isna().sum()
    inf_counts = np.isinf(feature_df.select_dtypes(include=[np.number])).sum()
    
    for name, series in feature_dict.items():
        feature_issues = []
        if nan_counts[name] > len(series) * 0.5:
            feature_issues.append(f"{nan_counts[name]} NaN values")
        if inf_counts[name] > 0:
            feature_issues.append(f"{inf_counts[name]} infinite values")
        if series.nunique() == 1 and series.notna().sum() > 1:
            feature_issues.append("constant value")
        if feature_issues:
            issues[name] = feature_issues
    
    return issues
```

**Expected Speedup**: 5-10% overall

---

#### 1.3 Feature Computation Ordering
**Impact**: 2-5% speedup | **Effort**: Low (1 hour) | **Risk**: Very Low

**Problem**: Slow features block progress, no early feedback

**Solution**: Compute fast features first, slow features last

**Fast Features** (compute first - give early progress):
- Price features (price, price_log, price_vs_ma200)
- Simple returns (daily_return, gap_pct)
- Candlestick components (candle_body_pct, etc.)
- Simple flags (higher_high_10d, higher_low_10d)

**Slow Features** (compute last):
- hurst_exponent (100-period rolling)
- fractal_dimension_index (100-period)
- beta_spy_252d (252-day rolling)
- volatility_of_volatility (nested rolling)
- All multi-timeframe features (resampling overhead)

**Implementation**: Sort enabled_features dict by estimated computation time

**Expected Speedup**: 2-5% (mostly perceived, some actual from better CPU cache usage)

---

#### 1.4 Disable Debug Logging During Computation
**Impact**: 2-3% speedup | **Effort**: Very Low (30 min) | **Risk**: Very Low

**Problem**: `logger.debug()` called 371 times per ticker × 500 tickers = 185,500 log calls

**Solution**: Set log level to INFO during feature computation, only log errors

**Expected Speedup**: 2-3% overall

---

### **PHASE 2: Medium Impact (3-5 days, Additional 20-30% speedup)**

#### 2.1 Feature Dependency Graph & Parallel Feature Groups
**Impact**: 15-25% speedup | **Effort**: Medium-High (2-3 days) | **Risk**: Medium

**Problem**: Independent features computed sequentially

**Solution**: Group features by dependencies, compute independent groups in parallel

**Dependency Groups**:
- **Group 0** (Base - no dependencies): price, price_log, daily_return, gap_pct
- **Group 1** (Depends on Group 0): sma20_ratio, sma50_ratio, volatility_5d
- **Group 2** (Depends on Group 1): rsi14, macd_line, trend_strength_20d
- **Group 3** (Depends on Group 2): signal_quality_score, gain_probability_score
- etc.

**Implementation**:
```python
def build_dependency_graph(features: Dict) -> Dict[int, List[str]]:
    """Group features by dependency level"""
    # Analyze feature signatures and imports to determine dependencies
    # Return dict: {level: [feature_names]}
    ...

def compute_features_parallel_groups(df, feature_groups, intermediates):
    """Compute features in parallel within each dependency group"""
    results = {}
    for level in sorted(feature_groups.keys()):
        group_features = feature_groups[level]
        # Compute this group in parallel (if > threshold)
        if len(group_features) > 10:
            group_results = Parallel(n_jobs=-1)(
                delayed(compute_feature)(name, func, df, intermediates, results)
                for name, func in group_features.items()
            )
        else:
            # Sequential for small groups
            group_results = [compute_feature(name, func, df, intermediates, results)
                           for name, func in group_features.items()]
        results.update(group_results)
    return results
```

**Expected Speedup**: 15-25% overall

---

#### 2.2 Vectorized Feature Computation (Where Possible)
**Impact**: 10-15% speedup | **Effort**: Medium (2 days) | **Risk**: Medium

**Problem**: Many features use similar patterns that could be vectorized

**Solution**: Batch compute similar features together

**Candidates**:
- All SMA ratios (sma20_ratio, sma50_ratio, sma200_ratio) - compute all SMAs once, then ratios
- All RSI variants (rsi7, rsi14, rsi21) - compute all at once
- All volatility measures (volatility_5d, volatility_21d) - compute all windows at once
- All momentum variants (momentum_5d, momentum_10d, momentum_20d) - compute all at once

**Implementation**:
```python
def compute_sma_ratios_vectorized(close, windows=[20, 50, 200]):
    """Compute all SMA ratios at once"""
    smas = {f'sma{w}': close.rolling(w).mean() for w in windows}
    ratios = {f'sma{w}_ratio': close / smas[f'sma{w}'] for w in windows}
    return ratios

def compute_rsi_vectorized(close, periods=[7, 14, 21]):
    """Compute all RSI variants at once"""
    rsis = {}
    for period in periods:
        rsis[f'rsi{period}'] = compute_rsi(close, period)
    return rsis
```

**Expected Speedup**: 10-15% overall

---

#### 2.3 Memory-Efficient Feature Concatenation
**Impact**: 5-10% speedup | **Effort**: Low (1 day) | **Risk**: Low

**Problem**: Creating 371 Series, then concatenating is memory-intensive

**Solution**: Build DataFrame incrementally or use more efficient concatenation

**Implementation**:
```python
# Current: Create 371 Series, then concat
feature_dict = {name: func(df) for name, func in enabled_features.items()}
feature_df = pd.DataFrame(feature_dict)

# Optimized: Build DataFrame column-by-column (more memory efficient)
feature_df = pd.DataFrame(index=df.index)
for name, func in enabled_features.items():
    feature_df[name] = func(df)
    # Optionally: Clear intermediate Series if not needed
```

**Expected Speedup**: 5-10% overall (mostly memory pressure reduction)

---

#### 2.4 Conditional Validation (Skip Simple Features)
**Impact**: 3-5% speedup | **Effort**: Low (2 hours) | **Risk**: Very Low

**Problem**: Validating simple features like `price` or `price_log` is unnecessary

**Solution**: Skip validation for "safe" features, only validate complex ones

**Safe Features** (skip validation):
- price, price_log
- daily_return, gap_pct
- Simple ratios (sma20_ratio, etc.)
- Simple flags (higher_high_10d, etc.)

**Complex Features** (always validate):
- Statistical features (skewness, kurtosis)
- Advanced indicators (hurst_exponent, fractal_dimension)
- Market context features (beta, SPY features)

**Expected Speedup**: 3-5% overall

---

### **PHASE 3: Advanced Optimizations (1-2 weeks, Additional 15-25% speedup)**

#### 3.1 Numba JIT for Hot Paths
**Impact**: 20-40% speedup for affected features | **Effort**: High (3-5 days) | **Risk**: Medium-High

**Problem**: Some features have computationally intensive loops

**Solution**: JIT-compile numeric-heavy portions with Numba

**Candidates** (profile first to confirm):
- `hurst_exponent` (complex rolling calculations)
- `fractal_dimension_index` (window-based calculations)
- `trend_residual` (rolling regression)
- `aroon_up`, `aroon_down` (rolling window with index tracking)
- Custom rolling correlation functions
- `historical_gain_probability` (nested loops)

**Implementation**:
```python
from numba import jit

@jit(nopython=True)
def compute_hurst_exponent_numba(returns, max_lag):
    # Rewrite numeric-heavy portion in Numba-compatible code
    ...

def feature_hurst_exponent(df):
    returns = df['close'].pct_change().values
    hurst = compute_hurst_exponent_numba(returns, 100)
    return pd.Series(hurst, index=df.index)
```

**Expected Speedup**: 20-40% for affected features (5-10% overall if 10-20 features optimized)

---

#### 3.2 Feature Computation Caching (Within Ticker)
**Impact**: 50-90% speedup for incremental updates | **Effort**: High (1 week) | **Risk**: Medium

**Problem**: Recomputing all features when only new rows added

**Solution**: Cache intermediate results, only compute new rows

**Use Cases**:
- Adding new tickers (only compute new tickers)
- Updating recent data (only recompute affected date ranges)
- Feature set changes (only recompute affected features)

**Implementation Complexity**: Very High
- Requires dependency tracking
- Date range handling for rolling windows
- Cache invalidation logic
- Storage for cached intermediates

**Recommendation**: Only if incremental updates are common use case

---

#### 3.3 Pre-compute Indicator Library
**Impact**: 30-50% speedup | **Effort**: Very High (2-3 weeks) | **Risk**: High

**Problem**: Features are transformations of indicators, but indicators computed repeatedly

**Solution**: Pre-compute all indicators once, store in standardized format

**Architecture**:
1. Compute all indicators once per ticker (SMAs, EMAs, RSIs, MACD, etc.)
2. Store in standardized DataFrame/Parquet format
3. Features become lookups/transformations of indicators
4. Enables feature versioning and easier testing

**Benefits**:
- Indicators computed once
- Features become fast transformations
- Easier to test and validate
- Enables feature versioning

**Drawbacks**:
- Major architecture change
- Requires refactoring all features
- Storage overhead for indicators

**Recommendation**: Consider for long-term architecture, but not immediate priority

---

## Implementation Priority & Timeline

### **Week 1: Quick Wins (Target: 30-50% speedup)**
1. ✅ **1.1 Shared Intermediate Calculations** (1 day) - **BIGGEST IMPACT**
2. ✅ **1.2 Batch Feature Validation** (2-3 hours)
3. ✅ **1.3 Feature Computation Ordering** (1 hour)
4. ✅ **1.4 Disable Debug Logging** (30 min)

**Expected Result**: 3-6 hours → **1.5-3 hours** (30-50% faster)

---

### **Week 2-3: Medium Impact (Target: Additional 20-30% speedup)**
5. **2.1 Feature Dependency Graph** (2-3 days)
6. **2.2 Vectorized Feature Computation** (2 days)
7. **2.3 Memory-Efficient Concatenation** (1 day)
8. **2.4 Conditional Validation** (2 hours)

**Expected Result**: 1.5-3 hours → **1-2 hours** (50-70% total speedup)

---

### **Month 2: Advanced (Target: Additional 15-25% speedup)**
9. **3.1 Numba JIT** (3-5 days) - Profile first to identify targets
10. **3.2 Feature Caching** (1 week) - Only if incremental updates needed
11. **3.3 Pre-compute Indicators** (2-3 weeks) - Long-term architecture

**Expected Result**: 1-2 hours → **30-60 minutes** (70-90% total speedup)

---

## Profiling Plan (Do First!)

Before implementing, profile to identify actual bottlenecks:

### Profiling Steps:
1. **Time each feature function** - Identify slowest 20 features
2. **Count redundant calculations** - How many times is SMA20 computed?
3. **Measure validation overhead** - Time spent in validation vs computation
4. **Profile memory usage** - Memory pressure points
5. **Measure I/O time** - Parquet read/write vs computation time

### Tools:
```python
# Add to apply_features():
import time
feature_times = {}
for name, func in enabled_features.items():
    start = time.perf_counter()
    feature_series = func(df)
    elapsed = time.perf_counter() - start
    feature_times[name] = elapsed
    logger.info(f"{name}: {elapsed:.4f}s")

# Sort and report top 20 slowest
slowest = sorted(feature_times.items(), key=lambda x: x[1], reverse=True)[:20]
```

### Key Metrics:
- Total time per ticker
- Time per feature (average, min, max, top 20)
- I/O time vs computation time ratio
- Memory usage per worker
- Redundant calculation count

---

## Success Metrics

### Performance Targets:
- **Phase 1**: 30-50% speedup (3-6 hours → 1.5-3 hours)
- **Phase 2**: Additional 20-30% (1.5-3 hours → 1-2 hours)
- **Phase 3**: Additional 15-25% (1-2 hours → 30-60 minutes)

### Quality Targets:
- ✅ No loss of data integrity
- ✅ No loss of accuracy
- ✅ All existing tests pass
- ✅ Feature outputs match original (within floating point tolerance)

### Monitoring:
- Track feature calculation time per run
- Monitor for regressions
- Track error rates (should not increase)

---

## Risk Mitigation

### For Each Optimization:
1. **Implement behind feature flag** - Can enable/disable per optimization
2. **Test independently** - Don't combine optimizations until each is proven
3. **Keep original code** - Comment out, don't delete, until optimization is stable
4. **Validate outputs** - Compare optimized vs original outputs
5. **Profile before/after** - Measure actual speedup

### Rollback Plan:
- Each optimization is independent
- Can disable any optimization via feature flag
- Original code preserved in comments
- Git branches for each optimization

---

## Alternative Approaches (If Above Not Sufficient)

### Option A: Reduce Feature Count
- Identify low-importance features (SHAP importance < threshold)
- Disable or remove redundant features
- **Impact**: Linear reduction (disable 100 features = ~27% faster)
- **Trade-off**: May lose some predictive power

### Option B: Incremental Computation
- Only compute features for new/updated tickers
- Cache feature results
- **Impact**: 50-90% faster for incremental updates
- **Trade-off**: Complexity, cache management

### Option C: Distributed Computing
- Use Dask or Ray to distribute across multiple machines
- **Impact**: Linear scaling with machines
- **Trade-off**: Infrastructure complexity, network overhead

### Option D: Feature Sampling
- Compute features in batches (e.g., 100 features at a time)
- Train multiple models, ensemble
- **Impact**: Faster per batch, but multiple training runs
- **Trade-off**: More complex training workflow

---

## Next Steps

1. **Profile current implementation** (1-2 hours)
   - Time each feature
   - Identify slowest features
   - Measure redundant calculations

2. **Implement Phase 1.1** (Shared Intermediates) - **START HERE**
   - Highest impact, medium effort
   - Expected 30-40% speedup

3. **Test and validate** (2-3 hours)
   - Compare outputs
   - Measure actual speedup
   - Fix any issues

4. **Continue with Phase 1.2-1.4** (Quick wins)
   - Lower impact but easy to implement
   - Additional 5-10% speedup

5. **Evaluate and decide on Phase 2**
   - If Phase 1 achieves target, may not need Phase 2
   - If not, proceed with Phase 2 optimizations

---

## Notes

- **Data Integrity**: All optimizations must maintain exact same outputs
- **Backward Compatibility**: Optimizations should not break existing workflows
- **Testing**: Each optimization tested independently
- **Documentation**: Update as optimizations are implemented

---

## Questions to Answer

1. **What's the actual bottleneck?** (Profile first!)
   - CPU-bound (feature computation)?
   - I/O-bound (reading/writing files)?
   - Memory-bound (not enough RAM)?

2. **Which features are slowest?** (Profile to identify)
   - Focus optimization efforts on slowest features

3. **How many redundant calculations?** (Count them)
   - SMA20 computed how many times?
   - RSI14 computed how many times?

4. **What's the typical use case?**
   - Full rebuilds (all tickers)?
   - Incremental updates (new tickers only)?
   - Feature set changes?

5. **What's the target time?**
   - < 1 hour for full rebuild?
   - < 10 minutes for incremental?
