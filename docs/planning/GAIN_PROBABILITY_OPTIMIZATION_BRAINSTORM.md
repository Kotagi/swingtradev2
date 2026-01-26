# Gain Probability Features Optimization Brainstorm
## Current Performance: 13 features take 700+ seconds (80% of total time)

**Date**: 2026-01-24

**Problem**: 13 gain_probability-related features take 50-100+ seconds EACH, totaling ~700 seconds (11.7 minutes) out of 15 minutes total

---

## 🔍 Root Cause Analysis

### **The Core Problem: `feature_historical_gain_probability()`**

This is the bottleneck. It's called by many other features and has **nested Python loops**:

```python
for i in range(horizon + 50, len(df)):  # ~4500 iterations
    # Inner loop: look back 252 days
    for j in range(i - 252, i - horizon):  # ~200 iterations per outer loop
        hist_close = close.iloc[j - horizon]
        hist_high = high.iloc[j - horizon:j].max()  # Another operation!
        hist_return = (hist_high / hist_close) - 1.0
        historical_achievements.append(1.0 if hist_return >= target_gain else 0.0)
```

**Complexity**: O(n × m) where:
- n = ~4500 rows (data length)
- m = ~200 rows (252 - horizon window)
- **Total iterations: ~900,000 per feature call!**

**Additional overhead**:
- `high.iloc[j - horizon:j].max()` called ~900,000 times
- `close.iloc[j - horizon]` called ~900,000 times
- List appends and mean calculations

### **Cascading Problem: Multiple Calls**

Many features call `feature_historical_gain_probability()` or `feature_gain_probability_score()` which calls it:

1. `feature_historical_gain_probability()` - **58.1s** (direct)
2. `feature_gain_probability_score()` - **53.8s** (calls #1)
3. `feature_gain_regime()` - **53.9s** (calls #2, which calls #1)
4. `feature_gain_consistency()` - **53.4s** (calls #1)
5. `feature_gain_probability_rank()` - **53.4s** (calls #2, which calls #1)
6. `feature_gain_probability_trend()` - **57.3s** (calls #2, which calls #1)
7. `feature_gain_probability_momentum()` - **52.6s** (calls #2, which calls #1)
8. `feature_gain_probability_volatility_adjusted()` - **55.7s** (calls #2, which calls #1)
9. `feature_gain_probability_consistency_rank()` - **48.6s** (calls #2, which calls #1)
10. `feature_gain_regime_transition_probability()` - **102.2s** (calls #2 and #3, which call #1)
11. `feature_volatility_gain_probability_interaction()` - **65.3s** (calls #2, which calls #1)
12. `feature_gain_probability_volatility_regime_interaction()` - **53.6s** (calls #2, which calls #1)
13. `feature_top_features_ensemble()` - **51.0s** (calls #2, which calls #1)

**Total**: `feature_historical_gain_probability()` is computed **~10+ times independently!**

---

## 💡 Optimization Strategies (Brainstorming)

### **STRATEGY 1: Cache `feature_historical_gain_probability()` Results** ⭐⭐⭐⭐⭐
**Impact**: 80-90% speedup | **Effort**: Low-Medium | **Risk**: Low | **Priority**: HIGHEST

**Concept**: Compute `historical_gain_probability` once, cache it, reuse for all dependent features

**Current Problem**:
- `feature_historical_gain_probability()` computed 10+ times
- Each call takes 58 seconds
- Total: 580+ seconds wasted

**Solution**:
```python
# In apply_features() or feature computation:
_gain_probability_cache = {}

def get_historical_gain_probability(df, target_gain=0.15, horizon=20):
    cache_key = (id(df), target_gain, horizon)
    if cache_key not in _gain_probability_cache:
        _gain_probability_cache[cache_key] = feature_historical_gain_probability(df, target_gain, horizon)
    return _gain_probability_cache[cache_key]

# Modify features to use cached version:
def feature_gain_probability_score(df, target_gain=0.15, horizon=20, cached_historical_prob=None):
    if cached_historical_prob is None:
        historical_prob = get_historical_gain_probability(df, target_gain, horizon)
    else:
        historical_prob = cached_historical_prob
    # ... rest of computation
```

**Expected Impact**: 
- Compute once: 58 seconds
- Reuse 10+ times: 0 seconds (just lookup)
- **Savings: ~520 seconds (8.7 minutes)**

**Implementation**:
- Add caching mechanism to `apply_features()`
- Pass cached results to dependent features
- Or use function-level caching with `functools.lru_cache`

---

### **STRATEGY 2: Vectorize `feature_historical_gain_probability()`** ⭐⭐⭐⭐⭐
**Impact**: 90-95% speedup for this feature | **Effort**: Medium-High | **Risk**: Medium | **Priority**: HIGH

**Concept**: Replace nested Python loops with vectorized pandas/numpy operations

**Current Implementation** (slow):
```python
for i in range(horizon + 50, len(df)):
    for j in range(i - 252, i - horizon):
        hist_close = close.iloc[j - horizon]
        hist_high = high.iloc[j - horizon:j].max()
        # ...
```

**Vectorized Approach** (fast):
```python
# Pre-compute rolling max high over horizon window
rolling_max_high = high.rolling(window=horizon, min_periods=1).max()

# Shift to look back horizon days
shifted_close = close.shift(horizon)
shifted_max_high = rolling_max_high.shift(horizon)

# Calculate returns vectorized
historical_returns = (shifted_max_high / shifted_close) - 1.0

# Check if target achieved (vectorized)
achieved = (historical_returns >= target_gain).astype(float)

# Calculate rolling probability over 252 days (vectorized)
probability = achieved.rolling(window=252-horizon, min_periods=1).mean()
```

**Key Optimizations**:
1. **Use `rolling().max()`** instead of `iloc[j-horizon:j].max()` in loop
2. **Use `shift()`** to look back instead of indexing in loop
3. **Vectorized comparisons** instead of if/else in loop
4. **Vectorized rolling mean** instead of manual list + mean

**Expected Impact**:
- Current: 58 seconds
- Vectorized: ~2-5 seconds
- **Speedup: 10-30x**

**Challenges**:
- Need to ensure lookback logic matches exactly
- Edge cases (first horizon+252 rows)
- Need careful testing to ensure same results

---

### **STRATEGY 3: Compute All Gain Probability Features Together** ⭐⭐⭐⭐
**Impact**: 50-70% speedup | **Effort**: Medium | **Risk**: Medium | **Priority**: HIGH

**Concept**: Compute all gain_probability features in one pass, sharing intermediate calculations

**Current Problem**:
- Each feature computes `historical_gain_probability` independently
- Each feature computes `gain_probability_score` independently
- Redundant calculations everywhere

**Solution**:
```python
def compute_gain_probability_family(df, target_gain=0.15, horizon=20):
    """Compute all gain_probability features in one pass"""
    
    # Compute base feature once
    historical_prob = feature_historical_gain_probability(df, target_gain, horizon)  # Once!
    
    # Compute score once
    gain_prob_score = feature_gain_probability_score(df, target_gain, horizon, 
                                                     cached_historical_prob=historical_prob)
    
    # Compute all dependent features using cached base
    results = {
        'historical_gain_probability': historical_prob,
        'gain_probability_score': gain_prob_score,
        'gain_regime': feature_gain_regime(df, target_gain, horizon, cached_score=gain_prob_score),
        'gain_consistency': feature_gain_consistency(df, target_gain, horizon, cached_historical_prob=historical_prob),
        'gain_probability_rank': feature_gain_probability_rank(df, cached_score=gain_prob_score),
        # ... etc for all 13 features
    }
    return results
```

**Expected Impact**:
- Compute base features once instead of 10+ times
- **Savings: ~500 seconds (8.3 minutes)**

---

### **STRATEGY 4: Optimize `feature_gain_regime()` Rolling Apply** ⭐⭐⭐
**Impact**: 20-30% speedup for this feature | **Effort**: Low | **Risk**: Low | **Priority**: MEDIUM

**Current Implementation** (slow):
```python
regime = probability_score.rolling(window=252, min_periods=50).apply(
    lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
    raw=False
)
```

**Problem**: `.apply()` with lambda is slow - called 4500 times

**Optimized Approach**:
```python
# Use vectorized percentile rank
regime = probability_score.rolling(window=252, min_periods=50).apply(
    lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) if len(x) > 1 else 0.5,
    raw=False
)

# Or better: use _rolling_percentile_rank utility if available
regime = _rolling_percentile_rank(probability_score, window=252, min_periods=50) / 100.0
```

**Expected Impact**: 53.9s → ~40s (20% faster)

---

### **STRATEGY 5: Optimize `feature_gain_regime_transition_probability()` Loop** ⭐⭐⭐
**Impact**: 30-50% speedup for this feature | **Effort**: Low-Medium | **Risk**: Low | **Priority**: MEDIUM

**Current Implementation** (slow):
```python
for i in range(len(df)):  # 4500 iterations
    if current_regime is None or abs(gain_regime.iloc[i] - current_regime) > 0.2:
        current_regime = gain_regime.iloc[i]
        duration = 1
    else:
        duration += 1
    regime_duration.iloc[i] = duration
```

**Vectorized Approach**:
```python
# Use pandas groupby or vectorized operations
regime_changes = (gain_regime.diff().abs() > 0.2).astype(int)
regime_groups = (regime_changes.cumsum())
regime_duration = regime_groups.groupby(regime_groups).cumcount() + 1
```

**Expected Impact**: 102.2s → ~50-70s (30-50% faster)

---

### **STRATEGY 6: Use Numba JIT for Hot Loops** ⭐⭐⭐
**Impact**: 50-80% speedup for affected features | **Effort**: High | **Risk**: High | **Priority**: MEDIUM (if other strategies insufficient)

**Concept**: JIT-compile the nested loops in `feature_historical_gain_probability()`

**Implementation**:
```python
from numba import jit

@jit(nopython=True)
def compute_historical_gain_probability_numba(close_array, high_array, target_gain, horizon):
    """Numba-optimized version"""
    n = len(close_array)
    probability = np.zeros(n)
    
    for i in range(horizon + 50, n):
        if i >= horizon + 252:
            achievements = []
            for j in range(i - 252, i - horizon):
                if j >= horizon:
                    hist_close = close_array[j - horizon]
                    hist_high = np.max(high_array[j - horizon:j])
                    hist_return = (hist_high / hist_close) - 1.0
                    achievements.append(1.0 if hist_return >= target_gain else 0.0)
            
            if len(achievements) > 0:
                probability[i] = np.mean(achievements)
    
    return probability

def feature_historical_gain_probability(df, target_gain=0.15, horizon=20):
    close = _get_close_series(df)
    high = _get_high_series(df)
    
    probability = compute_historical_gain_probability_numba(
        close.values, high.values, target_gain, horizon
    )
    
    return pd.Series(probability, index=df.index, name="historical_gain_probability")
```

**Expected Impact**: 58s → ~10-20s (3-6x faster)

**Challenges**:
- Need to rewrite in numba-compatible code (no pandas)
- Need to handle NaN values carefully
- Testing required to ensure precision matches

---

### **STRATEGY 7: Reduce Window Sizes (If Acceptable)** ⭐⭐
**Impact**: 30-50% speedup | **Effort**: Very Low | **Risk**: Medium (changes feature meaning) | **Priority**: LOW

**Concept**: Reduce rolling window from 252 days to smaller window (e.g., 126 days)

**Current**: 252-day rolling window
**Optimized**: 126-day rolling window (half the iterations)

**Trade-off**: 
- ✅ Faster computation
- ⚠️ Less historical data (may reduce accuracy)
- ⚠️ Changes feature semantics

**Only consider if**: Accuracy loss is acceptable for speed gain

---

### **STRATEGY 8: Parallelize Across Multiple Tickers** ⭐
**Impact**: Linear speedup with cores | **Effort**: Low | **Risk**: Low | **Priority**: LOW (already done)

**Note**: Already parallelized across tickers. This won't help per-ticker performance.

---

### **STRATEGY 9: Pre-compute and Cache Results** ⭐⭐
**Impact**: 100% speedup for repeated runs | **Effort**: Medium | **Risk**: Medium | **Priority**: LOW (only if incremental updates)

**Concept**: Cache computed gain_probability features to disk, only recompute when data changes

**Use Case**: Only useful if you're updating features incrementally (new data added)

**Not useful for**: Full rebuilds (which is current use case)

---

## 📊 Expected Combined Impact

### **Phase 1: Quick Wins** (1-2 days)
1. ✅ **Cache `historical_gain_probability`** (Strategy 1): 80-90% speedup
   - Current: 580+ seconds (10+ calls × 58s)
   - After: 58 seconds (compute once)
   - **Savings: ~520 seconds (8.7 minutes)**

**Result**: 15 minutes → **~6 minutes** (60% faster)

### **Phase 2: Vectorization** (2-3 days)
2. ✅ **Vectorize `historical_gain_probability`** (Strategy 2): Additional 90-95% speedup
   - Current: 58 seconds
   - After: 2-5 seconds
   - **Savings: ~53 seconds**

3. ✅ **Optimize `gain_regime` rolling apply** (Strategy 4): 20-30% speedup
   - Current: 53.9 seconds
   - After: ~40 seconds
   - **Savings: ~14 seconds**

4. ✅ **Optimize `gain_regime_transition_probability` loop** (Strategy 5): 30-50% speedup
   - Current: 102.2 seconds
   - After: ~50-70 seconds
   - **Savings: ~30-50 seconds**

**Result**: 6 minutes → **~4-5 minutes** (70-75% total faster)

### **Phase 3: Advanced** (1 week, if needed)
5. ✅ **Numba JIT** (Strategy 6): Additional 50-80% speedup for remaining loops
   - If vectorization doesn't work perfectly
   - Fallback option

**Result**: 4-5 minutes → **~2-3 minutes** (80-85% total faster)

---

## 🎯 Recommended Implementation Order

### **Immediate (Do First - Highest Impact)**:
1. **Strategy 1: Cache `historical_gain_probability`** 
   - Biggest impact (8.7 minutes saved)
   - Low risk
   - Easy to implement

### **Short Term (Next)**:
2. **Strategy 2: Vectorize `historical_gain_probability`**
   - Huge impact (53 seconds → 2-5 seconds)
   - Medium effort
   - Need careful testing

3. **Strategy 3: Compute gain_probability family together**
   - Additional savings
   - Medium effort
   - Good architecture

### **Medium Term (If Needed)**:
4. **Strategy 4 & 5: Optimize other slow features**
   - Additional 20-50% speedup
   - Lower priority after Strategy 1-2

---

## 🛡️ Data Integrity Considerations

### **For All Strategies**:

1. **Validation Testing**:
   - Compare optimized vs original outputs
   - Use `np.allclose()` with tight tolerance
   - Test edge cases (first rows, insufficient data)

2. **Precision Monitoring**:
   - Vectorized operations should match loop results
   - Numba might have slight precision differences
   - Document any acceptable differences

3. **Backward Compatibility**:
   - Keep original code as fallback
   - Feature flags to enable/disable optimizations
   - Gradual rollout

---

## ❓ Key Questions

1. **Can we change the algorithm?**
   - Current: Nested loops with lookback
   - Vectorized: Rolling windows with shifts
   - Need to verify they produce same results

2. **Is 252-day window required?**
   - Could reduce to 126 days for 2x speedup
   - But changes feature semantics

3. **Are all 13 features needed?**
   - Some might be redundant
   - Could disable less important ones

4. **Can we accept approximate results?**
   - Some optimizations might have tiny precision differences
   - Need to define acceptable tolerance

---

## 📝 Implementation Notes

### **Strategy 1 Implementation Details**:

```python
# In feature_pipeline.py or shared utils:
from functools import lru_cache

# Cache with key based on data hash and parameters
@lru_cache(maxsize=1)
def get_cached_historical_gain_probability(df_hash, target_gain, horizon):
    # This won't work directly with DataFrame
    # Need to use intermediate storage or pass through apply_features
    pass

# Better: Cache in apply_features()
_gain_prob_cache = {}

def apply_features(...):
    # Compute historical_gain_probability once
    cache_key = ('historical_gain_probability', 0.15, 20)
    if cache_key not in _gain_prob_cache:
        _gain_prob_cache[cache_key] = feature_historical_gain_probability(df, 0.15, 20)
    
    # Pass to dependent features
    for name, func in enabled_features.items():
        if 'cached_historical_prob' in inspect.signature(func).parameters:
            kwargs['cached_historical_prob'] = _gain_prob_cache[cache_key]
```

### **Strategy 2 Implementation Details**:

Key insight: The nested loop is computing:
- For each day i, look back 252 days
- For each day j in that window, check if j-horizon to j achieved target gain
- Calculate success rate

Vectorized approach:
1. Compute rolling max high over horizon window (once, vectorized)
2. Shift to look back horizon days
3. Calculate returns (vectorized)
4. Check target achievement (vectorized)
5. Rolling mean over 252-day window (vectorized)

---

## 🚀 Quick Win: Disable Features (If Not Critical)

**If these features aren't critical for your model**, you could:
- Disable the 13 gain_probability features
- **Savings: ~700 seconds (11.7 minutes)**
- **New total time: ~3-4 minutes**

**Trade-off**: Lose predictive power if these features are important

**Recommendation**: Only if you've confirmed they're not important via feature importance analysis

---

## 📊 Summary

**Current State**:
- 13 gain_probability features: ~700 seconds (80% of time)
- Rest of features: ~175 seconds (20% of time)
- **Total: 875 seconds (14.6 minutes)**

**After Strategy 1 (Caching)**:
- Compute `historical_gain_probability` once: 58 seconds
- Reuse 10+ times: 0 seconds
- **Savings: ~520 seconds**
- **New total: ~355 seconds (5.9 minutes)** ✅ **60% faster**

**After Strategy 1 + 2 (Caching + Vectorization)**:
- Compute `historical_gain_probability` once (vectorized): 2-5 seconds
- Reuse 10+ times: 0 seconds
- **Savings: ~650 seconds**
- **New total: ~225 seconds (3.8 minutes)** ✅ **75% faster**

**After All Strategies**:
- **New total: ~150-200 seconds (2.5-3.3 minutes)** ✅ **80-85% faster**
