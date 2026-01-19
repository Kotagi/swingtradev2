# Feature Calculation Optimization Roadmap

**Goal**: Speed up feature calculation without losing data integrity or accuracy

**Current State**: 
- Uses Joblib Parallel with multiprocessing backend
- Processes files in batches (5% or 10 files per batch)
- Each worker loads SPY data independently
- Features computed sequentially per ticker
- Full validation on every feature

**Last Updated**: 2025-01-19

---

## Phase 1: Quick Wins (High Impact, Low Effort)

### 1.1 Pre-load and Share SPY Data Across Processes ⭐ **HIGHEST PRIORITY**
- **Current**: Each worker process loads SPY data independently (module-level cache doesn't persist across processes)
- **Solution**: Load SPY data once in main process, pass as argument to workers
- **Impact**: Eliminates redundant I/O and parsing for market context features (beta, SPY distance, SPY slope)
- **Estimated Speedup**: 5-10% overall
- **Effort**: Low (2-3 hours)
- **Risk**: Low
- **Dependencies**: None

**Implementation Notes**:
- Modify `process_file()` to accept optional `spy_data` parameter
- Load SPY data in `main()` before parallel processing
- Pass SPY data to each worker via `delayed(process_file)(..., spy_data=spy_data)`
- Update feature functions to use passed SPY data instead of loading

---

### 1.2 Optimize Batch Size
- **Current**: `batch_size = max(1, min(10, len(file_list) // 20))` (5% or 10 files)
- **Solution**: Increase to 20-50 files per batch, or remove batching entirely
- **Impact**: Reduces batch overhead cycles
- **Estimated Speedup**: 5-10% overall
- **Effort**: Very Low (30 minutes)
- **Risk**: Very Low
- **Dependencies**: None

**Options**:
- **Option A**: Increase batch size to 20-30 files
- **Option B**: Remove batching, process all files in single Parallel call (simpler, less granular progress)
- **Option C**: Make batch size configurable via command-line argument

**Recommendation**: Start with Option A, test performance, then consider Option B if progress updates aren't critical.

---

### 1.3 Use PyArrow Engine for Parquet I/O
- **Current**: Default pandas Parquet engine (likely 'fastparquet')
- **Solution**: Use `engine='pyarrow'` for all Parquet read/write operations
- **Impact**: Faster I/O operations
- **Estimated Speedup**: 5-15% overall (I/O bound operations)
- **Effort**: Low (1 hour)
- **Risk**: Very Low (PyArrow is well-tested)
- **Dependencies**: Requires pyarrow package (likely already installed)

**Implementation**:
- Update `pd.read_parquet()` calls to include `engine='pyarrow'`
- Update `df.to_parquet()` calls to include `engine='pyarrow'`
- Test compatibility with existing Parquet files

---

### 1.4 Disable Debug Logging During Feature Computation
- **Current**: `logger.debug()` called for every feature computation
- **Solution**: Disable debug logging or batch log messages
- **Impact**: Reduces I/O overhead from logging
- **Estimated Speedup**: 2-5% overall
- **Effort**: Very Low (30 minutes)
- **Risk**: Very Low
- **Dependencies**: None

**Implementation**:
- Set logging level to INFO or WARNING during feature computation
- Keep ERROR level logging for failures
- Optionally: Batch log messages instead of per-feature logging

---

## Phase 2: Medium Impact Optimizations (Moderate Effort)

### 2.1 Feature Computation Dependency Optimization
- **Current**: Features computed sequentially, some redundantly
- **Solution**: Identify dependencies, compute once and reuse
- **Impact**: Eliminates redundant calculations
- **Estimated Speedup**: 10-20% overall
- **Effort**: Medium (1-2 days)
- **Risk**: Medium (requires careful dependency tracking)
- **Dependencies**: None

**Known Dependencies**:
- `volatility_ratio` depends on `volatility_5d` and `volatility_21d`
- `aroon_oscillator` depends on `aroon_up` and `aroon_down`
- Multiple features use SMA20, SMA50, SMA200 (could compute once)

**Implementation Approach**:
1. Create dependency graph of features
2. Compute base features first (SMAs, EMAs, volatility measures)
3. Pass computed intermediates to dependent features
4. Modify feature functions to accept optional pre-computed values

---

### 2.2 Shared Intermediate Calculations
- **Current**: Many features compute similar rolling operations independently
- **Solution**: Compute common intermediates once per ticker, pass to feature functions
- **Impact**: Reduces redundant rolling window calculations
- **Estimated Speedup**: 15-25% overall
- **Effort**: Medium-High (2-3 days)
- **Risk**: Medium (requires refactoring feature functions)
- **Dependencies**: 2.1 (dependency optimization)

**Common Intermediates to Pre-compute**:
- All SMAs (20, 50, 200)
- All EMAs (12, 26)
- Volatility measures (5d, 21d)
- True Range / ATR
- Price changes / returns

**Implementation Approach**:
1. Create `compute_intermediates()` function
2. Modify `apply_features()` to compute intermediates first
3. Update feature functions to accept intermediates as optional parameters
4. Gradually migrate features to use intermediates

---

### 2.3 Optimize Feature Function Order
- **Current**: Features computed in arbitrary order
- **Solution**: Compute fast features first, slow features later
- **Impact**: Better perceived performance, early progress indication
- **Estimated Speedup**: Perceived (actual speedup minimal)
- **Effort**: Low (2-3 hours)
- **Risk**: Very Low
- **Dependencies**: None

**Fast Features** (compute first):
- `price`, `price_log`, `price_vs_ma200`
- `daily_return`, `gap_pct`
- `candle_body_pct`, `candle_upper_wick_pct`, `candle_lower_wick_pct`
- `higher_high_10d`, `higher_low_10d`

**Slow Features** (compute last):
- `hurst_exponent` (100-period rolling window with complex calculations)
- `fractal_dimension_index` (100-period window)
- `beta_spy_252d` (252-day rolling calculations)
- `mkt_spy_dist_sma200`, `mkt_spy_sma200_slope` (complex market calculations)

**Implementation**:
- Sort `enabled_features` dict by estimated computation time
- Or create priority groups and process in order

---

### 2.4 Reduce Validation Overhead
- **Current**: Every feature validated individually (NaN checks, infinity checks, variance checks)
- **Solution**: Optimize validation strategy
- **Impact**: Reduces per-feature overhead
- **Estimated Speedup**: 5-10% overall
- **Effort**: Low-Medium (4-6 hours)
- **Risk**: Low (validation still happens, just more efficiently)
- **Dependencies**: None

**Options**:
- **Option A**: Validate only at end (batch validation)
- **Option B**: Skip validation for simple features (`price`, `price_log`, etc.)
- **Option C**: Batch validation (validate all features at once instead of one-by-one)
- **Option D**: Conditional validation (only validate if feature computation took > threshold time)

**Recommendation**: Start with Option B (skip validation for simple features), then consider Option C for batch validation.

---

## Phase 3: Advanced Optimizations (Higher Effort, Higher Risk)

### 3.1 Use Numba for Hot Paths
- **Current**: Pure Python/Pandas for computationally intensive features
- **Solution**: JIT-compile computationally intensive features with Numba
- **Impact**: 10-100x speedup for numeric-heavy code
- **Estimated Speedup**: 20-40% overall (for affected features)
- **Effort**: High (3-5 days)
- **Risk**: Medium-High (requires careful testing, Numba compatibility issues)
- **Dependencies**: Requires numba package

**Candidates for Numba JIT**:
- `hurst_exponent` (complex rolling calculations)
- `fractal_dimension_index` (window-based calculations)
- `trend_residual` (rolling regression)
- `aroon_up`, `aroon_down` (rolling window max/min with index tracking)
- Custom rolling operations

**Implementation Approach**:
1. Identify slowest features via profiling
2. Rewrite numeric-heavy portions in Numba-compatible code
3. Add `@numba.jit` decorator
4. Extensive testing to ensure accuracy matches original

---

### 3.2 Memory Optimization
- **Current**: Each worker loads full DataFrame into memory
- **Solution**: Optimize memory usage
- **Impact**: Allows processing more files in parallel, reduces memory pressure
- **Estimated Speedup**: 10-20% (indirect - allows more parallelism)
- **Effort**: Medium (2-3 days)
- **Risk**: Medium (requires careful memory management)
- **Dependencies**: None

**Strategies**:
- Use more efficient dtypes (float32 instead of float64 where precision allows)
- Clear intermediate DataFrames explicitly (`del df_intermediate`)
- Process very large tickers in chunks
- Use `copy=False` where safe (avoid unnecessary copies)

**Implementation**:
1. Profile memory usage per ticker
2. Identify memory hotspots
3. Optimize dtype usage
4. Add explicit memory cleanup

---

### 3.3 Parallelize Within Ticker (for Large Tickers)
- **Current**: Features computed sequentially per ticker
- **Solution**: For tickers with many rows, parallelize feature computation
- **Impact**: Speedup for large tickers only
- **Estimated Speedup**: 2-5x for tickers with >10k rows
- **Effort**: High (3-4 days)
- **Risk**: High (complexity, may not help for typical ticker sizes)
- **Dependencies**: None

**Considerations**:
- Only beneficial for tickers with many rows (>5000-10000)
- Overhead may outweigh benefits for typical tickers
- Requires careful handling of rolling windows (can't parallelize easily)

**Recommendation**: Profile first to see if large tickers are a bottleneck. If not, skip this optimization.

---

## Phase 4: Architecture-Level Changes (Major Refactoring)

### 4.1 Use Dask or Ray Instead of Joblib
- **Current**: Joblib with multiprocessing backend
- **Solution**: Migrate to Dask or Ray for better parallelization
- **Impact**: Better scalability, more flexible parallelization
- **Estimated Speedup**: 10-30% (better resource utilization)
- **Effort**: Very High (1-2 weeks)
- **Risk**: High (major refactoring, new dependencies)
- **Dependencies**: Requires dask or ray package

**Benefits**:
- Better handling of large datasets
- More flexible task scheduling
- Better memory management
- Can distribute across multiple machines (Ray)

**Drawbacks**:
- Significant refactoring required
- New dependency
- Learning curve
- May be overkill for current scale

**Recommendation**: Only consider if current parallelization is insufficient or if scaling to multiple machines.

---

### 4.2 Incremental Feature Updates
- **Current**: Recompute all features when any input changes
- **Solution**: Only recompute features that changed or depend on changed features
- **Impact**: Very fast for small updates
- **Estimated Speedup**: 50-90% for incremental updates
- **Effort**: Very High (2-3 weeks)
- **Risk**: Very High (complex dependency tracking, edge cases)
- **Dependencies**: Requires dependency graph implementation

**Use Cases**:
- Adding new tickers (only compute new tickers)
- Updating recent data (only recompute affected date ranges)
- Feature set changes (only recompute affected features)

**Implementation Complexity**:
- Requires comprehensive dependency graph
- Need to track what changed
- Complex date range handling for rolling windows
- Edge cases with dependencies

**Recommendation**: Consider only if incremental updates are a common use case.

---

### 4.3 Pre-compute Shared Indicators
- **Current**: Each feature computes its own indicators
- **Solution**: Pre-compute all technical indicators, store in shared format
- **Impact**: Features become lookups/transformations
- **Estimated Speedup**: 30-50% overall
- **Effort**: Very High (2-3 weeks)
- **Risk**: High (major architecture change)
- **Dependencies**: Requires indicator computation framework

**Architecture**:
1. Compute all indicators once per ticker
2. Store in standardized format
3. Features become transformations/lookups of indicators
4. Enables feature versioning and easier testing

**Recommendation**: Consider for long-term architecture, but not immediate priority.

---

## Implementation Priority Matrix

### Immediate (Do First - Week 1)
1. ✅ Pre-load and Share SPY Data (1.1)
2. ✅ Use PyArrow Engine (1.3)
3. ✅ Optimize Batch Size (1.2)
4. ✅ Disable Debug Logging (1.4)

**Expected Combined Speedup**: 15-30%

### Short Term (Next 2-4 Weeks)
5. Optimize Feature Function Order (2.3)
6. Reduce Validation Overhead (2.4)
7. Feature Computation Dependency Optimization (2.1)

**Expected Additional Speedup**: 15-30%

### Medium Term (1-2 Months)
8. Shared Intermediate Calculations (2.2)
9. Memory Optimization (3.2)
10. Use Numba for Hot Paths (3.1) - if profiling shows benefit

**Expected Additional Speedup**: 20-40%

### Long Term (Future Consideration)
11. Use Dask/Ray (4.1) - only if current parallelization insufficient
12. Incremental Feature Updates (4.2) - only if needed
13. Pre-compute Shared Indicators (4.3) - architecture decision

---

## Profiling Recommendations

Before implementing advanced optimizations, profile the code to identify actual bottlenecks:

### Profiling Steps:
1. **Time each feature function** - Identify slowest features
2. **Profile I/O operations** - Measure Parquet read/write times
3. **Profile memory usage** - Identify memory hotspots
4. **Profile parallelization overhead** - Measure batch processing overhead
5. **Profile SPY data loading** - Measure impact of redundant loading

### Tools:
- `cProfile` for function-level profiling
- `line_profiler` for line-by-line profiling
- `memory_profiler` for memory usage
- `py-spy` for runtime profiling

### Key Metrics to Track:
- Time per ticker (average, min, max)
- Time per feature (average, min, max)
- I/O time vs computation time ratio
- Memory usage per worker
- Parallelization efficiency (actual speedup vs theoretical)

---

## Success Metrics

### Performance Targets:
- **Phase 1**: 15-30% overall speedup
- **Phase 2**: Additional 15-30% speedup (30-60% total)
- **Phase 3**: Additional 20-40% speedup (50-100% total)

### Quality Targets:
- ✅ No loss of data integrity
- ✅ No loss of accuracy
- ✅ All existing tests pass
- ✅ Feature outputs match original (within floating point tolerance)

### Monitoring:
- Track feature calculation time per run
- Monitor for regressions in calculation speed
- Track error rates (should not increase)

---

## Notes

- **Data Integrity**: All optimizations must maintain exact same outputs (within floating point precision)
- **Backward Compatibility**: Optimizations should not break existing feature sets or workflows
- **Testing**: Each optimization should be tested independently before moving to next
- **Rollback Plan**: Keep original implementation until optimizations are proven stable

---

## Questions to Answer Before Implementation

1. **What's the current bottleneck?**
   - CPU-bound (feature computation)?
   - I/O-bound (reading/writing files)?
   - Memory-bound (not enough RAM)?

2. **How many tickers are being processed?**
   - Affects batch size optimization
   - Affects whether per-ticker parallelization helps

3. **Average rows per ticker?**
   - Affects whether within-ticker parallelization is beneficial
   - Affects memory optimization strategy

4. **Which features are slowest?**
   - Profile to identify targets for Numba optimization
   - Helps prioritize dependency optimization

5. **What's the typical use case?**
   - Full rebuilds (all tickers)?
   - Incremental updates (new tickers only)?
   - Feature set changes?

---

## References

- Current implementation: `src/feature_pipeline.py`
- Feature definitions: `features/sets/v2/technical.py`
- Shared utilities: `features/shared/utils.py`
- Joblib documentation: https://joblib.readthedocs.io/
- Numba documentation: https://numba.pydata.org/
- PyArrow documentation: https://arrow.apache.org/docs/python/
