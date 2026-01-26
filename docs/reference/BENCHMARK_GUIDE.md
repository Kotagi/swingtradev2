# Feature Pipeline Benchmark Guide

This guide explains how to measure performance improvements from Phase 1 optimizations.

## Quick Start

### 1. Establish Baseline (Before Optimization)

First, test the current performance to establish a baseline:

```bash
python benchmark_feature_pipeline.py --num-tickers 20 --runs 3
```

This will:
- Test with 20 tickers (adjust with `--num-tickers`)
- Run 3 times and average the results
- Save results to `benchmark_results.txt`

**Important:** Make sure you're on the `main` branch or before the optimization changes!

### 2. Apply Optimization

Switch to your optimization branch:

```bash
git checkout optimize/phase1-spy-data-sharing
```

### 3. Test After Optimization

Run the same benchmark:

```bash
python benchmark_feature_pipeline.py --num-tickers 20 --runs 3
```

### 4. Compare Results

```bash
python compare_benchmark_results.py
```

This will show you the improvement percentage.

## Detailed Usage

### Benchmark Script Options

```bash
python benchmark_feature_pipeline.py [OPTIONS]
```

**Options:**
- `--num-tickers N`: Number of tickers to test (default: 20)
  - Use fewer tickers (10-20) for quick tests
  - Use more tickers (50-100) for more accurate results
- `--feature-set SET`: Feature set to use (default: v1)
- `--runs N`: Number of benchmark runs to average (default: 3)
  - More runs = more accurate but slower
- `--input-dir PATH`: Input directory (default: data/clean)
- `--benchmark-spy`: Also benchmark SPY loading time separately

### Example Commands

**Quick test (10 tickers, 2 runs):**
```bash
python benchmark_feature_pipeline.py --num-tickers 10 --runs 2
```

**Comprehensive test (50 tickers, 5 runs):**
```bash
python benchmark_feature_pipeline.py --num-tickers 50 --runs 5 --benchmark-spy
```

**Test specific feature set:**
```bash
python benchmark_feature_pipeline.py --num-tickers 20 --feature-set v2
```

## Understanding Results

### Metrics Reported

1. **Total Time**: Time to process all tickers
   - This is the main metric for overall speedup
   - Expected improvement: 5-10% for optimization 1.1

2. **Time Per Ticker**: Average time per individual ticker
   - Useful for understanding per-ticker performance
   - Less variable than total time

3. **SPY Loading Time** (if `--benchmark-spy` used):
   - Time to load SPY data once
   - Multiply by number of workers to estimate total savings

### What to Look For

**Good signs:**
- Consistent results across multiple runs (low std dev)
- Measurable improvement in total time
- SPY loading happens once (visible in logs)

**Red flags:**
- High variance between runs (might indicate system load issues)
- No improvement (optimization might not be working)
- Errors during benchmark (check logs)

## Testing Strategy

### Phase 1: Quick Validation (5-10 minutes)

Test that the optimization works at all:

```bash
# Small test
python benchmark_feature_pipeline.py --num-tickers 10 --runs 2
```

### Phase 2: Accurate Measurement (20-30 minutes)

Get reliable numbers for comparison:

```bash
# Medium test
python benchmark_feature_pipeline.py --num-tickers 30 --runs 3 --benchmark-spy
```

### Phase 3: Full Validation (1+ hour)

Test with realistic workload:

```bash
# Large test (if you have time)
python benchmark_feature_pipeline.py --num-tickers 100 --runs 5
```

## Tips for Accurate Benchmarking

1. **Close other applications** to reduce system load variance
2. **Use `--full` flag** (automatic in benchmark) to avoid cache effects
3. **Run multiple times** and average (handled by `--runs`)
4. **Test same number of tickers** before and after for fair comparison
5. **Check logs** to verify SPY data is loaded once (not per worker)

## Troubleshooting

### "No Parquet files found"
- Make sure `data/clean` directory exists and has `.parquet` files
- Run data cleaning step first if needed

### "Config file not found"
- Check that feature set config exists in `config/` directory
- Use `--feature-set v1` for default

### High variance in results
- System might be under load
- Try running with fewer tickers or more runs
- Close other applications

### No improvement shown
- Verify you're on the optimization branch
- Check that SPY data is being loaded (see logs)
- Make sure you're comparing same number of tickers

## Expected Results

### Optimization 1.1: Pre-load SPY Data

**Expected improvement:**
- 5-10% overall speedup
- More noticeable with more workers/parallelism
- SPY loading should happen once (check logs)

**How to verify it's working:**
1. Check logs for "Loading SPY data for market context features..." (should appear once)
2. Check that market context features still compute correctly
3. Compare benchmark results before/after

## Next Steps

After validating optimization 1.1, you can:
1. Move to optimization 1.2 (Batch Size)
2. Move to optimization 1.3 (PyArrow Engine)
3. Move to optimization 1.4 (Debug Logging)
4. Test all Phase 1 optimizations together

Each optimization should be tested individually first, then together.
