# Feature Redundancy Guide

This document explains redundant features in the feature set and provides recommendations on when to use each variant.

## Overview

Some features are highly correlated or derived from the same base indicator. This is intentional - different variants can capture different aspects of the same signal. However, for model efficiency, you may want to select a subset.

## Redundant Feature Groups

### 1. Moving Averages (Trend Indicators)

**Features:**
- `sma_5`, `ema_5` - 5-day averages
- `sma_10`, `ema_10` - 10-day averages  
- `sma_50`, `ema_50` - 50-day averages
- `close_vs_ma10`, `close_vs_ma20` - Price relative to MAs

**Correlation:** High correlation between SMA and EMA of the same period.

**Recommendations:**
- **For trend following:** Use `ema_10` and `ema_50` (EMAs react faster)
- **For smoothing:** Use `sma_50` (less noise)
- **For price position:** Use `close_vs_ma20` (more stable than MA10)
- **Consider removing:** `sma_5` and `ema_5` if you have `sma_10`/`ema_10` (very similar)

### 2. OBV Variants (Volume Indicators)

**Features:**
- `obv` - Raw On-Balance Volume (cumulative, unbounded)
- `obv_pct` - Percent change in OBV (rate of change)
- `obv_z20` - Z-score of OBV (normalized)

**Correlation:** `obv_pct` and `obv_z20` are derived from `obv`, but capture different signals.

**Recommendations:**
- **For volume trend:** Use `obv` (raw cumulative signal)
- **For volume momentum:** Use `obv_pct` (rate of change)
- **For normalized volume:** Use `obv_z20` (statistically normalized)
- **Consider keeping all:** They capture different aspects (level, change, normalized)

### 3. MACD Variants (Momentum Indicators)

**Features:**
- `macd_line` - Raw MACD line (EMA12 - EMA26)
- `macd_histogram` - MACD - Signal line
- `macd_cross_signal` - Crossover events (-1, 0, 1)

**Correlation:** `macd_histogram` and `macd_cross_signal` are derived from `macd_line`.

**Recommendations:**
- **For momentum strength:** Use `macd_histogram` (most informative)
- **For crossover signals:** Use `macd_cross_signal` (discrete events)
- **Consider removing:** `macd_line` if you have `macd_histogram` (histogram is more useful)

### 4. Stochastic Variants (Momentum Indicators)

**Features:**
- `stoch_k` - %K line (0-100)
- `stoch_d` - %D line (smoothed %K)
- `stoch_cross` - Crossover signal (-1, 0, 1)

**Correlation:** `stoch_d` is derived from `stoch_k`, `stoch_cross` from both.

**Recommendations:**
- **For momentum level:** Use `stoch_k` or `stoch_d` (one is enough)
- **For crossover events:** Use `stoch_cross` (discrete signals)
- **Consider removing:** `stoch_d` if you have `stoch_k` (or vice versa)

### 5. Price Action - Returns

**Features:**
- `log_return_1d` - 1-day log return
- `log_return_5d` - 5-day log return

**Correlation:** Low correlation (different timeframes).

**Recommendations:**
- **Keep both:** They capture different timeframes
- **Consider:** Adding more timeframes (3d, 10d) if needed

### 6. Price Position Features

**Features:**
- `close_vs_ma10`, `close_vs_ma20` - Price vs moving averages
- `close_zscore_20` - Z-score of price
- `price_percentile_20d` - Percentile rank

**Correlation:** Moderate correlation (all measure price position).

**Recommendations:**
- **For MA-based signals:** Use `close_vs_ma20` (more stable)
- **For statistical position:** Use `close_zscore_20` (normalized)
- **For percentile rank:** Use `price_percentile_20d` (0-1 range)
- **Consider keeping 1-2:** They're similar but normalized differently

### 7. Ichimoku Components

**Features:**
- `ichimoku_conversion` - Conversion line
- `ichimoku_base` - Base line
- `ichimoku_lead_span_a` - Leading span A (modified, no lookahead)
- `ichimoku_lead_span_b` - Leading span B (modified, no lookahead)
- `ichimoku_lagging_span` - Lagging span (use with caution)

**Correlation:** All derived from price action, but serve different purposes.

**Recommendations:**
- **For trend signals:** Use `ichimoku_conversion` and `ichimoku_base`
- **For support/resistance:** Use `ichimoku_lead_span_a` and `ichimoku_lead_span_b`
- **Avoid in ML:** `ichimoku_lagging_span` (uses past data shifted forward)

## Feature Selection Strategy

### Conservative (Fewer Features)
Remove:
- `sma_5`, `ema_5` (keep 10-day and 50-day)
- `macd_line` (keep `macd_histogram`)
- `stoch_d` (keep `stoch_k` or vice versa)
- `close_vs_ma10` (keep `close_vs_ma20`)

### Balanced (Recommended)
Keep:
- One MA variant per timeframe (EMA preferred for responsiveness)
- All OBV variants (capture different signals)
- `macd_histogram` and `macd_cross_signal`
- `stoch_k` and `stoch_cross`
- One price position feature (`close_zscore_20` or `price_percentile_20d`)

### Comprehensive (All Features)
Keep all features and let the model select:
- Useful for feature importance analysis
- Model can learn which variants are most predictive
- More features = more training time, but better exploration

## Notes

1. **Feature Selection:** Use model-based feature selection (e.g., XGBoost feature importance) to identify which redundant features are actually useful.

2. **Correlation Analysis:** Run correlation analysis on your data to identify highly correlated features (>0.95) that might be redundant.

3. **Domain Knowledge:** Some "redundant" features might capture subtle differences that are important for your specific use case.

4. **Model Performance:** Test with different feature subsets to see what works best for your specific trading strategy.

