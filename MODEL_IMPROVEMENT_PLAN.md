# Model Improvement Plan

## Current Issues Identified

### 1. **Data Split Problems** (CRITICAL)
- **Current**: Train (2008-2022), Val (2023), Test (2024-2025)
- **Problems**:
  - Training on 15 years of data including multiple market regimes (2008 crisis, COVID crash)
  - Validation set too small (1 year)
  - Test set is very recent (may not represent future)
  - Model may be learning outdated patterns

### 2. **Class Imbalance** (HIGH PRIORITY)
- **Current**: ~10% positive class (90% negative)
- **Problem**: Model biased toward predicting "no trade"
- **Current handling**: `scale_pos_weight = neg/pos` (~9:1 ratio)
- **Issue**: May need more aggressive balancing or different approach

### 3. **Feature Quality** (MEDIUM PRIORITY)
- **Current**: 105 features, many may be redundant/noisy
- **Top feature**: `atr_pct_of_price` has 37.92% importance (very dominant)
- **Issue**: Many features may be adding noise without signal

### 4. **Win Rate** (TARGET)
- **Current**: 52.91% (barely above random)
- **Target**: >60% for profitable trading
- **Issue**: Model not discriminative enough

## Recommended Strategy (Priority Order)

### Phase 1: Fix Data Splits (DO THIS FIRST)

**Option A: More Recent Training Data (Recommended)**
- Train: 2015-2022 (7 years, more recent patterns)
- Validation: 2023 (1 year)
- Test: 2024-2025 (recent data)

**Option B: Rolling Window Approach**
- Train: 2018-2022 (5 years)
- Validation: 2023 (1 year)
- Test: 2024-2025

**Option C: Equal Split**
- Train: 2015-2020 (6 years)
- Validation: 2021-2022 (2 years)
- Test: 2023-2025 (3 years)

**Why**: Recent market conditions (post-COVID, high volatility, AI boom) may be more relevant than 2008-2014 patterns.

### Phase 2: Improve Class Imbalance Handling

**Current**: `scale_pos_weight = neg/pos` (~9:1)

**Options**:
1. **Increase scale_pos_weight**: Try 2x or 3x the ratio to favor positive class
2. **SMOTE/Undersampling**: Balance classes before training
3. **Focal Loss**: Use custom loss function that focuses on hard examples
4. **Threshold Tuning**: Optimize prediction threshold (not just 0.5)

**Recommendation**: Start with increasing `scale_pos_weight` to 15-20 (2x current), then tune threshold.

### Phase 3: Feature Selection/Pruning

**Strategy**:
1. **Remove low-importance features**: Drop features with <0.1% importance
2. **Correlation analysis**: Remove highly correlated features (>0.95)
3. **Recursive Feature Elimination**: Use RFE to find optimal feature set
4. **Focus on top categories**: Volatility, Multi-timeframe, Support/Resistance

**Expected Impact**: Reduce noise, improve generalization, faster training

### Phase 4: Experiment with Timeframes

**Current**: 5-day horizon, 5% threshold

**Options to Test**:
- **Longer horizons**: 10-day, 20-day, 30-day (may capture longer trends)
- **Higher thresholds**: 7.5%, 10% (fewer but higher quality signals)
- **Multiple horizons**: Train separate models for different timeframes

**Why**: Longer timeframes may have more predictable patterns, higher thresholds reduce noise.

### Phase 5: Add Non-Technical Features (LATER)

**Fundamental Features** (requires additional data):
- P/E ratio, P/B ratio
- Earnings growth
- Revenue growth
- Sector performance
- Market cap

**Market Features**:
- VIX (volatility index)
- Sector rotation
- Market regime indicators

**Why Later**: Technical features should be optimized first, then add fundamentals as enhancement.

## Implementation Order

1. ✅ **Fix data splits** - Use more recent training data (2015-2022)
2. ✅ **Improve class imbalance** - Increase scale_pos_weight, tune threshold
3. ✅ **Feature pruning** - Remove low-importance features
4. ⏸️ **Test longer timeframes** - Try 10-day, 20-day horizons
5. ⏸️ **Add fundamentals** - After technical features optimized

## Expected Outcomes

- **Win Rate**: 52.91% → 58-65% (target)
- **Sharpe Ratio**: 0.35 → >1.0 (target)
- **Trade Count**: Reduce from 784k to 50k-200k (higher quality)
- **Average P&L**: Increase from $4.20 to $10-20 per trade

## Next Steps

1. Implement improved data splits
2. Add class imbalance tuning options
3. Add feature selection/pruning
4. Test and compare results

