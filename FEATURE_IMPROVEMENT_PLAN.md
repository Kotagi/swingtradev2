# Feature Improvement Plan

## Current Status
- **Total Features:** 105
- **Top 20 Features:** Account for ~58% of importance
- **Top Feature Dominance:** `weekly_atr_pct` has 22% importance (very high)
- **Model Performance:** 49% win rate, 1.09 Sharpe (needs improvement)

## Strategy: Quality Over Quantity

Instead of adding more features, we'll:
1. **Analyze** current features (importance, correlation, redundancy)
2. **Prune** low-value and redundant features
3. **Improve** existing features (interactions, transforms, better engineering)
4. **Add** strategic new features (targeted, not just more of the same)

---

## Phase 1: Feature Analysis & Pruning (DO THIS FIRST)

### Step 1.1: Analyze Feature Importance Distribution

**Goal:** Identify which features are actually contributing vs just adding noise.

**Actions:**
1. Extract full feature importance list from training metadata
2. Categorize features by importance:
   - **High importance:** > 0.01 (top contributors)
   - **Medium importance:** 0.001 - 0.01 (some value)
   - **Low importance:** < 0.001 (likely noise)
3. Count how many features fall into each category

**Expected Findings:**
- Likely 20-30 high-importance features
- 30-50 medium-importance features
- 20-30 low-importance features (candidates for removal)

### Step 1.2: Correlation Analysis

**Goal:** Identify redundant features (highly correlated = redundant).

**Actions:**
1. Calculate correlation matrix for all features
2. Identify feature pairs with correlation > 0.95 (highly redundant)
3. For each redundant pair, keep the one with higher importance

**Expected Redundant Pairs:**
- `sma_5` vs `ema_5` (same timeframe)
- `sma_10` vs `ema_10`
- `macd_line` vs `macd_histogram` (histogram is more informative)
- `stoch_k` vs `stoch_d` (D is smoothed K)
- `close_vs_ma10` vs `close_vs_ma20` (similar signal)
- Multiple volatility features (ATR variants)

### Step 1.3: Prune Low-Value Features

**Goal:** Remove features that add noise without signal.

**Pruning Strategy:**
1. **Remove features with importance < 0.001** (bottom 20-30 features)
2. **Remove redundant features** (keep best from each correlated pair)
3. **Remove features with high NaN rates** (>10% missing)
4. **Remove constant/near-constant features** (no variance)

**Expected Reduction:** 105 → 60-70 features (remove 30-45 features)

**Benefits:**
- Faster training
- Less overfitting
- Better generalization
- Clearer signal

---

## Phase 2: Feature Engineering Improvements

### Step 2.1: Feature Interactions

**Goal:** Create interactions between top features to capture non-linear relationships.

**Top Interactions to Create:**
1. `weekly_atr_pct × volatility_regime` - Volatility context
2. `distance_to_resistance × price_vs_all_mas` - Trend + resistance
3. `macd_histogram × rsi` - Momentum confirmation
4. `volume_weighted_price × volume_avg_ratio_5d` - Volume context
5. `atr_pct_of_price × daily_range_pct` - Volatility consistency

**Implementation:** Multiply or divide feature pairs (after scaling)

**Expected Impact:** +5-10 high-value interaction features

### Step 2.2: Feature Transforms

**Goal:** Create alternative representations of existing features.

**Transforms to Add:**
1. **Log transforms** for unbounded features (OBV, volume)
2. **Square root transforms** for volatility features
3. **Percentile ranks** for price-based features (already have some, add more)
4. **Rolling z-scores** for momentum indicators (RSI, MACD)

**Expected Impact:** +5-10 transformed features

### Step 2.3: Improve Existing Features

**Goal:** Make existing features more informative.

**Improvements:**
1. **Weekly features:** Currently using forward-fill - consider using last known value with decay
2. **Support/Resistance:** Add strength metric (how many times touched, how long held)
3. **Volatility regime:** Add regime duration (how long in current regime)
4. **Trend strength:** Add trend quality (consistency + strength combined)

**Expected Impact:** Better signal quality from existing features

---

## Phase 3: Strategic New Features

### Step 3.1: Market Context Features

**Goal:** Add features that capture market-wide context (if data available).

**Features:**
1. **Sector relative strength** - Stock vs sector performance
2. **Market regime** - Bull/bear/sideways (if SPY data available)
3. **Correlation to market** - Rolling correlation to SPY
4. **Sector momentum** - Sector-level momentum indicators

**Data Requirement:** Need sector/market data (may not be available)

### Step 3.2: Time-Based Features

**Goal:** Capture time-of-week, time-of-month patterns.

**Features:**
1. `day_of_week` - Monday effect, Friday effect
2. `day_of_month` - Month-end effects
3. `days_since_earnings` - If earnings data available
4. `month_of_year` - Seasonal patterns

**Expected Impact:** +4-6 time-based features

### Step 3.3: Advanced Price Action

**Goal:** More sophisticated price pattern recognition.

**Features:**
1. **Gap analysis:** Gap size, gap fill probability
2. **Breakout strength:** Volume + price confirmation
3. **Pullback depth:** How far price pulls back from highs
4. **Consolidation detection:** Periods of low volatility before moves

**Expected Impact:** +5-8 advanced price action features

---

## Phase 4: Feature Selection & Validation

### Step 4.1: Recursive Feature Elimination (RFE)

**Goal:** Systematically identify optimal feature subset.

**Process:**
1. Train model with all features
2. Remove lowest-importance feature
3. Retrain and evaluate
4. Repeat until performance degrades
5. Select feature set with best validation performance

**Expected Result:** Optimal feature count (likely 50-70 features)

### Step 4.2: Cross-Validation Feature Importance

**Goal:** Validate feature importance across different time periods.

**Process:**
1. Calculate feature importance for each CV fold
2. Identify features with consistent high importance
3. Remove features with inconsistent importance (overfitting)

**Expected Result:** More robust feature set

---

## Implementation Priority

### **Week 1: Analysis & Pruning** (Highest ROI)
1. ✅ Extract full feature importance list
2. ✅ Calculate correlation matrix
3. ✅ Identify redundant features
4. ✅ Prune low-importance features
5. ✅ Retrain and compare performance

**Expected Outcome:** 105 → 60-70 features, similar or better performance

### **Week 2: Feature Engineering**
1. ✅ Add top 5-10 feature interactions
2. ✅ Add feature transforms
3. ✅ Improve existing features
4. ✅ Retrain and compare

**Expected Outcome:** 60-70 → 70-80 features, improved performance

### **Week 3: Strategic Additions**
1. ✅ Add time-based features
2. ✅ Add advanced price action features
3. ✅ Test market context features (if data available)
4. ✅ Final feature selection (RFE)

**Expected Outcome:** 70-80 → 60-75 optimal features, best performance

---

## Success Metrics

### Target Improvements:
- **Win Rate:** 49% → 52-55%
- **Sharpe Ratio:** 1.09 → 1.3-1.5
- **Precision:** 21% → 25-30%
- **Feature Count:** 105 → 60-75 (optimal subset)

### Validation:
- Compare backtest results before/after each phase
- Use validation set to prevent overfitting
- Track feature importance changes

---

## Tools Needed

1. **Feature Analysis Script:**
   - Extract feature importances
   - Calculate correlations
   - Identify redundant pairs
   - Generate pruning recommendations

2. **Feature Engineering Script:**
   - Create interactions
   - Apply transforms
   - Validate new features

3. **Feature Selection Script:**
   - RFE implementation
   - Cross-validation importance
   - Feature subset evaluation

---

## Next Steps

1. **Create feature analysis script** to extract importances and correlations
2. **Run analysis** on current model
3. **Generate pruning recommendations**
4. **Prune features** and retrain
5. **Compare performance** (backtest)
6. **Iterate** based on results

