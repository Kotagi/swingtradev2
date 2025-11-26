# Model Performance Analysis & Improvement Plan

## Current Performance Baseline

**Metrics (Multiplier 1.0, 75 features):**
- **ROC AUC:** 0.7152 (decent but not great)
- **Precision:** 14.89% (85% false positives - **CRITICAL ISSUE**)
- **Recall:** 55.76% (missing ~44% of opportunities)
- **F1 Score:** 0.2591
- **Model Threshold:** 0.5 (default)

**Feature Analysis:**
- **Total Features:** 75 (after pruning from 104)
- **Top Feature:** `weekly_atr_pct` at 29.6% importance (very dominant)
- **Top 3 Features:** Account for ~48% of importance
- **Zero-Importance Features:** 11 features (candlestick patterns, some flags)
- **Low-Importance Features:** Many features < 0.001 importance

---

## Root Cause Analysis

### 1. **High False Positive Rate (85% false positives)**

**Why this is happening:**
- **Default threshold (0.5) is too low** for this imbalanced problem
- **Features lack discriminative power** - many features are correlated with volatility/trend but not with actual price movements
- **No confirmation signals** - model sees volatility/trend patterns but can't distinguish between "volatile but going up" vs "volatile but going sideways/down"
- **Missing market context** - model doesn't know if the stock is outperforming/underperforming the market
- **No time-based context** - model doesn't account for day-of-week, month-end effects, etc.

**Evidence:**
- Top feature (`weekly_atr_pct`) is volatility-based, not directional
- Many features are trend/volatility indicators that don't predict direction
- Zero-importance features (candlestick patterns) suggest pattern recognition isn't working

### 2. **Missing Signals (44% missed opportunities)**

**Why this is happening:**
- **Model is too conservative** - threshold may be too high OR features aren't capturing all patterns
- **Feature gaps** - missing features that would identify certain patterns (e.g., relative strength, momentum confirmation)
- **Limited feature interactions** - model may miss patterns that require multiple features in combination
- **No market regime awareness** - model doesn't adapt to bull/bear/sideways markets

**Evidence:**
- Recall of 55.76% means model is missing nearly half of profitable opportunities
- Many important patterns may require feature combinations not currently captured

### 3. **Limited Model Discriminative Power**

**Why this is happening:**
- **Feature quality** - many features are noisy or redundant
- **No market context** - can't distinguish stock-specific vs market-wide moves
- **No confirmation signals** - single-feature patterns are weak
- **Limited feature engineering** - no interactions, transforms, or advanced patterns

**Evidence:**
- ROC AUC of 0.7152 is decent but not great (0.8+ would be ideal)
- Top feature dominance (29.6%) suggests over-reliance on one signal
- Many zero-importance features suggest feature quality issues

---

## Improvement Strategy (Prioritized by ROI)

### **🔥 PRIORITY 1: Immediate Wins (Highest ROI)**

#### 1.1 Threshold Tuning (30 minutes, High Impact)
**Problem:** Default 0.5 threshold is likely suboptimal for precision/recall tradeoff.

**Solution:**
- Find optimal threshold using validation set
- Use precision-recall curve to find threshold that maximizes F1 or precision at acceptable recall
- Test thresholds: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

**Expected Impact:**
- Precision: 14.89% → 20-25% (with threshold ~0.6-0.7)
- Recall: 55.76% → 40-50% (acceptable tradeoff)
- **ROI: Very High** - Quick win, no retraining needed

**Implementation:**
```python
# Add to train_model.py or create threshold_tuning.py
from sklearn.metrics import precision_recall_curve, f1_score

# Find optimal threshold on validation set
y_val_proba = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

# Find threshold that maximizes F1
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Or find threshold for target precision (e.g., 20%)
target_precision = 0.20
optimal_idx = np.where(precisions >= target_precision)[0][0]
optimal_threshold = thresholds[optimal_idx]
```

#### 1.2 Add Market Context Features (2-3 hours, High Impact)
**Problem:** Model doesn't know if stock is outperforming/underperforming the market.

**Solution:**
- Download SPY data (market index)
- Add relative strength features:
  - `relative_strength_spy_5d` - Stock return vs SPY return (5-day)
  - `relative_strength_spy_20d` - Stock return vs SPY return (20-day)
  - `rs_rank_20d` - Relative strength rank (0-100) vs market
  - `outperformance_flag` - Binary: outperforming market
  - `market_correlation_20d` - Rolling correlation to SPY

**Expected Impact:**
- Precision: +3-5% (better filtering of false positives)
- Recall: +2-3% (captures more true signals)
- **ROI: Very High** - Market context is critical for swing trading

**Implementation:**
- Add SPY download to `download_data.py`
- Add relative strength features to `features/technical.py`
- Update `features/registry.py` and `config/features.yaml`

#### 1.3 Add Time-Based Features (1 hour, Medium-High Impact)
**Problem:** Model doesn't account for day-of-week, month-end effects, etc.

**Solution:**
- Add cyclical time features:
  - `day_of_week` - Monday=0, Friday=4 (cyclical encoding: sin/cos)
  - `day_of_month` - 1-31 (cyclical encoding)
  - `month_of_year` - 1-12 (cyclical encoding)
  - `is_month_end` - Binary: last 3 days of month
  - `is_quarter_end` - Binary: last week of quarter

**Expected Impact:**
- Precision: +1-2% (filters out bad days)
- Recall: +1-2% (captures time-based patterns)
- **ROI: High** - Quick to implement, captures known market patterns

**Implementation:**
- Add time features to `features/technical.py`
- Use `pd.DatetimeIndex` to extract time components
- Apply cyclical encoding (sin/cos) for day_of_week, day_of_month, month_of_year

---

### **⚡ PRIORITY 2: Feature Engineering (Medium ROI)**

#### 2.1 Feature Interactions (2-3 hours, Medium Impact)
**Problem:** Model may miss patterns that require multiple features in combination.

**Solution:**
- Create top 5-10 feature interactions:
  - `weekly_atr_pct × volatility_regime` - Volatility context
  - `distance_to_resistance × price_vs_all_mas` - Trend + resistance
  - `macd_histogram × rsi` - Momentum confirmation
  - `volume_weighted_price × volume_avg_ratio_5d` - Volume context
  - `atr_pct_of_price × daily_range_pct` - Volatility consistency
  - `rsi × relative_strength_spy` - Momentum + market context (if added)

**Expected Impact:**
- Precision: +2-3%
- Recall: +2-3%
- **ROI: Medium-High** - Captures non-linear relationships

**Implementation:**
- Multiply/divide feature pairs (after scaling)
- Add to `features/technical.py` as interaction features
- Test on validation set to ensure they improve performance

#### 2.2 Remove More Zero-Importance Features (30 minutes, Low-Medium Impact)
**Problem:** Still have 11 zero-importance features adding noise.

**Solution:**
- Remove all features with importance < 0.0001
- Remove features that are highly correlated (>0.95) with more important features
- Current zero-importance features:
  - `log_return_1d`, `macd_cross_signal`, `stoch_cross`
  - `bullish_engulfing`, `bearish_engulfing`, `hammer_signal`
  - `long_legged_doji`, `morning_star`
  - `price_near_resistance`, `price_near_support`
  - `bb_squeeze`, `bb_expansion`, `high_volatility_flag`
  - `ema_alignment`, `volume_climax`, `volume_dry_up`

**Expected Impact:**
- Training speed: +10-15%
- Model clarity: Better (less noise)
- **ROI: Medium** - Reduces noise, may slightly improve performance

**Implementation:**
- Run `analyze_features.py` again
- Update `config/train_features.yaml` to disable zero-importance features
- Retrain and compare

#### 2.3 Improve Existing Features (3-4 hours, Medium Impact)
**Problem:** Some features could be more informative.

**Solution:**
- **Support/Resistance:** Add strength metric (how many times touched, how long held)
- **Volatility regime:** Add regime duration (how long in current regime)
- **Trend strength:** Add trend quality (consistency + strength combined)
- **Volume features:** Add volume confirmation (volume on up days vs down days)

**Expected Impact:**
- Precision: +1-2%
- Recall: +1-2%
- **ROI: Medium** - Better signal quality from existing features

---

### **🚀 PRIORITY 3: Model Improvements (Lower ROI, but Important)**

#### 3.1 Model Calibration (1 hour, Medium Impact)
**Problem:** Model probabilities may not be well-calibrated.

**Solution:**
- Use `CalibratedClassifierCV` to calibrate probabilities
- Or use Platt scaling / isotonic regression
- This ensures predicted probabilities match actual frequencies

**Expected Impact:**
- Better threshold selection
- More reliable probability estimates
- **ROI: Medium** - Improves threshold tuning effectiveness

#### 3.2 Hyperparameter Tuning (2-3 hours, Medium Impact)
**Problem:** Current hyperparameters may not be optimal.

**Solution:**
- Tune with focus on precision/recall tradeoff:
  - Increase `min_child_weight` (reduce overfitting)
  - Adjust `scale_pos_weight` (balance precision/recall)
  - Tune `max_depth` (prevent overfitting)
  - Adjust `reg_alpha` and `reg_lambda` (regularization)

**Expected Impact:**
- Precision: +1-2%
- Recall: +1-2%
- **ROI: Medium** - Incremental improvement

#### 3.3 Ensemble Methods (4-5 hours, Medium Impact)
**Problem:** Single model may have limitations.

**Solution:**
- Train multiple models with different:
  - Feature subsets
  - Hyperparameters
  - Random seeds
- Ensemble predictions (voting or weighted average)

**Expected Impact:**
- Precision: +2-3%
- Recall: +2-3%
- **ROI: Medium** - More complex, but can improve robustness

---

## Recommended Implementation Order

### **Week 1: Quick Wins (Highest ROI)**
1. ✅ **Threshold Tuning** (30 min) - Find optimal threshold
2. ✅ **Add Market Context Features** (2-3 hours) - SPY relative strength
3. ✅ **Add Time-Based Features** (1 hour) - Day of week, month effects
4. ✅ **Retrain & Test** (1 hour) - Evaluate improvements

**Expected Outcome:**
- Precision: 14.89% → 22-28%
- Recall: 55.76% → 50-55%
- ROC AUC: 0.7152 → 0.73-0.75

### **Week 2: Feature Engineering**
1. ✅ **Feature Interactions** (2-3 hours) - Top 5-10 interactions
2. ✅ **Remove Zero-Importance Features** (30 min) - Clean up noise
3. ✅ **Improve Existing Features** (3-4 hours) - Better signal quality
4. ✅ **Retrain & Test** (1 hour) - Evaluate improvements

**Expected Outcome:**
- Precision: 22-28% → 25-30%
- Recall: 50-55% → 52-57%
- ROC AUC: 0.73-0.75 → 0.75-0.77

### **Week 3: Model Improvements**
1. ✅ **Model Calibration** (1 hour) - Better probability estimates
2. ✅ **Hyperparameter Tuning** (2-3 hours) - Optimize for precision/recall
3. ✅ **Final Evaluation** (1 hour) - Compare all improvements

**Expected Outcome:**
- Precision: 25-30% → 28-32%
- Recall: 52-57% → 54-58%
- ROC AUC: 0.75-0.77 → 0.77-0.80

---

## Success Metrics

### **Target Improvements:**
- **Precision:** 14.89% → **25-30%** (reduce false positives by 50-60%)
- **Recall:** 55.76% → **54-58%** (maintain or slightly improve)
- **F1 Score:** 0.2591 → **0.35-0.40** (balanced improvement)
- **ROC AUC:** 0.7152 → **0.77-0.80** (better discriminative power)

### **Backtest Targets:**
- **Win Rate:** 49% → **55-60%**
- **Sharpe Ratio:** 1.09 → **1.3-1.5**
- **Trade Count:** Reduce by 20-30% (fewer false positives)
- **Profit Factor:** Maintain or improve

---

## Next Steps

1. **Start with Threshold Tuning** - Quick win, no retraining needed
2. **Add Market Context Features** - Highest impact feature addition
3. **Add Time-Based Features** - Quick to implement, known patterns
4. **Retrain and Evaluate** - Compare before/after metrics
5. **Iterate** - Continue with feature engineering and model improvements

---

## Notes

- **Threshold tuning is the fastest win** - Can improve precision immediately without retraining
- **Market context is critical** - Swing trading requires knowing if stock is outperforming market
- **Time-based features capture known patterns** - Day-of-week, month-end effects are well-documented
- **Feature interactions capture non-linear relationships** - XGBoost can find these, but explicit interactions help
- **Model calibration improves threshold selection** - Better probability estimates = better threshold tuning

