# Phase 3 Testing Guide: Feature Set Isolation

This guide walks you through testing the feature set isolation implementation to verify everything works correctly with the v1 feature set.

## Quick Start (5 Minutes)

If you just want to verify everything works quickly:

```bash
# 1. Navigate to project directory
cd c:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2

# 2. Run automated test
python test_phase3_comprehensive.py

# 3. Test feature pipeline (if you have data)
python src/swing_trade_app.py features --feature-set v1

# 4. Open GUI
python gui/app.py
```

If all of these work, Phase 3 is complete! ✅

---

## Prerequisites

- Python environment set up
- Data already downloaded (at least some tickers in `data/clean/`)
- GUI dependencies installed: `pip install PyQt6` (if testing GUI)

## Quick Verification Test

First, run the automated test script to verify basic functionality:

```bash
cd c:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2
python test_phase3_comprehensive.py
```

This should show all tests passing. If any fail, fix those issues first.

---

## Step-by-Step Manual Testing

### Step 1: Test Feature Pipeline with v1

**Purpose:** Verify that features can be built using the new v1 feature set structure.

**Command:**
```bash
cd c:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2
python src/swing_trade_app.py features --feature-set v1
```

**What to check:**
1. ✅ Command runs without errors
2. ✅ Output shows: "Using feature set: v1"
3. ✅ Output shows correct config path: `config/features_v1.yaml`
4. ✅ Output shows correct output directory: `data/features_labeled_v1/`
5. ✅ Features are computed successfully
6. ✅ Feature files are created in `data/features_labeled_v1/` (not `data/features_labeled/`)

**Expected output:**
```
Using feature set: v1
  Config: C:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2\config\features_v1.yaml
  Output: C:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2\data\features_labeled_v1
```

**If you want to force rebuild all features:**
```bash
python src/swing_trade_app.py features --feature-set v1 --full
```

---

### Step 2: Test Model Training with v1

**Purpose:** Verify that models can be trained using v1 features and that the model is saved with correct feature set metadata.

**Command:**
```bash
python src/swing_trade_app.py train --feature-set v1 --horizon 30 --return-threshold 0.15
```

**What to check:**
1. ✅ Command runs without errors
2. ✅ Output shows: "=== USING FEATURE SET: v1 ==="
3. ✅ Output shows correct data directory: `data/features_labeled_v1/`
4. ✅ Output shows correct train config: `config/train_features_v1.yaml`
5. ✅ Model trains successfully
6. ✅ Model is saved with feature set in name: `models/xgb_classifier_selected_features_v1.pkl`
7. ✅ Training metadata includes feature set information

**Expected output:**
```
=== USING FEATURE SET: v1 ===
Data directory: C:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2\data\features_labeled_v1
Train config: C:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2\config\train_features_v1.yaml
Model output: C:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2\models\xgb_classifier_selected_features_v1.pkl
======================================================================
```

**Verify model file:**
```bash
# Check that model file exists
dir models\xgb_classifier_selected_features_v1.pkl
```

**Check training metadata:**
The model pickle file should contain metadata that includes the feature set. You can verify this by checking the training metadata JSON file if it's created.

---

### Step 3: Test Backtesting with v1 Model

**Purpose:** Verify that backtesting works with a model trained on v1 features.

**Command:**
```bash
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --model models/xgb_classifier_selected_features_v1.pkl --model-threshold 0.5 --position-size 1000
```

**What to check:**
1. ✅ Command runs without errors
2. ✅ Model loads successfully
3. ✅ Backtest completes successfully
4. ✅ Results are reasonable (not all zeros or errors)
5. ✅ Model uses features from v1 feature set

**Expected output:**
```
Loading model from: models/xgb_classifier_selected_features_v1.pkl
...
Backtest Results:
  Total Trades: ...
  Win Rate: ...
  ...
```

---

### Step 4: Test GUI

**Purpose:** Verify that the GUI works with the v1 feature set.

#### Opening the GUI

**Method 1: Direct Python execution (Recommended)**
```bash
cd c:\Users\brand\Documents\Stocks\SwingTradeApp\SwingTradeV2
python gui/app.py
```

**Method 2: Using the batch file (if it exists)**
```bash
# Check if there's a batch file
dir *.bat
# If there's a run_gui.bat or similar, use it
```

**Method 3: Using the main app (if it has GUI option)**
```bash
python src/swing_trade_app.py gui
# (if this command exists)
```

#### Testing GUI Features

Once the GUI opens, test the following:

**1. Feature Engineering Tab:**
- ✅ Tab opens without errors
- ✅ Can see feature list
- ✅ Can build features (if button exists)
- ✅ Verify it uses v1 feature set (check any status messages or logs)

**2. Model Training Tab:**
- ✅ Tab opens without errors
- ✅ Can select/see feature set (if selector exists)
- ✅ Can start training
- ✅ Training uses v1 features
- ✅ Model saves with correct name

**3. Backtesting Tab:**
- ✅ Tab opens without errors
- ✅ Can load model: `models/xgb_classifier_selected_features_v1.pkl`
- ✅ Can run backtest
- ✅ Results display correctly

**4. Trade Identification Tab:**
- ✅ Tab opens without errors
- ✅ Can identify trades
- ✅ Uses correct model/features

**Note:** The GUI may not yet have a feature set selector UI (that's Phase 6). For now, it should default to v1 or use the feature set manager to determine the default.

---

## Verification Checklist

After running all tests, verify:

### File Structure
- [ ] `features/sets/v1/technical.py` exists
- [ ] `features/sets/v1/registry.py` exists
- [ ] `config/features_v1.yaml` exists
- [ ] `config/train_features_v1.yaml` exists
- [ ] `config/feature_sets_metadata.yaml` exists
- [ ] `data/features_labeled_v1/` directory exists with feature files

### Old Files (Should Still Exist - Will Delete in Phase 4)
- [ ] `features/technical.py` still exists (will delete later)
- [ ] `features/registry.py` still exists (will delete later)
- [ ] `data/features_labeled/` still exists (will migrate/delete later)

### Functionality
- [ ] Feature pipeline works with `--feature-set v1`
- [ ] Model training works with `--feature-set v1`
- [ ] Backtesting works with v1 model
- [ ] GUI opens and basic functionality works
- [ ] No import errors in any scripts

---

## Troubleshooting

### Issue: "Feature set 'v1' does not exist"
**Solution:** Check that `config/feature_sets_metadata.yaml` exists and has v1 entry.

### Issue: Import errors
**Solution:** 
1. Make sure you're in the project root directory
2. Check that `features/sets/v1/` directory exists
3. Run `python test_phase3_comprehensive.py` to diagnose

### Issue: GUI won't open
**Solution:**
1. Check that GUI dependencies are installed: `pip install PyQt6` (this project uses PyQt6)
2. Check for error messages in console
3. Try running from project root: `python gui/app.py`
4. If PyQt6 is not installed: `pip install PyQt6`

### Issue: Model training fails
**Solution:**
1. Make sure features are built first: `python src/swing_trade_app.py features --feature-set v1`
2. Check that `data/features_labeled_v1/` has parquet files
3. Verify `config/train_features_v1.yaml` exists

### Issue: Backtest can't find model
**Solution:**
1. Make sure model was trained: `python src/swing_trade_app.py train --feature-set v1`
2. Check model file exists: `dir models\xgb_classifier_selected_features_v1.pkl`
3. Use full path if needed: `--model models/xgb_classifier_selected_features_v1.pkl`

---

## Expected Results Summary

After completing all tests, you should have:

1. ✅ Features built in `data/features_labeled_v1/`
2. ✅ Model trained and saved as `models/xgb_classifier_selected_features_v1.pkl`
3. ✅ Backtest results showing trades and performance
4. ✅ GUI opens and basic tabs work
5. ✅ No errors related to feature set isolation

---

## Next Steps After Testing

Once all tests pass:

1. **Phase 3 Complete** ✅ - All functionality verified
2. **Phase 4** - Cleanup old files (`features/technical.py`, `features/registry.py`, etc.)
3. **Phase 5** - Feature set management tools
4. **Phase 6** - GUI integration with feature set selector
5. **Phase 7** - Documentation updates

---

## Quick Test Commands Summary

```bash
# 1. Quick verification
python test_phase3_comprehensive.py

# 2. Build features
python src/swing_trade_app.py features --feature-set v1

# 3. Train model
python src/swing_trade_app.py train --feature-set v1 --horizon 30 --return-threshold 0.15

# 4. Run backtest
python src/swing_trade_app.py backtest --horizon 30 --return-threshold 0.15 --model models/xgb_classifier_selected_features_v1.pkl --model-threshold 0.5

# 5. Open GUI
python gui/app.py
```

---

## Questions?

If you encounter any issues during testing, check:
1. The error message details
2. That all required files exist (see File Structure checklist)
3. That you're running commands from the project root directory
4. The `test_phase3_comprehensive.py` output for diagnostic info
