# Stop-Loss Analysis Feature - Complete Test Plan

## Prerequisites

Before testing, ensure you have:
1. ✅ Feature data built (run feature engineering)
2. ✅ At least one backtest CSV file in `data/backtest_results/`
3. ✅ The backtest CSV should contain trades with various exit reasons (stop_loss, target_reached, etc.)

---

## Test 1: Basic Tab Access & UI

### Steps:
1. Launch the GUI: `python run_gui.py`
2. Navigate to **Analysis** tab
3. Click on **Stop-Loss Analysis** sub-tab

### Expected Results:
- ✅ Tab appears and is accessible
- ✅ UI shows:
  - Input section with "Backtest CSV:" field and "Change..." button
  - "Analyze Stop-Losses" button (initially disabled or enabled)
  - Summary cards (Total Trades, Stop-Losses, Winners, Target Reached) showing "0"
  - Feature comparison table (empty)
  - Recommendations panel (showing "Run analysis to generate recommendations")
  - Charts section with 3 tabs (Timing, Holding Period, Returns)
  - Export section with 3 buttons

---

## Test 2: Load Backtest CSV

### Steps:
1. In Stop-Loss Analysis tab, click **"Change..."** button
2. Navigate to `data/backtest_results/`
3. Select a backtest CSV file (one that has trades)
4. Click "Open"

### Expected Results:
- ✅ CSV path appears in the "Backtest CSV:" field
- ✅ Status label shows "Loaded: [filename]"
- ✅ "Analyze Stop-Losses" button becomes enabled (if not already)

---

## Test 3: Run Analysis (Phase 1)

### Steps:
1. Click **"Analyze Stop-Losses"** button
2. Watch the progress bar and status messages

### Expected Results:
- ✅ Progress bar appears and shows progress (0% to 100%)
- ✅ Status messages update:
  - "Loading backtest CSV..."
  - "Loaded X trades. Extracting features..."
  - Progress updates during feature extraction
  - "Running stop-loss analysis..."
  - "Analysis complete: X stop-losses found"
- ✅ Progress bar disappears when complete
- ✅ Status shows success message in green

### Verify Summary Cards:
- ✅ **Total Trades** shows correct count
- ✅ **Stop-Losses** shows count and percentage (e.g., "308 (25%)")
- ✅ **Winners** shows count and percentage
- ✅ **Target Reached** shows count and percentage

### Verify Feature Comparison Table:
- ✅ Table populates with features
- ✅ Columns: Feature, Stop-Loss Mean, Winner Mean, Difference, Effect Size
- ✅ Can sort by clicking column headers
- ✅ Numbers are formatted correctly

---

## Test 4: Recommendations Display (Phase 2)

### Steps:
1. After analysis completes, scroll to **Recommendations** section

### Expected Results:
- ✅ Summary label shows count (e.g., "Showing: 10 recommendations")
- ✅ Recommendations grouped into sections:
  - **Strong Recommendations** (effect size > 0.5) - visible
  - **Moderate Recommendations** (0.3-0.5) - visible
  - **Weak Recommendations** (0.2-0.3) - collapsed by default
- ✅ Each recommendation shows:
  - Checkbox (unchecked)
  - Description (e.g., "Filter: rsi14 > 0.2 (effect size: 0.65)")
  - Info button (ℹ)
  - Preview Impact button

### Test Info Buttons:
1. Click an **Info button (ℹ)** next to any recommendation

### Expected Results:
- ✅ Dialog opens showing:
  - Feature name in readable format
  - Description of the feature
  - Interpretation guidance for normalized values
- ✅ Dialog is centered and styled correctly
- ✅ Can close dialog

---

## Test 5: Recommendation Selection & Impact Preview (Phase 2)

### Steps:
1. Check 2-3 recommendation checkboxes
2. Observe the **Impact Preview** panel

### Expected Results:
- ✅ Impact Preview panel appears and expands automatically
- ✅ Shows:
  - "Selected filters would:"
  - Percentage of stop-loss trades excluded
  - Percentage of winning trades excluded
  - Estimated new stop-loss rate with improvement indicator (↓ or ↑)
  - Estimated total trades remaining
- ✅ If warnings apply (50% trades or 20% winners excluded), warnings appear in orange

### Test Select All / Deselect All:
1. Click **"Select All"** button
2. Verify all checkboxes are checked
3. Verify Impact Preview updates
4. Click **"Deselect All"** button
5. Verify all checkboxes are unchecked
6. Verify Impact Preview hides

---

## Test 6: Immediate Stop-Loss Analysis (Phase 3)

### Steps:
1. Scroll to **"Immediate Stop-Loss Analysis"** section
2. If immediate stops exist (≤1 day), section should be visible
3. Click **"Analyze Immediate Stops"** button

### Expected Results:
- ✅ Summary label updates with immediate stop count
- ✅ Special recommendations appear below
- ✅ Recommendations grouped by Strong/Moderate/Weak
- ✅ **"Include in Main Recommendations"** button works
- ✅ **"Exclude"** button works (if immediate recs were included)

---

## Test 7: Save Preset (Phase 4)

### Steps:
1. Select 3-5 recommendations (checkboxes)
2. Click **"Save Selected as Preset"** button
3. Enter a preset name (e.g., "Test Preset 1")
4. Click "OK"

### Expected Results:
- ✅ Success message appears showing:
  - Preset name
  - File location
  - Filter count
  - Stop-loss rate improvement
- ✅ Preset file created in `data/filter_presets/`
- ✅ File is JSON format and contains:
  - Preset name
  - Created date
  - Filters array
  - Metadata (source_backtest, stop_loss_rate_before/after, etc.)

---

## Test 8: Load Preset in Backtesting Tab (Phase 4)

### Steps:
1. Navigate to **Backtesting** tab
2. Click **"Load Preset ▼"** button
3. Select a preset from the menu (should show recent stop-loss filters at top)
4. Click on a preset

### Expected Results:
- ✅ Success message appears
- ✅ Status label shows "Preset loaded: [name] (X filters)"
- ✅ Filters are loaded into `entry_filters`

### Test Apply Features Dialog:
1. Click **"Apply Features"** button
2. Verify filters are pre-populated in the dialog
3. Click **"Load Preset"** in the dialog
4. Select a different preset

### Expected Results:
- ✅ Dialog shows loaded preset name
- ✅ Filters populate correctly
- ✅ Can modify filters
- ✅ Can clear preset

---

## Test 9: Preset Management (Phase 4)

### Steps:
1. In Backtesting tab, click **"Load Preset ▼"**
2. Click **"Manage Presets..."**
3. In the management dialog:
   - View all presets in table
   - Select a preset
   - Click **"Rename"** - change name
   - Select another preset
   - Click **"Delete"** - confirm deletion

### Expected Results:
- ✅ Management dialog opens with table showing:
  - Name, Source Backtest, Stop-Loss Rate, Filters count, Created date
- ✅ Rename works and persists
- ✅ Delete works with confirmation
- ✅ Changes reflect immediately

---

## Test 10: Run Backtest with Filters (Phase 6)

### Steps:
1. In **Backtesting** tab, ensure filters are loaded (from Test 8)
2. Configure backtest parameters (model, strategy, etc.)
3. Set output filename (or use auto-name)
4. Click **"Run Backtest"**
5. Wait for completion

### Expected Results:
- ✅ Backtest runs with filters applied
- ✅ Output CSV is created
- ✅ **Metadata JSON file** is created alongside CSV (`{backtest_name}_metadata.json`)
- ✅ Metadata file contains:
  - `filter_preset_name` (if preset was used)
  - `filters_applied` array
  - `backtest_settings`
  - `model_name`

---

## Test 11: Filter Tracking in Comparison Tab (Phase 6)

### Steps:
1. Navigate to **Analysis** tab → **Backtest Comparison** sub-tab
2. Click **"Refresh"** button
3. Look at the **"Filters Used"** column

### Expected Results:
- ✅ Table shows "Filters Used" column
- ✅ For backtests with filters:
  - Shows preset name and count (e.g., "SL Analysis - backtest.csv (5 filters)")
  - Or shows filter count if no preset (e.g., "3 filters")
- ✅ For backtests without filters: Shows "None"

### Test Filter Details:
1. **Double-click** on a "Filters Used" cell that has filters
2. Verify dialog opens showing:
  - Preset name (if applicable)
  - List of all filters with feature, operator, value

### Expected Results:
- ✅ Dialog displays correctly
- ✅ All filters are listed
- ✅ Can close dialog

---

## Test 12: Visualizations (Phase 5)

### Steps:
1. Return to **Stop-Loss Analysis** tab
2. After running analysis, check the **Charts** section
3. Click through the 3 tabs: **Timing**, **Holding Period**, **Returns**

### Expected Results:

#### Timing Tab:
- ✅ Two bar charts side-by-side:
  - **Day of Week Distribution** (Mon-Sun)
  - **Month Distribution** (Jan-Dec)
- ✅ Charts are styled with dark theme
- ✅ Data is accurate

#### Holding Period Tab:
- ✅ Histogram showing days to stop-loss
- ✅ Immediate stops (≤1 day) highlighted in **red**
- ✅ Statistics box showing: Mean, Median, Min, Max

#### Returns Tab:
- ✅ Histogram of stop-loss returns (as percentages)
- ✅ Vertical dashed line at -7% (typical stop-loss threshold)
- ✅ Statistics box showing: Mean, Median, Min, Max, Std Dev

---

## Test 13: Export Functionality (Phase 7)

### Steps:

#### Export Full Report:
1. After analysis completes, click **"Export Full Report"**
2. Select format: **HTML**
3. Choose save location and filename
4. Click "Save"

### Expected Results:
- ✅ HTML file is created
- ✅ Success message appears
- ✅ HTML file contains:
  - Summary section with cards
  - Feature comparison table
  - Recommendations table
  - Proper styling (dark theme)

#### Export Recommendations Table:
1. Click **"Export Recommendations Table"**
2. Choose save location
3. Click "Save"

### Expected Results:
- ✅ CSV file is created
- ✅ Contains all recommendations with columns:
  - feature, operator, value, effect_size, cohens_d, category, description
- ✅ Can open in Excel/spreadsheet

#### Export Selected as Preset:
1. Select some recommendations
2. Click **"Export Selected as Preset"**
3. Enter name and save

### Expected Results:
- ✅ Same as Test 7 (Save Preset)
- ✅ Preset is saved correctly

---

## Test 14: Edge Cases & Error Handling

### Test Missing Feature Files:
1. Use a backtest CSV with tickers that don't have feature files
2. Run analysis

### Expected Results:
- ✅ Analysis continues (doesn't crash)
- ✅ Warning messages in log about missing files
- ✅ Trades without features are skipped gracefully

### Test Empty Backtest:
1. Create an empty CSV or use one with no trades
2. Try to run analysis

### Expected Results:
- ✅ Appropriate error message
- ✅ No crash

### Test No Stop-Loss Trades:
1. Use a backtest CSV with only winning trades
2. Run analysis

### Expected Results:
- ✅ Analysis completes
- ✅ Stop-Loss count shows 0
- ✅ Recommendations may be empty or minimal
- ✅ Charts may show "No data available"

### Test Very Large Backtest:
1. Use a backtest with 1000+ trades
2. Run analysis

### Expected Results:
- ✅ Progress bar shows progress
- ✅ Analysis completes (may take a few minutes)
- ✅ All features work correctly

---

## Test 15: Performance & Caching

### Steps:
1. Run analysis on a backtest CSV (Test 3)
2. Wait for completion
3. **Without changing anything**, click **"Analyze Stop-Losses"** again

### Expected Results:
- ✅ Status shows "Using cached features..."
- ✅ Analysis runs faster (skips feature extraction)
- ✅ Results are identical

---

## Test 16: Full Workflow Integration

### Complete End-to-End Test:
1. **Analysis Tab** → **Stop-Loss Analysis**:
   - Load CSV
   - Run analysis
   - Select 5 recommendations
   - Save as preset: "My Test Preset"

2. **Backtesting Tab**:
   - Load the preset you just saved
   - Configure backtest parameters
   - Run backtest with filters
   - Verify output CSV and metadata JSON are created

3. **Analysis Tab** → **Backtest Comparison**:
   - Refresh list
   - Verify your backtest appears
   - Verify "Filters Used" column shows your preset
   - Double-click to view filter details

4. **Backtesting Tab** → **Apply Features**:
   - Click "Apply Features"
   - Verify filters are pre-populated
   - Modify a filter value
   - Apply filters
   - Run another backtest

5. **Analysis Tab** → **Backtest Comparison**:
   - Compare the two backtests
   - Verify both show filters used

---

## Success Criteria Checklist

After completing all tests, verify:

- ✅ All UI elements are visible and functional
- ✅ Analysis runs without errors
- ✅ Summary cards show correct data
- ✅ Feature comparison table populates and sorts
- ✅ Recommendations are grouped correctly
- ✅ Impact preview works accurately
- ✅ Immediate stop analysis works
- ✅ Presets can be saved, loaded, renamed, deleted
- ✅ Filters are tracked in backtest metadata
- ✅ Comparison tab shows filter information
- ✅ All charts display correctly
- ✅ Export functions work (HTML, CSV)
- ✅ Caching improves performance
- ✅ Error handling is graceful
- ✅ Full workflow integrates correctly

---

## Known Limitations / Notes

1. **PDF Export**: Requires `weasyprint` library (`pip install weasyprint`). Falls back to HTML if not available.
2. **Feature Extraction**: May take several minutes for large backtests (1000+ trades).
3. **Caching**: Cache is per-session. Restarting the app clears the cache.

---

## Reporting Issues

If you encounter any issues during testing, note:
- Which test number failed
- Exact steps taken
- Error messages (if any)
- Expected vs actual behavior
- Screenshots (if helpful)

