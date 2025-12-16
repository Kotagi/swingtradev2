# Stop-Loss Analysis Implementation Guide

## Overview

This document provides complete specifications for implementing the Stop-Loss Analysis feature in the Swing Trading GUI. This is a comprehensive feature that integrates with the Analysis tab, Backtesting tab, and introduces a filter preset system.

**Baseline Commit**: `08401a7` (Pre-stop-loss-analysis: stable baseline)  
**Tag**: `v1.0-pre-sl-analysis`  
**Feature Branch**: `feature/stop-loss-analysis`

---

## Table of Contents

1. [Feature Specification](#feature-specification)
2. [UI Layouts & Components](#ui-layouts--components)
3. [Data Structures](#data-structures)
4. [Service Layer API](#service-layer-api)
5. [Phased Implementation Plan](#phased-implementation-plan)
6. [Design Decisions](#design-decisions)
7. [Integration Points](#integration-points)
8. [Testing Strategy](#testing-strategy)

---

## Feature Specification

### Core Functionality

The Stop-Loss Analysis feature allows users to:
1. Analyze backtest trades to identify patterns in stop-loss exits
2. Compare features between stop-loss trades and winning trades
3. Generate filter recommendations based on statistical analysis
4. Preview the impact of applying recommended filters
5. Save filter sets as reusable presets
6. Load presets in the Backtesting tab for future backtests
7. Track which filters were used in backtest comparisons

### Key Features

- **Statistical Analysis**: Uses Cohen's d (effect size) to identify significant differences
- **Threshold-Based Recommendations**: Default threshold 0.3 (medium effect)
- **Effect Size Buckets**: Strong (>0.5), Moderate (0.3-0.5), Weak (0.2-0.3)
- **User Selection**: Checkboxes for individual recommendation selection
- **Impact Preview**: Applies filters to historical trades to estimate impact
- **Warning System**: Alerts if filters would exclude too many trades
- **Immediate Stop Analysis**: Special analysis for stops ≤1 day
- **Preset Management**: Save, load, delete, rename filter presets
- **Visualizations**: Timing charts, holding period, return distributions

---

## UI Layouts & Components

### Analysis Tab - Stop-Loss Analysis Sub-Tab

```
┌─────────────────────────────────────────────────────────┐
│ Stop-Loss Analysis                                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ [Input Section]                                         │
│   Backtest CSV: [backtest_20240115.csv] [Change...]   │
│   [Analyze Stop-Losses] button                          │
│                                                         │
│ [Summary Cards - 4 cards in row]                       │
│   [Total Trades: 1,234] [Stop-Losses: 308 (25%)]      │
│   [Winners: 612 (50%)] [Target Reached: 314 (25%)]    │
│                                                         │
│ [Analysis Settings - Collapsible]                      │
│   Effect Size Threshold: [0.3] ▼                       │
│   Max Recommendations: [10] (1-20) - Note: No hard max │
│   [Apply Settings]                                      │
│                                                         │
│ [Top Differentiating Features - Table]                 │
│   Sortable: Feature | SL Mean | Winner Mean |          │
│            Difference | Effect Size | Action           │
│   [Export Table]                                        │
│                                                         │
│ [Recommendations Panel]                                │
│   Showing: 10 recommendations (15 total above 0.3)    │
│   [Show All 15] [Show Top 5] [Show Top 10]             │
│                                                         │
│   Strong Recommendations (Effect Size > 0.5):           │
│     ☑ Filter: rsi14 > 0.2 [Info] [Preview Impact]     │
│     ☑ Filter: volatility_ratio < 1.5 [Info] [Preview] │
│                                                         │
│   Moderate Recommendations (0.3 - 0.5):                │
│     ☑ Filter: sma20_ratio > 1.02 [Info] [Preview]      │
│     ☐ Filter: atr14_normalized < 0.05 [Info] [Preview]│
│     ...                                                 │
│                                                         │
│   Weak Recommendations (0.2 - 0.3) - Collapsed:         │
│     [▶ Show 3 weak recommendations]                     │
│                                                         │
│   [Select All] [Deselect All]                          │
│   [Apply Selected] [Save Selected as Preset]            │
│                                                         │
│ [Impact Preview - Expandable]                           │
│   Selected filters would:                               │
│   • Exclude ~45% of stop-loss trades                   │
│   • Exclude ~12% of winning trades                     │
│   • Estimated new stop-loss rate: 18% (down from 25%) │
│   • Estimated total trades: 1,085 (down from 1,234)  │
│   ⚠️ Warning: Excluding 12% of winners may reduce     │
│      overall profitability                             │
│                                                         │
│ [Immediate Stop-Loss Analysis - Collapsible]           │
│   Immediate Stops (≤1 day): 45 (14.6% of stop-losses) │
│   [Analyze Immediate Stops]                             │
│   Special Recommendations:                              │
│     ☑ Filter: rsi14 > 0.15 [Info] [Preview]           │
│   [Include in Main Recommendations] [Exclude]         │
│                                                         │
│ [Charts Section - Tabs]                                │
│   [Timing] [Holding Period] [Returns]                   │
│                                                         │
│   Timing Tab:                                           │
│     • Day of Week Distribution (bar chart)              │
│     • Month Distribution (bar chart)                    │
│                                                         │
│   Holding Period Tab:                                   │
│     • Days to Stop-Loss Histogram                       │
│     • Immediate stops (≤1 day) highlighted             │
│     • Stats: Avg, Median, Min, Max                     │
│                                                         │
│   Returns Tab:                                          │
│     • Stop-Loss Return Distribution                     │
│     • Return vs Stop-Loss Threshold comparison          │
│                                                         │
│ [Action Buttons]                                        │
│   [Export Report] [Save All Recommendations]           │
└─────────────────────────────────────────────────────────┘
```

### Backtesting Tab Enhancement

**Current:**
```
[Apply Features] [Browse...]
```

**New:**
```
[Apply Features] [Load Preset ▼] [Browse...]
```

**"Load Preset" Dropdown Menu:**
```
Recent Stop-Loss Filters:
  • SL Analysis - backtest.csv - 2024-01-15 (SL: 25% → 18%)
  • SL Analysis - backtest.csv - 2024-01-10 (SL: 30% → 22%)
─────────────────────────────
All Saved Presets:
  • SL Analysis - backtest.csv - 2024-01-15
  • SL Analysis - backtest.csv - 2024-01-10
  • Custom Filter Set 1
─────────────────────────────
[Manage Presets...]
[Search Presets...]
```

### Apply Features Dialog Enhancement

**Add at top:**
```
[Load Preset ▼] [Manage Presets...]
Loaded Preset: SL Analysis - backtest.csv - 2024-01-15 [Clear]
─────────────────────────────────────────────────────
[Feature List...]
```

---

## Data Structures

### Preset File Format (JSON)

**Location**: `data/filter_presets/{preset_name}.json`

```json
{
  "name": "SL Analysis - backtest.csv - 2024-01-15",
  "source_backtest": "backtest_20240115.csv",
  "model_name": "xgb_classifier_selected_features.pkl",
  "created_date": "2024-01-15T10:30:00",
  "stop_loss_rate_before": 0.25,
  "stop_loss_rate_after": 0.18,
  "total_trades_before": 1234,
  "total_trades_after": 1085,
  "filters": [
    {
      "feature": "rsi14",
      "operator": ">",
      "value": 0.2,
      "effect_size": 0.65,
      "category": "strong"
    },
    {
      "feature": "volatility_ratio",
      "operator": "<",
      "value": 1.5,
      "effect_size": 0.58,
      "category": "strong"
    }
  ],
  "immediate_stop_filters": [
    {
      "feature": "rsi14",
      "operator": ">",
      "value": 0.15,
      "effect_size": 0.72,
      "category": "strong"
    }
  ],
  "notes": "User-added notes here"
}
```

### Backtest Metadata File Format (JSON)

**Location**: `data/backtest_results/{backtest_name}_metadata.json`

```json
{
  "backtest_file": "backtest_20240115.csv",
  "created_date": "2024-01-15T10:30:00",
  "model_name": "xgb_classifier_selected_features.pkl",
  "filter_preset_name": "SL Analysis - backtest.csv - 2024-01-15",
  "filters_applied": [
    {
      "feature": "rsi14",
      "operator": ">",
      "value": 0.2
    }
  ],
  "backtest_settings": {
    "horizon": 30,
    "return_threshold": 0.15,
    "stop_loss_mode": "adaptive_atr"
  }
}
```

### Analysis Results Structure

```python
{
    'stop_loss_count': int,
    'winning_count': int,
    'target_count': int,
    'total_trades': int,
    'stop_loss_rate': float,
    'immediate_stop_count': int,
    'immediate_stop_rate': float,
    'feature_comparisons': [
        {
            'feature': str,
            'stop_loss_mean': float,
            'winner_mean': float,
            'difference': float,
            'cohens_d': float,
            'abs_effect': float
        }
    ],
    'recommendations': [
        {
            'feature': str,
            'operator': str,  # '>', '>=', '<', '<='
            'value': float,
            'effect_size': float,
            'category': str,  # 'strong', 'moderate', 'weak'
            'description': str
        }
    ],
    'immediate_stop_recommendations': [...],
    'timing_analysis': {
        'day_of_week': {0: count, 1: count, ...},
        'month': {1: count, 2: count, ...}
    },
    'holding_period_stats': {
        'mean': float,
        'median': float,
        'min': float,
        'max': float,
        'distribution': {bin: count}
    },
    'return_stats': {
        'mean': float,
        'median': float,
        'min': float,
        'max': float,
        'std_dev': float
    }
}
```

---

## Service Layer API

### StopLossAnalysisService

**Location**: `gui/services.py`

```python
class StopLossAnalysisService:
    """Service for stop-loss analysis functionality."""
    
    def analyze_stop_losses(
        self,
        csv_path: str,
        data_dir: Path,
        effect_size_threshold: float = 0.3,
        progress_callback: Callable = None
    ) -> Dict:
        """
        Analyze stop-loss trades from backtest CSV.
        
        Args:
            csv_path: Path to backtest CSV file
            data_dir: Directory containing feature parquet files
            effect_size_threshold: Minimum effect size for recommendations
            progress_callback: Callback(completed, total, message)
            
        Returns:
            Analysis results dictionary
        """
    
    def get_entry_features(
        self,
        trades: pd.DataFrame,
        data_dir: Path,
        progress_callback: Callable = None
    ) -> pd.DataFrame:
        """
        Extract feature values at entry time for each trade.
        
        Args:
            trades: DataFrame with trade data
            data_dir: Directory containing feature parquet files
            progress_callback: Callback(completed, total, message)
            
        Returns:
            DataFrame with trades + feature values at entry
        """
    
    def calculate_impact(
        self,
        filters: List[Tuple[str, str, float]],
        trades_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> Dict:
        """
        Calculate impact of applying filters to trades.
        
        Args:
            filters: List of (feature, operator, value) tuples
            trades_df: Original trades DataFrame
            features_df: Features at entry time
            
        Returns:
            Dictionary with impact metrics:
            - stop_loss_excluded_pct
            - winner_excluded_pct
            - estimated_new_sl_rate
            - estimated_total_trades
            - warnings: List of warning messages
        """
    
    def save_preset(
        self,
        name: str,
        filters: List[Dict],
        metadata: Dict
    ) -> str:
        """
        Save filter preset to file.
        
        Args:
            name: Preset name
            filters: List of filter dictionaries
            metadata: Additional metadata (source_backtest, model_name, etc.)
            
        Returns:
            Path to saved preset file
        """
    
    def load_preset(self, preset_name: str) -> Dict:
        """
        Load filter preset from file.
        
        Args:
            preset_name: Name of preset to load
            
        Returns:
            Preset dictionary with filters and metadata
        """
    
    def list_presets(self) -> List[Dict]:
        """
        List all available presets.
        
        Returns:
            List of preset metadata dictionaries
        """
    
    def delete_preset(self, preset_name: str) -> bool:
        """
        Delete a preset.
        
        Args:
            preset_name: Name of preset to delete
            
        Returns:
            True if deleted, False if not found
        """
    
    def rename_preset(self, old_name: str, new_name: str) -> bool:
        """
        Rename a preset.
        
        Args:
            old_name: Current preset name
            new_name: New preset name
            
        Returns:
            True if renamed, False if not found
        """
```

---

## Phased Implementation Plan

### Phase 1: Core Analysis (5 units)

**Unit 1.1: Service Layer Foundation**
- Create `StopLossAnalysisService` class in `gui/services.py`
- Implement basic structure and imports
- Add placeholder methods
- **Test**: Service class exists, can be instantiated

**Unit 1.2: UI Tab Structure**
- Add "Stop-Loss Analysis" sub-tab to Analysis tab
- Create basic layout structure
- Add input section (CSV selector, Analyze button)
- Add placeholder sections for results
- **Test**: Tab appears, UI elements visible

**Unit 1.3: Data Loading & Feature Extraction**
- Implement `get_entry_features()` in service
- Connect to UI (load CSV, extract features)
- Display progress during extraction
- Handle missing feature files gracefully
- **Test**: Can load CSV, features extracted correctly

**Unit 1.4: Analysis Execution**
- Implement `analyze_stop_loss_patterns()` in service
- Run in background thread (QThread)
- Display summary cards with results
- Handle errors gracefully
- **Test**: Analysis completes, summary shows correct data

**Unit 1.5: Feature Comparison Table**
- Calculate feature comparisons (means, effect sizes)
- Display in sortable QTableWidget
- Format numbers appropriately
- **Test**: Table populates, sorting works, data accurate

### Phase 2: Recommendations & Selection (4 units)

**Unit 2.1: Recommendation Generation**
- Generate recommendations from analysis
- Group by effect size buckets (Strong/Moderate/Weak)
- Display in collapsible sections
- Weak recommendations collapsed by default
- **Test**: Recommendations generated, grouped correctly

**Unit 2.2: User Selection Interface**
- Add checkboxes to each recommendation
- Implement select all/deselect all
- Track selected filters state
- Update UI when selection changes
- **Test**: Selection works, state persists

**Unit 2.3: Impact Preview**
- Implement `calculate_impact()` in service
- Apply selected filters to historical trades
- Display preview panel with estimates
- Calculate stop-loss rate improvement
- **Test**: Preview shows accurate impact estimates

**Unit 2.4: Warning System**
- Implement warning thresholds (50% trades, 20% winners)
- Display warnings in preview panel
- Color-code warnings (yellow for caution, red for severe)
- **Test**: Warnings appear when appropriate

### Phase 3: Immediate Stop-Loss Analysis (2 units)

**Unit 3.1: Immediate Stop Detection**
- Detect trades with holding_days ≤ 1
- Display immediate stop statistics
- Add separate section in UI
- **Test**: Immediate stops detected correctly

**Unit 3.2: Special Recommendations**
- Generate recommendations for immediate stops
- Display in separate section
- Option to include/exclude in main recommendations
- **Test**: Special recommendations generated, can toggle inclusion

### Phase 4: Preset System (4 units)

**Unit 4.1: Preset Save**
- Implement preset save functionality
- Create preset directory structure (`data/filter_presets/`)
- Save JSON with metadata (including model_name)
- Auto-naming with user customization
- **Test**: Presets save correctly, metadata accurate

**Unit 4.2: Preset Load (Backtesting Tab)**
- Add "Load Preset" button to Backtesting tab
- Implement dropdown menu with presets
- Show recent stop-loss filters at top
- Load preset into Apply Features dialog
- **Test**: Can load preset, filters populate correctly

**Unit 4.3: Apply Features Dialog Enhancement**
- Add "Load Preset" button to dialog
- Show loaded preset name
- Add "Clear Preset" functionality
- Pre-populate filters from preset
- **Test**: Dialog enhancements work, preset state managed

**Unit 4.4: Preset Management**
- Implement delete preset
- Implement rename preset
- Create preset management dialog
- **Test**: Can delete/rename presets, changes persist

### Phase 5: Visualizations (3 units)

**Unit 5.1: Timing Charts**
- Day of week distribution chart (bar chart)
- Month distribution chart (bar chart)
- Use Matplotlib embedded in PyQt6
- **Test**: Charts display correctly, data accurate

**Unit 5.2: Holding Period Chart**
- Histogram of days to stop-loss
- Highlight immediate stops (≤1 day)
- Show statistics (avg, median, min, max)
- **Test**: Chart displays, highlights work

**Unit 5.3: Return Distribution Chart**
- Histogram of stop-loss returns
- Overlay threshold line
- Show statistics (mean, median, min, max, std dev)
- **Test**: Chart displays, overlay correct

### Phase 6: Backtest Comparison Integration (2 units)

**Unit 6.1: Filter Tracking**
- Store filter preset name in backtest metadata
- Create metadata file alongside CSV (`{backtest_name}_metadata.json`)
- Include filters_applied array
- Save metadata when backtest completes
- **Test**: Metadata saved, can be read back

**Unit 6.2: Display in Comparison Tab**
- Add "Filters Used" column to comparison table
- Show preset name or filter count
- Click to view filter details (tooltip or dialog)
- **Test**: Filters displayed, details viewable

### Phase 7: Export & Polish (2 units)

**Unit 7.1: Export Functionality**
- Export full report (PDF/HTML)
- Export recommendations table (CSV)
- Export preset (JSON)
- **Test**: All exports work, data accurate

**Unit 7.2: Performance & Polish**
- Cache feature extraction results per CSV
- Add progress bars for long operations
- Enhanced error handling
- UI refinements (auto-run, remember settings)
- **Test**: Performance acceptable, errors handled gracefully

---

## Design Decisions

### Defaults & Settings
- **Effect Size Threshold**: 0.3 (medium effect)
- **Show Weak Recommendations**: Yes, but collapsed by default
- **Max Recommendations**: No hard limit, show all above threshold
- **Warning Thresholds**: 
  - 50% of total trades excluded
  - 20% of winning trades excluded
- **Preset Naming**: Auto-name with user customization option
- **Auto-run Analysis**: Yes, when CSV loaded (with setting to disable)
- **Remember Settings**: Yes, persist in session/config

### Filter Application
- **Impact Preview**: Apply filters to historical trades (Option A - most accurate)
- **User Selection**: Checkboxes for individual selection
- **Apply Selected**: Only apply selected filters, not all recommendations

### Preset System
- **Model Name**: Include as optional metadata for future trade ID integration
- **Preset Location**: `data/filter_presets/` directory
- **Metadata File**: Separate JSON file alongside backtest CSV
- **Preset Format**: JSON with comprehensive metadata

### Immediate Stop-Losses
- **Definition**: Trades with holding_days ≤ 1
- **Special Analysis**: Yes, separate section with special recommendations
- **Include/Exclude**: Option to include in main recommendations or keep separate

### Integration Points
- **Analysis Tab**: New sub-tab "Stop-Loss Analysis"
- **Backtesting Tab**: "Load Preset" button next to "Apply Features"
- **Apply Features Dialog**: Enhanced with preset loading
- **Backtest Comparison**: Show filters used column

---

## Integration Points

### 1. Analysis Tab
- Add new sub-tab to existing `QTabWidget`
- Reuse existing backtest CSV loading logic
- Integrate with existing DataService

### 2. Backtesting Tab
- Add "Load Preset" button
- Enhance ApplyFeaturesDialog
- Connect preset loading to filter application

### 3. Apply Features Dialog
- Add preset loading functionality
- Show loaded preset indicator
- Pre-populate filters from preset

### 4. Backtest Comparison Tab
- Read metadata files when loading backtests
- Display filter information in table
- Show filter details on click

### 5. Feature Info System
- Reuse existing FeatureInfoDialog
- Add "Info" button to recommendations
- Show feature descriptions

---

## Testing Strategy

### Unit Testing Checklist

For each unit, verify:
- [ ] Feature works as expected
- [ ] Handles edge cases (empty data, missing files, etc.)
- [ ] Integrates with existing features
- [ ] UI looks correct
- [ ] No errors in console
- [ ] Performance acceptable
- [ ] Error messages are user-friendly

### Edge Cases to Test

1. **Empty Backtest CSV**: Should show appropriate message
2. **No Stop-Loss Trades**: Should handle gracefully
3. **Missing Feature Files**: Skip trades, show warning, continue
4. **All Trades Are Stop-Losses**: Should still work
5. **No Features Above Threshold**: Should show appropriate message
6. **Very Large Backtests**: Should handle performance (1000+ trades)
7. **Invalid Preset Files**: Should handle corruption gracefully
8. **Missing Metadata Files**: Should work without filter info

### Integration Testing

1. **Full Workflow**: Load CSV → Analyze → Select Filters → Save Preset → Load in Backtesting → Apply → Run Backtest
2. **Preset Management**: Create → Rename → Delete → Verify changes persist
3. **Backtest Comparison**: Run filtered backtest → Verify metadata saved → Check comparison tab shows filters

---

## Implementation Notes

### Code Organization

- **Service Layer**: `gui/services.py` - Add `StopLossAnalysisService` class
- **UI Tab**: `gui/tabs/stop_loss_analysis_tab.py` - New file
- **Preset Management**: `gui/utils/preset_manager.py` - New file (or extend existing)
- **Metadata Handling**: `gui/utils/backtest_metadata.py` - New file

### Dependencies

- Reuse existing: `DataService`, `BacktestService`
- Reuse existing: Feature info dialog, chart widgets
- New: Statistical calculations (Cohen's d, effect sizes)

### Performance Considerations

- Cache feature extraction results per CSV file
- Use background threads for long operations
- Progress bars for user feedback
- Handle large backtests efficiently (1000+ trades)

### Error Handling

- Missing feature files: Skip trade, log warning, continue
- Empty results: Show appropriate message
- Analysis errors: Show user-friendly error message
- Preset errors: Handle corruption, missing files gracefully

---

## Success Criteria

### Phase 1 Complete When:
- Can load backtest CSV
- Analysis runs and completes
- Summary cards show correct data
- Feature comparison table displays and sorts correctly

### Phase 2 Complete When:
- Recommendations generated and grouped correctly
- User can select/deselect recommendations
- Impact preview shows accurate estimates
- Warnings appear when appropriate

### Phase 3 Complete When:
- Immediate stops detected and analyzed
- Special recommendations generated
- Can include/exclude in main recommendations

### Phase 4 Complete When:
- Can save presets with metadata
- Can load presets in Backtesting tab
- Presets populate Apply Features dialog
- Can manage presets (delete, rename)

### Phase 5 Complete When:
- All charts display correctly
- Data is accurate
- Charts are interactive and informative

### Phase 6 Complete When:
- Backtest metadata saved with filters
- Comparison tab shows filter information
- Can view filter details

### Phase 7 Complete When:
- All export formats work
- Performance is acceptable
- Error handling is robust
- UI is polished

---

## Future Enhancements (Not in Current Scope)

1. Filter effectiveness tracking across backtests
2. Comparative analysis (compare filtered vs unfiltered)
3. Filter optimization (automatically find best thresholds)
4. Trade identification integration (use presets in identify tab)
5. Machine learning integration (predict stop-loss probability)
6. Stop-loss optimization (find optimal stop-loss thresholds)

---

## Revision History

- **2024-01-XX**: Initial implementation document created
- Updates will be logged here as implementation progresses

