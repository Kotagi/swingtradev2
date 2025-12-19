# Feature Set Isolation Implementation Plan

## Overview

This document outlines the implementation plan for making feature sets completely independent. Each feature set will have its own implementations, registry, and data, allowing experimentation without affecting other sets.

## Current State

### Current Structure
```
features/
├── technical.py          # All feature implementations (shared)
├── registry.py           # Feature registry (shared)
└── metadata.py          # Feature metadata (if exists)

config/
├── features.yaml         # v1 feature config
├── train_features.yaml   # v1 training config
└── features_v2.yaml      # v2 feature config (if exists)

data/
├── features_labeled/     # v1 feature data
└── features_labeled_v2/  # v2 feature data (if exists)
```

### Current Problems
1. **Shared implementations**: All feature sets use the same `features/technical.py` and `features/registry.py`
2. **Intertwined configs**: Features get added to same registry, making pruning difficult
3. **Rebuild requirement**: Must rebuild all sets when testing different ones
4. **Risk of breaking**: Changes to one set can affect others

### Current Dependencies
- `src/feature_pipeline.py` imports `from features.registry import load_enabled_features`
- `features/registry.py` imports from `features.technical`
- GUI reads from `config/features.yaml` (not directly importing feature modules)

## Target State

### Target Structure
```
features/
├── shared/
│   ├── __init__.py
│   ├── base/                    # Empty for now, ready for future
│   │   ├── __init__.py
│   │   ├── technical.py        # Empty (ready for future shared features)
│   │   └── registry.py         # Empty (ready for future shared registry)
│   └── utils.py                # Non-feature utilities (SPY loading, helpers)
└── sets/
    ├── v1/
    │   ├── __init__.py
    │   ├── technical.py        # v1 feature implementations (migrated)
    │   └── registry.py         # v1 feature registry (migrated)
    └── v2/                      # Created when needed
        ├── __init__.py
        ├── technical.py        # v2 feature implementations
        └── registry.py         # v2 feature registry

config/
├── features.yaml               # v1 (will be deleted after migration)
├── features_v1.yaml            # v1 (new structure)
├── features_v2.yaml            # v2 (when created)
├── train_features.yaml         # v1 (will be deleted after migration)
├── train_features_v1.yaml      # v1 (new structure)
├── train_features_v2.yaml     # v2 (when created)
└── feature_sets_metadata.yaml  # NEW: Registry of all feature sets

data/
├── features_labeled/           # v1 (will be deleted after migration)
├── features_labeled_v1/        # v1 (new structure)
└── features_labeled_v2/        # v2 (when created)
```

### Target Behavior
1. **Complete isolation**: Each feature set has its own implementations
2. **Independent testing**: Can test new features without affecting existing sets
3. **Easy pruning**: Delete a feature set without affecting others
4. **GUI integration**: Feature set selector in all relevant tabs
5. **Model binding**: Models are strictly tied to their feature set

## Implementation Phases

### Phase 1: Structure Setup (Low Risk)
**Goal**: Create directory structure and move shared utilities

**Tasks**:
1. Create `features/shared/` directory
2. Create `features/shared/base/` directory (empty)
3. Create `features/shared/base/__init__.py`
4. Create `features/shared/base/technical.py` (empty, with comments)
5. Create `features/shared/base/registry.py` (empty, with comments)
6. Create `features/shared/utils.py`
7. Move `_load_spy_data()` and other non-feature utilities to `features/shared/utils.py`
8. Update imports in `features/technical.py` to use `features.shared.utils`
9. Create `features/sets/` directory
10. Create `features/sets/v1/` directory
11. Create `features/sets/v1/__init__.py`
12. Create `config/feature_sets_metadata.yaml` with v1 entry

**Files Created**:
- `features/shared/__init__.py`
- `features/shared/base/__init__.py`
- `features/shared/base/technical.py` (empty)
- `features/shared/base/registry.py` (empty)
- `features/shared/utils.py`
- `features/sets/__init__.py`
- `features/sets/v1/__init__.py`
- `config/feature_sets_metadata.yaml`

**Files Modified**:
- `features/technical.py` (update imports for shared utils)

**Testing**:
- Verify directory structure created
- Verify shared utilities can be imported
- Run feature pipeline to ensure nothing broke

---

### Phase 2: Migration (Medium Risk)
**Goal**: Move v1 features to new structure and update all imports

**Tasks**:
1. Copy `features/technical.py` → `features/sets/v1/technical.py`
2. Copy `features/registry.py` → `features/sets/v1/registry.py`
3. Update `features/sets/v1/technical.py`:
   - Update imports to use `features.shared.utils` instead of local helpers
   - Verify all imports work
4. Update `features/sets/v1/registry.py`:
   - Update import: `from features.sets.v1.technical import ...`
   - Verify all feature functions are imported correctly
5. Update `src/feature_pipeline.py`:
   - Change: `from features.registry import load_enabled_features`
   - To: `from features.sets.v1.registry import load_enabled_features`
   - Add feature set support (use feature_set_manager to determine which registry to load)
6. Update `config/features.yaml` → `config/features_v1.yaml` (copy)
7. Update `config/train_features.yaml` → `config/train_features_v1.yaml` (copy)
8. Update `src/feature_set_manager.py`:
   - Ensure it points to new v1 locations
   - Update `get_feature_set_config_path()` and `get_feature_set_data_path()`
9. Update `config/feature_sets_metadata.yaml`:
   - Add v1 entry with correct paths

**Files Created**:
- `features/sets/v1/technical.py` (copied from old location)
- `features/sets/v1/registry.py` (copied from old location)
- `config/features_v1.yaml` (copied from features.yaml)
- `config/train_features_v1.yaml` (copied from train_features.yaml)

**Files Modified**:
- `src/feature_pipeline.py` (update imports, add feature set support)
- `src/feature_set_manager.py` (update paths)
- `config/feature_sets_metadata.yaml` (add v1 entry)

**Files to Delete Later** (after testing):
- `features/technical.py`
- `features/registry.py`
- `config/features.yaml` (after confirming v1 works)
- `config/train_features.yaml` (after confirming v1 works)

**Testing**:
- Run feature pipeline: `python src/feature_pipeline.py --feature-set v1`
- Verify features build correctly
- Verify feature data is saved to `data/features_labeled_v1/`
- Check that all features are computed correctly

---

### Phase 3: Testing & Validation (Critical)
**Goal**: Verify everything works before deleting old files

**Tasks**:
1. **Feature Pipeline Testing**:
   - Build features for v1: `python src/feature_pipeline.py --feature-set v1 --full`
   - Verify all features are computed
   - Compare feature counts with old system
   - Verify data files are created in correct location

2. **Model Training Testing**:
   - Train a model with v1: `python src/train_model.py --feature-set v1`
   - Verify model trains successfully
   - Verify model is saved with correct feature set metadata
   - Check model registry entry includes feature_set

3. **Backtesting Testing**:
   - Run backtest with v1 model
   - Verify backtest completes successfully
   - Verify results are correct

4. **GUI Testing**:
   - Test Feature Engineering tab (should work with v1)
   - Test Model Training tab (should work with v1)
   - Test Backtesting tab (should work with v1)
   - Test Filter Editor tab (should work with v1)
   - Verify all tabs can access v1 features

5. **Import Verification**:
   - Search codebase for any remaining `from features.technical` imports
   - Search for `from features.registry` imports
   - Update any found imports

**Success Criteria**:
- ✅ Feature pipeline builds features successfully
- ✅ Model training works with v1
- ✅ Backtesting works with v1
- ✅ GUI works with v1
- ✅ No import errors
- ✅ All features computed correctly

---

### Phase 4: Cleanup (After Testing)
**Goal**: Remove old files after confirming new system works

**Tasks**:
1. Verify all tests pass
2. Delete `features/technical.py`
3. Delete `features/registry.py`
4. Delete `config/features.yaml` (keep `config/features_v1.yaml`)
5. Delete `config/train_features.yaml` (keep `config/train_features_v1.yaml`)
6. Delete `data/features_labeled/` (keep `data/features_labeled_v1/`)
7. Update any remaining references

**Files to Delete**:
- `features/technical.py`
- `features/registry.py`
- `config/features.yaml`
- `config/train_features.yaml`
- `data/features_labeled/` (after confirming v1 data exists)

**Verification**:
- Run full test suite again
- Verify nothing breaks

---

### Phase 5: Feature Set Manager Enhancement
**Goal**: Enhance feature set management tools

**Tasks**:
1. Update `src/manage_feature_sets.py`:
   - Add "create" command that copies from existing set
   - Add "delete" command with cleanup options
   - Add "list" command with more details
   - Add "info" command showing set details

2. Update `src/feature_set_manager.py`:
   - Add function to get feature set metadata
   - Add function to validate feature set
   - Add function to get all feature sets

**Files Modified**:
- `src/manage_feature_sets.py`
- `src/feature_set_manager.py`

---

### Phase 6: GUI Integration (Medium Risk)
**Goal**: Add feature set management and selection to GUI

**Tasks**:
1. **Feature Set Management UI** (Feature Engineering Tab):
   - Add "Feature Sets" section
   - List all available feature sets
   - "Create New" button (opens dialog)
   - "Copy From" dropdown (select source set)
   - "Delete" button (with confirmation)
   - Show feature set details (feature count, data files, models)

2. **Global Feature Set Selector**:
   - Add to main window (toolbar or menu bar)
   - Dropdown showing all feature sets
   - Current selection highlighted
   - Persist selection across sessions

3. **Per-Tab Feature Set Selection**:
   - Feature Engineering tab: Use selected feature set
   - Model Training tab: Use selected feature set
   - Backtesting tab: Use selected feature set
   - Trade Identification tab: Use selected feature set
   - Filter Editor tab: Use selected feature set
   - Analysis tabs: Show which feature set was used

4. **Feature Selection Dialog Updates**:
   - Filter features by active feature set
   - Show feature set name in dialog
   - Load from correct `train_features_<set>.yaml`

5. **Model Compatibility Validation**:
   - When loading model, verify feature set matches
   - Show error if mismatch
   - Prevent cross-set usage

**Files Created**:
- `gui/widgets/feature_set_selector.py` (new widget)
- `gui/tabs/feature_set_management_dialog.py` (new dialog)

**Files Modified**:
- `gui/main_window.py` (add global selector)
- `gui/tabs/features_tab.py` (add feature set management)
- `gui/tabs/training_tab.py` (use feature set selector)
- `gui/tabs/backtest_tab.py` (use feature set selector)
- `gui/tabs/identify_tab.py` (use feature set selector)
- `gui/tabs/filter_editor_tab.py` (use feature set selector)
- `gui/tabs/feature_selection_dialog.py` (filter by feature set)
- `gui/services.py` (update services to use feature sets)

---

### Phase 7: Documentation & Final Testing
**Goal**: Update documentation and perform final validation

**Tasks**:
1. Update `README.md`:
   - Document feature set system
   - Update file structure section
   - Add feature set workflow

2. Update `info/FEATURE_SETS_GUIDE.md`:
   - Update with new structure
   - Add GUI workflow
   - Add migration notes

3. Update in-app help:
   - Add Feature Set Management help
   - Update Feature Engineering help
   - Update Model Training help

4. Final end-to-end testing:
   - Create v2 feature set
   - Build features for v2
   - Train model on v2
   - Run backtest with v2
   - Compare v1 vs v2
   - Delete v2 (test cleanup)

**Files Modified**:
- `README.md`
- `info/FEATURE_SETS_GUIDE.md`
- `gui/help_panel.py`

---

## Detailed File Changes

### New Files to Create

1. **`features/shared/__init__.py`**
   ```python
   """Shared utilities for feature sets."""
   ```

2. **`features/shared/base/__init__.py`**
   ```python
   """
   Base feature implementations (shared across feature sets).
   
   Currently empty - ready for future shared features.
   When a feature implementation is finalized and proven,
   it can be moved here to be shared across all feature sets.
   """
   ```

3. **`features/shared/base/technical.py`**
   ```python
   """
   Base technical feature implementations.
   
   This file is currently empty and ready for future use.
   When feature implementations are finalized and proven to work
   across all feature sets, they can be moved here to avoid duplication.
   
   For now, each feature set has its own complete implementation
   in features/sets/<set_name>/technical.py
   """
   # Empty for now - ready for future shared features
   ```

4. **`features/shared/base/registry.py`**
   ```python
   """
   Base feature registry.
   
   This file is currently empty and ready for future use.
   When shared features are added to base/technical.py,
   they can be registered here.
   """
   # Empty for now - ready for future shared registry
   ```

5. **`features/shared/utils.py`**
   ```python
   """
   Shared utilities for feature computation.
   
   Contains non-feature-specific utilities like:
   - SPY data loading
   - Helper functions used by multiple features
   - Common data processing utilities
   """
   # Move _load_spy_data() and other helpers here
   ```

6. **`features/sets/__init__.py`**
   ```python
   """Feature set implementations."""
   ```

7. **`features/sets/v1/__init__.py`**
   ```python
   """v1 feature set implementations."""
   ```

8. **`config/feature_sets_metadata.yaml`**
   ```yaml
   feature_sets:
     v1:
       name: "v1"
       description: "Default feature set"
       created: "2024-12-19"
       version: "1.0"
       features_dir: "features/sets/v1"
       config_file: "config/features_v1.yaml"
       train_config_file: "config/train_features_v1.yaml"
       data_dir: "data/features_labeled_v1"
   ```

### Files to Modify

1. **`features/technical.py`** → **`features/sets/v1/technical.py`**
   - Move entire file
   - Update imports: `from features.shared.utils import _load_spy_data`
   - No other changes needed

2. **`features/registry.py`** → **`features/sets/v1/registry.py`**
   - Move entire file
   - Update import: `from features.sets.v1.technical import ...`
   - No other changes needed

3. **`src/feature_pipeline.py`**
   - Change: `from features.registry import load_enabled_features`
   - To: Dynamic import based on feature set
   - Add feature set parameter handling
   - Use `feature_set_manager` to get correct registry path

4. **`src/feature_set_manager.py`**
   - Update `get_feature_set_config_path()` to use new structure
   - Update `get_feature_set_data_path()` to use new structure
   - Add function to get feature set registry path
   - Add function to load feature set metadata

5. **`src/train_model.py`**
   - Already supports feature sets via `feature_set_manager`
   - Verify it works with new structure
   - Ensure model registry includes feature_set

6. **`gui/services.py`**
   - Update `FeatureService` to use feature sets
   - Update `TrainingService` to use feature sets
   - Update `BacktestService` to use feature sets

## Testing Strategy

### Unit Tests
1. Test feature set discovery
2. Test feature set creation
3. Test feature set deletion
4. Test registry loading for each set

### Integration Tests
1. Build features for v1 → verify success
2. Train model with v1 → verify success
3. Run backtest with v1 → verify success
4. Create v2, build features → verify isolation
5. Train model with v2 → verify different from v1
6. Delete v2 → verify v1 unaffected

### GUI Tests
1. Feature set selector works
2. Feature set creation dialog works
3. Feature set deletion works
4. All tabs use correct feature set
5. Feature selection dialog filters correctly

## Risk Mitigation

### Risks
1. **Import errors**: Old code still imports from old locations
   - **Mitigation**: Search entire codebase for old imports, update all

2. **Missing features**: Features not migrated correctly
   - **Mitigation**: Compare feature counts before/after migration

3. **Data loss**: Accidentally delete important data
   - **Mitigation**: Keep old files until testing complete, backup before deletion

4. **Model incompatibility**: Models trained on old structure don't work
   - **Mitigation**: Test model loading, retrain if needed

5. **GUI breaks**: GUI doesn't work with new structure
   - **Mitigation**: Test all GUI tabs, update services layer

### Rollback Plan
1. Keep old files until Phase 3 testing complete
2. Git branch for easy rollback
3. Test in feature branch before merging
4. Can revert to old structure if needed

## Success Criteria

### Phase 1 Success
- ✅ Directory structure created
- ✅ Shared utilities moved
- ✅ No import errors

### Phase 2 Success
- ✅ v1 features migrated to new location
- ✅ All imports updated
- ✅ Feature pipeline works with v1

### Phase 3 Success
- ✅ Feature pipeline builds features
- ✅ Model training works
- ✅ Backtesting works
- ✅ GUI works
- ✅ All features computed correctly

### Phase 4 Success
- ✅ Old files deleted
- ✅ No broken imports
- ✅ Everything still works

### Phase 5 Success
- ✅ Feature set management tools work
- ✅ Can create/copy/delete sets

### Phase 6 Success
- ✅ GUI has feature set selector
- ✅ All tabs use feature sets correctly
- ✅ Feature set management UI works

### Phase 7 Success
- ✅ Documentation updated
- ✅ End-to-end workflow tested
- ✅ Can create and test new feature sets independently

## Implementation Order

1. **Phase 1**: Structure Setup (Low risk, foundation)
2. **Phase 2**: Migration (Medium risk, core changes)
3. **Phase 3**: Testing (Critical, verify everything works)
4. **Phase 4**: Cleanup (Remove old files)
5. **Phase 5**: Feature Set Manager (Enhancement)
6. **Phase 6**: GUI Integration (User-facing)
7. **Phase 7**: Documentation (Final polish)

## Git Workflow

1. Create feature branch: `feature/feature-set-isolation`
2. Commit after each phase
3. Test after each phase
4. Push to GitHub after each phase (for backup)
5. Merge to main only after all phases complete and tested

## Questions Resolved

✅ **Backward Compatibility**: Not needed - will delete old files after testing
✅ **Feature Set Discovery**: Auto-discover from config files + metadata
✅ **Testing Strategy**: Test everything before deleting old files
✅ **Shared Utilities**: Move SPY loading and helpers to shared
✅ **Feature Set Creation**: Start empty, with option to copy from existing
✅ **GUI Selector**: Global selector with per-tab override capability
✅ **Model Compatibility**: Strict - models tied to feature set
✅ **Implementation Approach**: Phased - structure → migration → testing → GUI

## Next Steps

1. Review this document
2. Create feature branch
3. Begin Phase 1 implementation
4. Test after each phase
5. Proceed to next phase only after previous phase is verified

