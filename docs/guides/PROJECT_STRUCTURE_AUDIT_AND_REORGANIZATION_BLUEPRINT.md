# Project Structure Audit & Reorganization Blueprint
## Comprehensive Professional-Level Analysis

**Date:** January 26, 2026  
**Project:** SwingTradeV2 - Swing Trading ML Application  
**Status:** Active Development - v3_New_Dawn Feature Set (415 features complete)

---

## EXECUTIVE SUMMARY

This document provides a comprehensive audit of the current project structure and a detailed blueprint for reorganizing it to professional/master's level standards. The reorganization will:

1. **Separate concerns** - Clear boundaries between source code, tests, documentation, data, and outputs
2. **Eliminate root clutter** - Move all test scripts, benchmarks, and output files to dedicated directories
3. **Organize documentation** - Consolidate and categorize info folder contents
4. **Standardize naming** - Consistent conventions across all files and folders
5. **Improve maintainability** - Structure that scales with project growth

**Key Principle:** A master's level programmer would structure this as a production-ready application with clear separation of concerns, not a research prototype.

---

## PART 1: CURRENT STATE ANALYSIS

### 1.1 Root Directory Issues

#### Files That Should NOT Be in Root:

**Test Scripts (6 files):**
- `test_block_1_1.py` - Feature block testing
- `test_feature_integrity.py` - Feature validation testing
- `test_label_calculation.py` - Label calculation testing
- `test_optimization_integrity.py` - Optimization validation
- `test_phase3_comprehensive.py` - Phase 3 comprehensive testing
- `test_phase3_imports.py` - Import validation testing
- `test_spy_data_optimization.py` - SPY data optimization testing

**Benchmark/Profile Scripts (3 files):**
- `benchmark_feature_pipeline.py` - Performance benchmarking
- `profile_feature_pipeline.py` - Performance profiling
- `compare_benchmark_results.py` - Benchmark comparison utility

**Output Files (2 files):**
- `benchmark_results.txt` - Benchmark output (should be in outputs/)
- `feature_pipeline.log` - Log file (should be in logs/)

**Legitimate Root Files:**
- `README.md` ✅
- `requirements.txt` ✅
- `run_gui.py` ✅ (entry point - acceptable)
- `.gitignore` ✅

### 1.2 Info Folder Analysis

**Current Status:** 26 markdown files + 1 text file

#### Documentation Categories:

**A. ACTIVE/RELEVANT (Keep & Organize):**
1. **Core Documentation:**
   - `README.md` - Main project README
   - `QUICKSTART.md` - Quick start guide
   - `INSTALL.md` - Installation instructions
   - `APP_README.md` - Application-specific README

2. **Feature Documentation (Active):**
   - `FEATURE_GUIDE.md` - Complete feature guide (61 features - may be outdated)
   - `FEATURE_SETS_GUIDE.md` - Feature sets documentation
   - `NEW_DAWN_IMPLEMENTATION_ROADMAP.md` - **CURRENT** - Active roadmap (415 features)
   - `NEW_DAWN_FEATURE_CHECKLIST.md` - Feature checklist
   - `FEATURE_REDUNDANCY_GUIDE.md` - Feature selection guide

3. **Implementation Guides (Active):**
   - `FEATURE_SET_ISOLATION_IMPLEMENTATION.md` - Feature set architecture
   - `LOOKAHEAD_BIAS_PREVENTION_GUIDE.md` - Data integrity guide
   - `PIPELINE_STEPS.md` - Pipeline documentation

4. **Optimization Brainstorms (Active):**
   - `TECHNICAL_CALCULATIONS_OPTIMIZATION_BRAINSTORM.md` - **CURRENT** - Performance optimization
   - `FEATURE_OPTIMIZATION_GAMEPLAN.md` - Feature pipeline optimization
   - `PERFORMANCE_DEBUGGING_BRAINSTORM.md` - Performance debugging

5. **GUI Documentation (Active):**
   - `GUI_PHASE1_SUMMARY.md` - Historical
   - `GUI_PHASE2_SUMMARY.md` - Historical
   - `GUI_PHASE3_COMPLETE.md` - Historical
   - `GUI_PHASE4_COMPLETE.md` - Historical

6. **Model Documentation (Active):**
   - `MODEL_IMPROVEMENT_PLAN.md` - Model improvement strategies
   - `MODEL_PERFORMANCE_ANALYSIS.md` - Performance analysis
   - `model_training_overview.md` - Training overview

7. **Testing Documentation (Active):**
   - `TEST_GUIDE.md` - Testing guide
   - `PHASE3_TESTING_GUIDE.md` - Phase 3 specific testing
   - `BENCHMARK_GUIDE.md` - Benchmarking guide

**B. HISTORICAL/ARCHIVAL (Archive or Consolidate):**
1. **Phase Summaries (Historical):**
   - `PHASE1_FEATURE_ANALYSIS.md` - Phase 1 analysis (completed)
   - `PHASE1_IMPLEMENTATION_SUMMARY.md` - Phase 1 summary (completed)
   - `BLOCK_1_1_IMPLEMENTATION_SUMMARY.md` - Block 1.1 summary (completed)
   - `ROADMAP_P4-9.md` - Old roadmap (superseded by NEW_DAWN)

2. **Outdated Roadmaps:**
   - `FEATURE_ROADMAP.md` - Old feature roadmap (superseded)
   - `FEATURE_IMPROVEMENT_PLAN.md` - Old improvement plan (may be superseded)

3. **Data/Download Documentation:**
   - `DATA_SOURCES_EXPLAINED.md` - Data sources (reference)
   - `DOWNLOAD_ERROR_FIXES.md` - Error fixes (reference)
   - `DOWNLOAD_IMPROVEMENTS_SUMMARY.md` - Download improvements (reference)
   - `SEC_EDGAR_SETUP.md` - SEC EDGAR setup (reference)

4. **Future Planning:**
   - `FUTURE_FEATURES.md` - Future features list
   - `GAIN_PROBABILITY_OPTIMIZATION_BRAINSTORM.md` - Future optimization

5. **Miscellaneous:**
   - `Filters.txt` - Filter definitions (should be in config or data)

### 1.3 Notebooks Folder Analysis

**Current Status:** 9 files (mix of .txt, .rtf)

**Issues:**
- Mix of formats (.txt, .rtf)
- Unclear naming conventions
- Some appear to be outdated (original_roadmap.txt, phase 3 Road Map.txt)
- Contains both reference material and temporary notes

**Files:**
- `Comprehensive Feature Guide.rtf` - Duplicate of FEATURE_GUIDE.md?
- `Comprehensive Feature Guide.txt` - Duplicate?
- `feature_checklist.txt` - Feature checklist
- `feature_list.txt` - Feature list
- `original_roadmap.txt` - Historical roadmap
- `phase 3 Road Map.txt` - Historical (space in name - bad practice)
- `phase_3_roadmap.txt` - Historical
- `SearchParams.txt` - Search parameters
- `tickererror.txt` - Error log (should be in logs/)

### 1.4 Source Code Structure Analysis

**Current Structure:**
```
src/                    # Main application code ✅
features/               # Feature engineering ✅
gui/                    # GUI application ✅
utils/                  # Utility modules ✅
config/                 # Configuration files ✅
tests/                  # Test files ✅
scripts/                # Utility scripts ✅
```

**Issues:**
- `src/` contains both core modules and utility scripts
- Some scripts in `src/` could be in `scripts/`
- `tests/` has an `archive/` subfolder (good practice)

### 1.5 Data Folder Structure

**Current Structure:**
```
data/
├── raw/                    # Raw CSV data ✅
├── clean/                  # Cleaned Parquet ✅
├── features_labeled_v2/    # Feature data v2 ✅
├── features_labeled_v3_New_Dawn/  # Feature data v3 ✅
├── backtest_results/       # Backtest outputs ✅
├── filter_presets/         # Filter presets ✅
├── opportunities/          # Trade opportunities ✅
├── tickers/                # Ticker lists ✅
├── macro/                  # Macro data ✅
├── api_keys/               # API keys (security concern)
├── inspect_parquet/        # Inspection outputs (should be in outputs/)
├── temp_feature_test_input/  # Temp files (should be in temp/)
└── temp_feature_test_output/ # Temp files (should be in temp/)
```

**Issues:**
- `api_keys/` should be in `.env` or outside repo
- `inspect_parquet/` outputs should be in `outputs/`
- Temp folders should be in dedicated `temp/` or `data/temp/`
- Multiple feature version folders (expected, but could be better organized)

### 1.6 Models Folder Structure

**Current Structure:**
```
models/
├── models_registry.json        # Model registry ✅
├── training_metadata.json      # Training metadata ✅
├── feature_importances_all.csv  # Feature importances ✅
└── shap_artifacts/             # SHAP outputs ✅
    └── [multiple model folders with timestamps]
```

**Issues:**
- `shap_artifacts/` contains many timestamped folders (expected, but could use cleanup policy)
- No clear organization by feature set or date

### 1.7 Reports Folder

**Current Status:** 2 CSV files
- `backtest_filtered_20251210_030831.csv` - Timestamped backtest output
- `Filters.csv` - Filter definitions

**Issues:**
- Should be organized by date or type
- `Filters.csv` might belong in `config/` or `data/filter_presets/`

---

## PART 2: PROPOSED MASTER'S LEVEL STRUCTURE

### 2.1 Root Directory Structure

```
SwingTradeV2/
├── README.md                    # Main project README
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── .env.example                # Environment variable template (NEW)
├── run_gui.py                  # GUI entry point
│
├── src/                        # Source code (application logic)
│   ├── __init__.py
│   ├── swing_trade_app.py      # Main CLI entry point
│   ├── core/                   # Core business logic (NEW)
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   ├── clean_data.py
│   │   ├── feature_pipeline.py
│   │   ├── train_model.py
│   │   ├── enhanced_backtest.py
│   │   ├── identify_trades.py
│   │   └── shap_service.py
│   ├── analysis/               # Analysis modules (NEW)
│   │   ├── __init__.py
│   │   ├── analyze_features.py
│   │   ├── analyze_stop_losses.py
│   │   ├── compare_filters.py
│   │   └── apply_entry_filters.py
│   ├── data/                   # Data management (NEW)
│   │   ├── __init__.py
│   │   ├── inspect_parquet.py
│   │   ├── inspect_parquet_gui.py
│   │   └── download_vix.py
│   ├── features/               # Feature management (NEW)
│   │   ├── __init__.py
│   │   ├── feature_set_manager.py
│   │   ├── manage_feature_sets.py
│   │   ├── apply_feature_pruning.py
│   │   └── clean_features_labeled.py
│   └── utils/                  # Utility modules (NEW)
│       ├── __init__.py
│       ├── tune_threshold.py
│       └── backtest.py
│
├── features/                   # Feature engineering (unchanged)
│   ├── __init__.py
│   ├── metadata.py
│   ├── sets/
│   │   ├── v1/
│   │   ├── v2/
│   │   └── v3_New_Dawn/
│   └── shared/
│
├── gui/                        # GUI application (unchanged)
│   ├── __init__.py
│   ├── app.py
│   ├── main_window.py
│   ├── config_manager.py
│   ├── services.py
│   ├── styles.py
│   ├── help_panel.py
│   ├── tabs/
│   ├── widgets/
│   └── utils/
│
├── utils/                      # Shared utilities (unchanged)
│   ├── __init__.py
│   ├── labeling.py
│   ├── logger.py
│   └── stop_loss_policy.py
│
├── config/                     # Configuration files (unchanged)
│   ├── features_v1.yaml
│   ├── features_v2.yaml
│   ├── features_v3_New_Dawn.yaml
│   ├── train_features_v1.yaml
│   ├── train_features_v2.yaml
│   ├── train_features_v3_New_Dawn.yaml
│   ├── trading_config.yaml
│   ├── feature_sets_metadata.yaml
│   └── feature_presets/
│
├── tests/                      # Test suite (reorganized)
│   ├── __init__.py
│   ├── unit/                   # Unit tests (NEW)
│   │   ├── __init__.py
│   │   ├── test_features/
│   │   │   ├── test_block_1_1.py
│   │   │   ├── test_feature_integrity.py
│   │   │   └── test_label_calculation.py
│   │   ├── test_models/
│   │   │   └── test_optimization_integrity.py
│   │   └── test_data/
│   │       └── test_spy_data_optimization.py
│   ├── integration/           # Integration tests (NEW)
│   │   ├── __init__.py
│   │   ├── test_phase3_comprehensive.py
│   │   └── test_phase3_imports.py
│   ├── performance/           # Performance tests (NEW)
│   │   ├── __init__.py
│   │   ├── benchmark_feature_pipeline.py
│   │   ├── profile_feature_pipeline.py
│   │   └── compare_benchmark_results.py
│   └── archive/                # Archived tests (existing)
│
├── scripts/                    # Utility scripts (expanded)
│   ├── readme_generator.py
│   ├── data_management/        # Data scripts (NEW)
│   └── maintenance/            # Maintenance scripts (NEW)
│
├── docs/                       # Documentation (reorganized from info/)
│   ├── README.md               # Documentation index
│   ├── getting_started/        # Getting started guides
│   │   ├── QUICKSTART.md
│   │   ├── INSTALL.md
│   │   └── APP_README.md
│   ├── features/               # Feature documentation
│   │   ├── FEATURE_GUIDE.md
│   │   ├── FEATURE_SETS_GUIDE.md
│   │   ├── NEW_DAWN_IMPLEMENTATION_ROADMAP.md
│   │   ├── NEW_DAWN_FEATURE_CHECKLIST.md
│   │   └── FEATURE_REDUNDANCY_GUIDE.md
│   ├── guides/                 # Implementation guides
│   │   ├── PIPELINE_STEPS.md
│   │   ├── FEATURE_SET_ISOLATION_IMPLEMENTATION.md
│   │   ├── LOOKAHEAD_BIAS_PREVENTION_GUIDE.md
│   │   └── TEST_GUIDE.md
│   ├── optimization/           # Optimization documentation
│   │   ├── TECHNICAL_CALCULATIONS_OPTIMIZATION_BRAINSTORM.md
│   │   ├── FEATURE_OPTIMIZATION_GAMEPLAN.md
│   │   └── PERFORMANCE_DEBUGGING_BRAINSTORM.md
│   ├── gui/                    # GUI documentation
│   │   ├── GUI_PHASE1_SUMMARY.md
│   │   ├── GUI_PHASE2_SUMMARY.md
│   │   ├── GUI_PHASE3_COMPLETE.md
│   │   └── GUI_PHASE4_COMPLETE.md
│   ├── models/                 # Model documentation
│   │   ├── MODEL_IMPROVEMENT_PLAN.md
│   │   ├── MODEL_PERFORMANCE_ANALYSIS.md
│   │   └── model_training_overview.md
│   ├── reference/              # Reference documentation
│   │   ├── DATA_SOURCES_EXPLAINED.md
│   │   ├── DOWNLOAD_ERROR_FIXES.md
│   │   ├── DOWNLOAD_IMPROVEMENTS_SUMMARY.md
│   │   ├── SEC_EDGAR_SETUP.md
│   │   └── BENCHMARK_GUIDE.md
│   ├── archive/                # Historical documentation
│   │   ├── PHASE1_FEATURE_ANALYSIS.md
│   │   ├── PHASE1_IMPLEMENTATION_SUMMARY.md
│   │   ├── BLOCK_1_1_IMPLEMENTATION_SUMMARY.md
│   │   ├── ROADMAP_P4-9.md
│   │   ├── FEATURE_ROADMAP.md
│   │   └── FEATURE_IMPROVEMENT_PLAN.md
│   └── planning/               # Future planning
│       ├── FUTURE_FEATURES.md
│       └── GAIN_PROBABILITY_OPTIMIZATION_BRAINSTORM.md
│
├── data/                       # Data storage (reorganized)
│   ├── raw/                    # Raw data (unchanged)
│   ├── clean/                  # Cleaned data (unchanged)
│   ├── features/               # Feature data (reorganized)
│   │   ├── v1/
│   │   ├── v2/
│   │   └── v3_New_Dawn/
│   ├── backtest_results/       # Backtest outputs (unchanged)
│   ├── filter_presets/         # Filter presets (unchanged)
│   ├── opportunities/          # Trade opportunities (unchanged)
│   ├── tickers/                # Ticker lists (unchanged)
│   ├── macro/                  # Macro data (unchanged)
│   ├── temp/                   # Temporary files (NEW)
│   │   ├── feature_test_input/
│   │   └── feature_test_output/
│   └── .gitkeep                # Keep folder in git
│
├── models/                     # Model storage (unchanged)
│   ├── models_registry.json
│   ├── training_metadata.json
│   ├── feature_importances_all.csv
│   └── shap_artifacts/
│       └── [organized by feature_set/date]
│
├── outputs/                    # Generated outputs (NEW)
│   ├── logs/                   # Log files
│   │   ├── feature_pipeline.log
│   │   └── .gitkeep
│   ├── benchmarks/             # Benchmark results
│   │   ├── benchmark_results.txt
│   │   └── .gitkeep
│   ├── reports/                # Generated reports
│   │   └── .gitkeep
│   └── inspections/            # Inspection outputs
│       └── .gitkeep
│
├── notebooks/                   # Jupyter notebooks & notes (reorganized)
│   ├── README.md               # Notebooks index
│   ├── research/               # Research notebooks (NEW)
│   │   └── .gitkeep
│   └── archive/                # Archived notes
│       ├── Comprehensive Feature Guide.rtf
│       ├── Comprehensive Feature Guide.txt
│       ├── original_roadmap.txt
│       ├── phase_3_roadmap.txt
│       └── phase 3 Road Map.txt
│
├── bats/                       # Batch scripts (unchanged)
│
└── .env                        # Environment variables (gitignored)
```

### 2.2 Key Structural Changes

#### A. Source Code Reorganization (`src/`)

**Current:** All modules flat in `src/`
**Proposed:** Organized by domain:
- `src/core/` - Core business logic (download, clean, features, train, backtest, identify)
- `src/analysis/` - Analysis modules (analyze_features, analyze_stop_losses, etc.)
- `src/data/` - Data management (inspect_parquet, download_vix)
- `src/features/` - Feature management (feature_set_manager, etc.)
- `src/utils/` - Utility scripts (tune_threshold, backtest)

**Rationale:** Clear separation of concerns, easier to navigate, follows domain-driven design principles.

#### B. Test Suite Reorganization (`tests/`)

**Current:** Mix of test files in root + some in `tests/`
**Proposed:** Organized by test type:
- `tests/unit/` - Unit tests (test individual functions/modules)
  - `test_features/` - Feature tests
  - `test_models/` - Model tests
  - `test_data/` - Data tests
- `tests/integration/` - Integration tests (test full workflows)
- `tests/performance/` - Performance/benchmark tests

**Rationale:** Standard testing structure, easier to run specific test suites, clear test categorization.

#### C. Documentation Reorganization (`docs/`)

**Current:** All docs in `info/` folder, flat structure
**Proposed:** Organized by purpose:
- `docs/getting_started/` - Quick start guides
- `docs/features/` - Feature documentation
- `docs/guides/` - Implementation guides
- `docs/optimization/` - Optimization brainstorms
- `docs/gui/` - GUI documentation
- `docs/models/` - Model documentation
- `docs/reference/` - Reference materials
- `docs/archive/` - Historical docs
- `docs/planning/` - Future planning

**Rationale:** Easy to find relevant documentation, clear categorization, professional structure.

#### D. Output Management (`outputs/`)

**Current:** Outputs scattered (root, data/, reports/)
**Proposed:** Centralized output directory:
- `outputs/logs/` - All log files
- `outputs/benchmarks/` - Benchmark results
- `outputs/reports/` - Generated reports
- `outputs/inspections/` - Inspection outputs

**Rationale:** All generated content in one place, easy to clean up, clear separation from source/data.

#### E. Data Organization (`data/`)

**Current:** Mix of data and outputs
**Proposed:**
- `data/features/` - Organized by version (v1, v2, v3_New_Dawn)
- `data/temp/` - Temporary test files
- Move `api_keys/` to `.env` (security)

**Rationale:** Clear data organization, security improvement, temp files isolated.

#### F. Notebooks Organization (`notebooks/`)

**Current:** Mix of formats, unclear purpose
**Proposed:**
- `notebooks/research/` - Active research notebooks
- `notebooks/archive/` - Historical notes
- Convert `.rtf` to `.md` or remove duplicates

**Rationale:** Clear purpose, easier to find active work, archive historical content.

---

## PART 3: NAMING CONVENTIONS

### 3.1 File Naming Standards

#### Python Files:
- **Modules:** `snake_case.py` (e.g., `feature_pipeline.py`)
- **Test Files:** `test_<module_name>.py` (e.g., `test_feature_integrity.py`)
- **Scripts:** `snake_case.py` (e.g., `benchmark_feature_pipeline.py`)

#### Configuration Files:
- **YAML:** `snake_case.yaml` (e.g., `features_v3_New_Dawn.yaml`)
- **JSON:** `snake_case.json` (e.g., `models_registry.json`)

#### Documentation Files:
- **Markdown:** `UPPER_SNAKE_CASE.md` for guides (e.g., `QUICKSTART.md`)
- **Markdown:** `snake_case.md` for detailed docs (e.g., `feature_guide.md`)

#### Data Files:
- **Parquet:** `<ticker>.parquet` (e.g., `AAPL.parquet`)
- **CSV:** `<descriptive_name>_<timestamp>.csv` (e.g., `backtest_20260126_120000.csv`)

### 3.2 Directory Naming Standards

- **Lowercase with underscores:** `feature_sets/`, `backtest_results/`
- **No spaces:** Use underscores or hyphens
- **Plural for collections:** `features/`, `models/`, `tests/`
- **Singular for single-purpose:** `config/`, `data/`

### 3.3 Model/Artifact Naming

**Current:** `model_<horizon>d_<threshold>pct_<feature_set>_<num_feat>feat_<tuned>_<cv>_<timestamp>`

**Proposed:** Keep current format but organize in folders:
```
models/shap_artifacts/
├── v3_New_Dawn/
│   ├── 20d_15pct/
│   │   ├── 2026-01-26/
│   │   │   └── model_20d_15pct_v3_New_Dawn_263feat_notuned_nocv_20260126_202217/
```

**Rationale:** Easier to find models by feature set and date, cleaner structure.

---

## PART 4: MIGRATION PLAN

### 4.1 Phase 1: Create New Structure (Non-Breaking)

**Goal:** Create new folders without moving files yet.

**Steps:**
1. Create `docs/` structure with all subdirectories
2. Create `outputs/` structure with subdirectories
3. Create `tests/unit/`, `tests/integration/`, `tests/performance/`
4. Create `src/core/`, `src/analysis/`, `src/data/`, `src/features/`, `src/utils/`
5. Create `data/temp/` and `data/features/` structure
6. Create `notebooks/research/` and `notebooks/archive/`

**Risk:** Low - Only creating folders, no file moves yet.

### 4.2 Phase 2: Move Documentation (Low Risk)

**Goal:** Reorganize documentation into `docs/` structure.

**Steps:**
1. Move active documentation to appropriate `docs/` subdirectories
2. Move historical docs to `docs/archive/`
3. Update any internal links in documentation
4. Create `docs/README.md` index

**Risk:** Low - Documentation moves don't affect code execution.

**Files to Move:**
- `info/QUICKSTART.md` → `docs/getting_started/QUICKSTART.md`
- `info/FEATURE_GUIDE.md` → `docs/features/FEATURE_GUIDE.md`
- `info/NEW_DAWN_IMPLEMENTATION_ROADMAP.md` → `docs/features/NEW_DAWN_IMPLEMENTATION_ROADMAP.md`
- etc.

### 4.3 Phase 3: Move Test Files (Medium Risk)

**Goal:** Move all test files from root to `tests/` structure.

**Steps:**
1. Move unit tests to `tests/unit/test_features/`, etc.
2. Move integration tests to `tests/integration/`
3. Move benchmark/profile scripts to `tests/performance/`
4. Update imports in test files (if needed)
5. Verify tests still run

**Risk:** Medium - Need to verify imports and test execution.

**Files to Move:**
- `test_block_1_1.py` → `tests/unit/test_features/test_block_1_1.py`
- `test_feature_integrity.py` → `tests/unit/test_features/test_feature_integrity.py`
- `benchmark_feature_pipeline.py` → `tests/performance/benchmark_feature_pipeline.py`
- etc.

### 4.4 Phase 4: Reorganize Source Code (High Risk)

**Goal:** Reorganize `src/` into domain-based structure.

**Steps:**
1. Create new directory structure in `src/`
2. Move files to appropriate subdirectories
3. Update all imports across the codebase
4. Update `__init__.py` files
5. Run full test suite
6. Fix any broken imports

**Risk:** High - Many imports need updating, could break functionality.

**Files to Move:**
- `src/download_data.py` → `src/core/download_data.py`
- `src/analyze_features.py` → `src/analysis/analyze_features.py`
- `src/inspect_parquet.py` → `src/data/inspect_parquet.py`
- etc.

**Import Updates Required:**
- Update `gui/services.py` imports
- Update `run_gui.py` imports
- Update test file imports
- Update batch script imports

### 4.5 Phase 5: Move Output Files (Low Risk)

**Goal:** Move all output files to `outputs/` directory.

**Steps:**
1. Move `benchmark_results.txt` → `outputs/benchmarks/`
2. Move `feature_pipeline.log` → `outputs/logs/`
3. Move `data/inspect_parquet/` → `outputs/inspections/`
4. Update code that writes to these locations
5. Update `.gitignore` if needed

**Risk:** Low - Output file moves, just need to update write paths.

### 4.6 Phase 6: Reorganize Data (Medium Risk)

**Goal:** Clean up data folder structure.

**Steps:**
1. Move `data/temp_feature_test_*` → `data/temp/feature_test_*`
2. Reorganize feature data into `data/features/v1/`, etc.
3. Move `data/api_keys/` contents to `.env` (manual step)
4. Update code that references these paths

**Risk:** Medium - Need to update data paths in code.

### 4.7 Phase 7: Clean Up Notebooks (Low Risk)

**Goal:** Organize notebooks folder.

**Steps:**
1. Move historical files to `notebooks/archive/`
2. Convert or remove duplicate `.rtf` files
3. Create `notebooks/README.md`
4. Move `tickererror.txt` to `outputs/logs/` if it's a log

**Risk:** Low - Notebooks are reference material.

### 4.8 Phase 8: Update Configuration & Documentation

**Goal:** Update all references to new structure.

**Steps:**
1. Update `README.md` with new structure
2. Update batch scripts with new paths
3. Update `.gitignore` for new output locations
4. Create `.env.example` template
5. Update any hardcoded paths in code

**Risk:** Low - Documentation and config updates.

---

## PART 5: DETAILED FILE MAPPING

### 5.1 Root Directory Files

| Current Location | Proposed Location | Action | Priority |
|-----------------|-------------------|--------|----------|
| `test_block_1_1.py` | `tests/unit/test_features/test_block_1_1.py` | Move | High |
| `test_feature_integrity.py` | `tests/unit/test_features/test_feature_integrity.py` | Move | High |
| `test_label_calculation.py` | `tests/unit/test_features/test_label_calculation.py` | Move | High |
| `test_optimization_integrity.py` | `tests/unit/test_models/test_optimization_integrity.py` | Move | High |
| `test_phase3_comprehensive.py` | `tests/integration/test_phase3_comprehensive.py` | Move | High |
| `test_phase3_imports.py` | `tests/integration/test_phase3_imports.py` | Move | High |
| `test_spy_data_optimization.py` | `tests/unit/test_data/test_spy_data_optimization.py` | Move | High |
| `benchmark_feature_pipeline.py` | `tests/performance/benchmark_feature_pipeline.py` | Move | High |
| `profile_feature_pipeline.py` | `tests/performance/profile_feature_pipeline.py` | Move | High |
| `compare_benchmark_results.py` | `tests/performance/compare_benchmark_results.py` | Move | High |
| `benchmark_results.txt` | `outputs/benchmarks/benchmark_results.txt` | Move | Medium |
| `feature_pipeline.log` | `outputs/logs/feature_pipeline.log` | Move | Medium |
| `README.md` | `README.md` | Keep | - |
| `requirements.txt` | `requirements.txt` | Keep | - |
| `run_gui.py` | `run_gui.py` | Keep | - |

### 5.2 Source Code Files

| Current Location | Proposed Location | Action | Priority |
|-----------------|-------------------|--------|----------|
| `src/download_data.py` | `src/core/download_data.py` | Move | High |
| `src/clean_data.py` | `src/core/clean_data.py` | Move | High |
| `src/feature_pipeline.py` | `src/core/feature_pipeline.py` | Move | High |
| `src/train_model.py` | `src/core/train_model.py` | Move | High |
| `src/enhanced_backtest.py` | `src/core/enhanced_backtest.py` | Move | High |
| `src/identify_trades.py` | `src/core/identify_trades.py` | Move | High |
| `src/shap_service.py` | `src/core/shap_service.py` | Move | High |
| `src/swing_trade_app.py` | `src/swing_trade_app.py` | Keep (entry point) | - |
| `src/analyze_features.py` | `src/analysis/analyze_features.py` | Move | High |
| `src/analyze_stop_losses.py` | `src/analysis/analyze_stop_losses.py` | Move | High |
| `src/compare_filters.py` | `src/analysis/compare_filters.py` | Move | High |
| `src/apply_entry_filters.py` | `src/analysis/apply_entry_filters.py` | Move | High |
| `src/inspect_parquet.py` | `src/data/inspect_parquet.py` | Move | High |
| `src/inspect_parquet_gui.py` | `src/data/inspect_parquet_gui.py` | Move | High |
| `src/download_vix.py` | `src/data/download_vix.py` | Move | High |
| `src/feature_set_manager.py` | `src/features/feature_set_manager.py` | Move | High |
| `src/manage_feature_sets.py` | `src/features/manage_feature_sets.py` | Move | High |
| `src/apply_feature_pruning.py` | `src/features/apply_feature_pruning.py` | Move | High |
| `src/clean_features_labeled.py` | `src/features/clean_features_labeled.py` | Move | High |
| `src/tune_threshold.py` | `src/utils/tune_threshold.py` | Move | High |
| `src/backtest.py` | `src/utils/backtest.py` | Move | High |

### 5.3 Documentation Files

| Current Location | Proposed Location | Action | Priority |
|-----------------|-------------------|--------|----------|
| `info/README.md` | `docs/README.md` | Move | High |
| `info/QUICKSTART.md` | `docs/getting_started/QUICKSTART.md` | Move | High |
| `info/INSTALL.md` | `docs/getting_started/INSTALL.md` | Move | High |
| `info/APP_README.md` | `docs/getting_started/APP_README.md` | Move | High |
| `info/FEATURE_GUIDE.md` | `docs/features/FEATURE_GUIDE.md` | Move | High |
| `info/FEATURE_SETS_GUIDE.md` | `docs/features/FEATURE_SETS_GUIDE.md` | Move | High |
| `info/NEW_DAWN_IMPLEMENTATION_ROADMAP.md` | `docs/features/NEW_DAWN_IMPLEMENTATION_ROADMAP.md` | Move | High |
| `info/NEW_DAWN_FEATURE_CHECKLIST.md` | `docs/features/NEW_DAWN_FEATURE_CHECKLIST.md` | Move | High |
| `info/FEATURE_REDUNDANCY_GUIDE.md` | `docs/features/FEATURE_REDUNDANCY_GUIDE.md` | Move | High |
| `info/PIPELINE_STEPS.md` | `docs/guides/PIPELINE_STEPS.md` | Move | High |
| `info/FEATURE_SET_ISOLATION_IMPLEMENTATION.md` | `docs/guides/FEATURE_SET_ISOLATION_IMPLEMENTATION.md` | Move | High |
| `info/LOOKAHEAD_BIAS_PREVENTION_GUIDE.md` | `docs/guides/LOOKAHEAD_BIAS_PREVENTION_GUIDE.md` | Move | High |
| `info/TEST_GUIDE.md` | `docs/guides/TEST_GUIDE.md` | Move | High |
| `info/TECHNICAL_CALCULATIONS_OPTIMIZATION_BRAINSTORM.md` | `docs/optimization/TECHNICAL_CALCULATIONS_OPTIMIZATION_BRAINSTORM.md` | Move | High |
| `info/FEATURE_OPTIMIZATION_GAMEPLAN.md` | `docs/optimization/FEATURE_OPTIMIZATION_GAMEPLAN.md` | Move | High |
| `info/PERFORMANCE_DEBUGGING_BRAINSTORM.md` | `docs/optimization/PERFORMANCE_DEBUGGING_BRAINSTORM.md` | Move | High |
| `info/GUI_PHASE1_SUMMARY.md` | `docs/gui/GUI_PHASE1_SUMMARY.md` | Move | Medium |
| `info/GUI_PHASE2_SUMMARY.md` | `docs/gui/GUI_PHASE2_SUMMARY.md` | Move | Medium |
| `info/GUI_PHASE3_COMPLETE.md` | `docs/gui/GUI_PHASE3_COMPLETE.md` | Move | Medium |
| `info/GUI_PHASE4_COMPLETE.md` | `docs/gui/GUI_PHASE4_COMPLETE.md` | Move | Medium |
| `info/MODEL_IMPROVEMENT_PLAN.md` | `docs/models/MODEL_IMPROVEMENT_PLAN.md` | Move | High |
| `info/MODEL_PERFORMANCE_ANALYSIS.md` | `docs/models/MODEL_PERFORMANCE_ANALYSIS.md` | Move | High |
| `info/model_training_overview.md` | `docs/models/model_training_overview.md` | Move | High |
| `info/DATA_SOURCES_EXPLAINED.md` | `docs/reference/DATA_SOURCES_EXPLAINED.md` | Move | Medium |
| `info/DOWNLOAD_ERROR_FIXES.md` | `docs/reference/DOWNLOAD_ERROR_FIXES.md` | Move | Medium |
| `info/DOWNLOAD_IMPROVEMENTS_SUMMARY.md` | `docs/reference/DOWNLOAD_IMPROVEMENTS_SUMMARY.md` | Move | Medium |
| `info/SEC_EDGAR_SETUP.md` | `docs/reference/SEC_EDGAR_SETUP.md` | Move | Medium |
| `info/BENCHMARK_GUIDE.md` | `docs/reference/BENCHMARK_GUIDE.md` | Move | Medium |
| `info/PHASE1_FEATURE_ANALYSIS.md` | `docs/archive/PHASE1_FEATURE_ANALYSIS.md` | Move | Low |
| `info/PHASE1_IMPLEMENTATION_SUMMARY.md` | `docs/archive/PHASE1_IMPLEMENTATION_SUMMARY.md` | Move | Low |
| `info/BLOCK_1_1_IMPLEMENTATION_SUMMARY.md` | `docs/archive/BLOCK_1_1_IMPLEMENTATION_SUMMARY.md` | Move | Low |
| `info/ROADMAP_P4-9.md` | `docs/archive/ROADMAP_P4-9.md` | Move | Low |
| `info/FEATURE_ROADMAP.md` | `docs/archive/FEATURE_ROADMAP.md` | Move | Low |
| `info/FEATURE_IMPROVEMENT_PLAN.md` | `docs/archive/FEATURE_IMPROVEMENT_PLAN.md` | Move | Low |
| `info/FUTURE_FEATURES.md` | `docs/planning/FUTURE_FEATURES.md` | Move | Low |
| `info/GAIN_PROBABILITY_OPTIMIZATION_BRAINSTORM.md` | `docs/planning/GAIN_PROBABILITY_OPTIMIZATION_BRAINSTORM.md` | Move | Low |
| `info/Filters.txt` | `config/filters.txt` or `data/filter_presets/filters.txt` | Move | Medium |
| `info/train_results.txt` | `outputs/reports/train_results.txt` | Move | Low |

### 5.4 Notebook Files

| Current Location | Proposed Location | Action | Priority |
|-----------------|-------------------|--------|----------|
| `notebooks/Comprehensive Feature Guide.rtf` | `notebooks/archive/Comprehensive Feature Guide.rtf` | Move | Low |
| `notebooks/Comprehensive Feature Guide.txt` | `notebooks/archive/Comprehensive Feature Guide.txt` | Move | Low |
| `notebooks/original_roadmap.txt` | `notebooks/archive/original_roadmap.txt` | Move | Low |
| `notebooks/phase 3 Road Map.txt` | `notebooks/archive/phase_3_roadmap.txt` | Move & Rename | Low |
| `notebooks/phase_3_roadmap.txt` | `notebooks/archive/phase_3_roadmap.txt` | Move | Low |
| `notebooks/feature_checklist.txt` | `notebooks/research/feature_checklist.txt` | Move | Medium |
| `notebooks/feature_list.txt` | `notebooks/research/feature_list.txt` | Move | Medium |
| `notebooks/SearchParams.txt` | `notebooks/research/SearchParams.txt` | Move | Medium |
| `notebooks/tickererror.txt` | `outputs/logs/tickererror.txt` | Move | Low |

### 5.5 Data Files

| Current Location | Proposed Location | Action | Priority |
|-----------------|-------------------|--------|----------|
| `data/features_labeled_v2/` | `data/features/v2/` | Move | Medium |
| `data/features_labeled_v3_New_Dawn/` | `data/features/v3_New_Dawn/` | Move | Medium |
| `data/temp_feature_test_input/` | `data/temp/feature_test_input/` | Move | Low |
| `data/temp_feature_test_output/` | `data/temp/feature_test_output/` | Move | Low |
| `data/inspect_parquet/` | `outputs/inspections/` | Move | Medium |
| `data/api_keys/` | `.env` (manual migration) | Move | High (security) |

### 5.6 Reports Files

| Current Location | Proposed Location | Action | Priority |
|-----------------|-------------------|--------|----------|
| `reports/backtest_filtered_20251210_030831.csv` | `outputs/reports/backtest_filtered_20251210_030831.csv` | Move | Low |
| `reports/Filters.csv` | `config/filters.csv` or `data/filter_presets/filters.csv` | Move | Medium |

---

## PART 6: IMPORT PATH UPDATES

### 6.1 Source Code Imports

After moving files, update imports throughout the codebase:

**Current:**
```python
from src.download_data import download_stock_data
from src.analyze_features import analyze_feature_importance
```

**Proposed:**
```python
from src.core.download_data import download_stock_data
from src.analysis.analyze_features import analyze_feature_importance
```

### 6.2 Test Imports

**Current:**
```python
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

**Proposed:**
```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

### 6.3 GUI Imports

**Current (in `gui/services.py`):**
```python
from src.identify_trades import identify_trades
from src.train_model import train_model
```

**Proposed:**
```python
from src.core.identify_trades import identify_trades
from src.core.train_model import train_model
```

### 6.4 Batch Script Imports

Update batch scripts in `bats/` to use new paths:
- Update Python paths if needed
- Update data paths if moved

---

## PART 7: CONFIGURATION UPDATES

### 7.1 .gitignore Updates

Add new output directories:
```
# Outputs
outputs/logs/
outputs/benchmarks/
outputs/reports/
outputs/inspections/
*.log

# Temporary files
data/temp/
```

### 7.2 Environment Variables

Create `.env.example`:
```
# API Keys
FRED_API_KEY=your_fred_api_key_here

# Data Paths (if custom)
DATA_ROOT=data
MODELS_ROOT=models
OUTPUTS_ROOT=outputs
```

### 7.3 Path Configuration

Consider creating `config/paths.yaml`:
```yaml
data:
  raw: data/raw
  clean: data/clean
  features: data/features
  backtest_results: data/backtest_results
  
models:
  root: models
  shap_artifacts: models/shap_artifacts
  
outputs:
  logs: outputs/logs
  benchmarks: outputs/benchmarks
  reports: outputs/reports
  inspections: outputs/inspections
```

---

## PART 8: TESTING STRATEGY

### 8.1 Pre-Migration Testing

Before starting migration:
1. Run full test suite
2. Document current test results
3. Create backup branch
4. Verify all functionality works

### 8.2 During Migration Testing

After each phase:
1. Run affected tests
2. Verify imports work
3. Check GUI functionality
4. Verify CLI commands work

### 8.3 Post-Migration Testing

After complete migration:
1. Run full test suite
2. Test GUI end-to-end
3. Test CLI pipeline end-to-end
4. Verify all outputs go to correct locations
5. Check documentation links

---

## PART 9: ROLLBACK PLAN

### 9.1 Git Strategy

1. **Create feature branch:** `git checkout -b refactor/project-structure`
2. **Commit after each phase:** Small, incremental commits
3. **Tag before major changes:** `git tag pre-refactor-v1`
4. **Keep backup branch:** `git branch backup/pre-refactor`

### 9.2 Rollback Steps

If issues arise:
1. Revert to last working commit
2. Identify problematic phase
3. Fix issues in isolation
4. Re-apply changes incrementally

---

## PART 10: ONGOING MAINTENANCE PLAN

### 10.1 Output File Management

**Problem:** Output files accumulating in root or wrong locations.

**Solution:**
1. All scripts should write to `outputs/` subdirectories
2. Add output path validation in scripts
3. Create utility function for output paths:
   ```python
   def get_output_path(category: str, filename: str) -> Path:
       """Get standardized output path."""
       output_dir = Path("outputs") / category
       output_dir.mkdir(parents=True, exist_ok=True)
       return output_dir / filename
   ```

### 10.2 Test File Management

**Problem:** Test files created in root during development.

**Solution:**
1. Always create tests in appropriate `tests/` subdirectory
2. Use test discovery: `pytest tests/`
3. Add pre-commit hook to check for test files in root

### 10.3 Documentation Management

**Problem:** New documentation files created without organization.

**Solution:**
1. Create `docs/TEMPLATE.md` for new docs
2. Add documentation guidelines to `docs/README.md`
3. Review documentation quarterly for organization

### 10.4 Temporary File Management

**Problem:** Temp files accumulating in various locations.

**Solution:**
1. All temp files go to `data/temp/`
2. Add cleanup script: `scripts/maintenance/cleanup_temp.py`
3. Add temp file cleanup to `.gitignore`
4. Consider using `tempfile` module for truly temporary files

### 10.5 Model Artifact Management

**Problem:** SHAP artifacts accumulating without organization.

**Solution:**
1. Organize by feature set and date (see structure above)
2. Create cleanup script for old artifacts
3. Add retention policy (e.g., keep last 10 models per config)
4. Archive old models to `models/archive/`

---

## PART 11: BEST PRACTICES GOING FORWARD

### 11.1 File Creation Guidelines

**Before creating a new file, ask:**
1. **Is this source code?** → `src/` (appropriate subdirectory)
2. **Is this a test?** → `tests/` (unit/integration/performance)
3. **Is this documentation?** → `docs/` (appropriate category)
4. **Is this a script?** → `scripts/` (appropriate subdirectory)
5. **Is this output?** → `outputs/` (appropriate category)
6. **Is this temporary?** → `data/temp/` or use `tempfile` module
7. **Is this configuration?** → `config/`

### 11.2 Naming Guidelines

1. **Use descriptive names:** `analyze_feature_importance.py` not `analyze.py`
2. **Follow conventions:** snake_case for Python, UPPER_SNAKE for guides
3. **Include context:** `test_feature_integrity.py` not `test_integrity.py`
4. **Avoid spaces:** Use underscores or hyphens
5. **Be consistent:** Follow existing patterns

### 11.3 Import Guidelines

1. **Use absolute imports:** `from src.core.download_data import ...`
2. **Group imports:** Standard library, third-party, local
3. **Avoid circular imports:** Keep dependencies clear
4. **Use `__init__.py`:** Export public API from packages

### 11.4 Documentation Guidelines

1. **Update docs when code changes:** Keep docs in sync
2. **Use consistent format:** Follow existing doc style
3. **Link related docs:** Cross-reference when helpful
4. **Archive old docs:** Don't delete, move to archive

### 11.5 Testing Guidelines

1. **Write tests alongside code:** Don't defer testing
2. **Use appropriate test type:** Unit vs integration vs performance
3. **Keep tests organized:** Follow test directory structure
4. **Run tests before committing:** Verify nothing breaks

---

## PART 12: SUMMARY & RECOMMENDATIONS

### 12.1 Critical Issues to Address

1. **Root Directory Clutter** - 11 files that don't belong in root
2. **Test File Organization** - Tests scattered, need proper structure
3. **Output File Management** - Outputs in multiple locations
4. **Documentation Organization** - 26 files in flat structure
5. **Source Code Organization** - All modules flat in `src/`
6. **Security Concern** - API keys in `data/api_keys/` should be in `.env`

### 12.2 Recommended Migration Order

1. **Phase 1:** Create new structure (no risk)
2. **Phase 2:** Move documentation (low risk)
3. **Phase 3:** Move test files (medium risk, but isolated)
4. **Phase 4:** Move output files (low risk)
5. **Phase 5:** Reorganize source code (high risk, do last)
6. **Phase 6:** Clean up data (medium risk)
7. **Phase 7:** Update configs and docs (low risk)

### 12.3 Estimated Time

- **Phase 1:** 30 minutes (create folders)
- **Phase 2:** 1-2 hours (move docs, update links)
- **Phase 3:** 2-3 hours (move tests, update imports, verify)
- **Phase 4:** 1 hour (move outputs, update write paths)
- **Phase 5:** 4-6 hours (reorganize src, update all imports, test)
- **Phase 6:** 1-2 hours (reorganize data, update paths)
- **Phase 7:** 1 hour (update configs, docs)

**Total:** 10-15 hours of focused work

### 12.4 Benefits of Reorganization

1. **Professional Structure** - Master's level organization
2. **Easier Navigation** - Clear separation of concerns
3. **Better Maintainability** - Easier to find and update files
4. **Scalability** - Structure supports growth
5. **Team Readiness** - Easy for new contributors to understand
6. **Clean Root** - No clutter, professional appearance
7. **Better Testing** - Organized test structure
8. **Output Management** - Centralized output location

### 12.5 Risks & Mitigation

**Risk:** Breaking imports and functionality
**Mitigation:** 
- Incremental migration
- Test after each phase
- Keep backup branch
- Update imports systematically

**Risk:** Missing some file references
**Mitigation:**
- Use IDE refactoring tools
- Search codebase for old paths
- Comprehensive testing

**Risk:** Time investment
**Mitigation:**
- Do in phases during low-activity periods
- Can pause between phases
- Benefits outweigh costs long-term

---

## PART 13: IMPLEMENTATION CHECKLIST

### Pre-Migration
- [ ] Create backup branch
- [ ] Run full test suite (baseline)
- [ ] Document current structure
- [ ] Review this blueprint

### Phase 1: Create Structure
- [ ] Create `docs/` structure
- [ ] Create `outputs/` structure
- [ ] Create `tests/` subdirectories
- [ ] Create `src/` subdirectories
- [ ] Create `data/temp/` and `data/features/`
- [ ] Create `notebooks/research/` and `notebooks/archive/`

### Phase 2: Move Documentation
- [ ] Move active docs to `docs/`
- [ ] Move historical docs to `docs/archive/`
- [ ] Create `docs/README.md` index
- [ ] Update internal doc links
- [ ] Verify all docs accessible

### Phase 3: Move Test Files
- [ ] Move unit tests
- [ ] Move integration tests
- [ ] Move performance tests
- [ ] Update test imports
- [ ] Run test suite
- [ ] Fix any broken tests

### Phase 4: Move Output Files
- [ ] Move log files
- [ ] Move benchmark results
- [ ] Move inspection outputs
- [ ] Update code that writes outputs
- [ ] Verify outputs go to correct locations

### Phase 5: Reorganize Source Code
- [ ] Move core modules
- [ ] Move analysis modules
- [ ] Move data modules
- [ ] Move feature modules
- [ ] Move utility modules
- [ ] Update all imports
- [ ] Update `__init__.py` files
- [ ] Run full test suite
- [ ] Test GUI
- [ ] Test CLI

### Phase 6: Reorganize Data
- [ ] Move feature data to `data/features/`
- [ ] Move temp files to `data/temp/`
- [ ] Move inspection outputs to `outputs/inspections/`
- [ ] Migrate API keys to `.env`
- [ ] Update data paths in code
- [ ] Verify data access works

### Phase 7: Clean Up Notebooks
- [ ] Move historical notebooks to archive
- [ ] Organize research notebooks
- [ ] Create `notebooks/README.md`
- [ ] Remove duplicates

### Phase 8: Update Configuration
- [ ] Update `README.md`
- [ ] Update batch scripts
- [ ] Update `.gitignore`
- [ ] Create `.env.example`
- [ ] Update hardcoded paths

### Post-Migration
- [ ] Run full test suite
- [ ] Test GUI end-to-end
- [ ] Test CLI pipeline end-to-end
- [ ] Verify all outputs
- [ ] Update team documentation
- [ ] Commit changes
- [ ] Create migration summary

---

## CONCLUSION

This blueprint provides a comprehensive plan for reorganizing the SwingTradeV2 project to professional/master's level standards. The proposed structure:

- **Separates concerns** clearly (source, tests, docs, data, outputs)
- **Eliminates root clutter** (moves 11+ files to appropriate locations)
- **Organizes documentation** (26 files into logical categories)
- **Standardizes naming** (consistent conventions throughout)
- **Improves maintainability** (easier to find and update files)
- **Scales with growth** (structure supports project expansion)

The migration is designed to be **incremental and safe**, with testing at each phase and a clear rollback plan. The estimated time investment (10-15 hours) is justified by the long-term benefits in maintainability, scalability, and professional appearance.

**Next Steps:**
1. Review this blueprint
2. Create backup branch
3. Begin Phase 1 (create structure)
4. Proceed incrementally through phases
5. Test thoroughly at each step

---

**Document Version:** 1.0  
**Last Updated:** January 26, 2026  
**Status:** Ready for Implementation
