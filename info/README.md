# Swing Trading ML Application - GUI

This is the graphical user interface for the Swing Trading ML Application, built with PyQt6.

## Features (Phase 1)

### ✅ Implemented

- **Main Window**: Tabbed interface with modern dark theme
- **Trade Identification Tab**: Fully functional tab for identifying trading opportunities
  - Model loading and selection
  - Configurable identification parameters (probability threshold, top N)
  - Entry filters (recommended filters from stop-loss analysis)
  - Stop-loss configuration (adaptive ATR, swing ATR)
  - Results table with sorting
  - Export to CSV
  - Background processing (non-blocking UI)

### 🚧 Placeholder Tabs (Future Implementation)

- Dashboard
- Data Management
- Feature Engineering
- Model Training
- Backtesting
- Analysis & Reports

## Running the GUI

### Option 1: Using the launcher script
```bash
python run_gui.py
```

### Option 2: Direct execution
```bash
python gui/app.py
```

## Requirements

- PyQt6 >= 6.6.0 (automatically installed via requirements.txt)
- All other dependencies from the main application

## Architecture

### Service Layer (`gui/services.py`)
- Wraps CLI functions for use in GUI
- Provides clean interface between UI and business logic
- `TradeIdentificationService`: Handles model loading and opportunity identification
- `DataService`: Provides data paths and utilities

### Main Window (`gui/main_window.py`)
- Tabbed interface
- Status bar
- Central widget management

### Tabs (`gui/tabs/`)
- `identify_tab.py`: Trade identification functionality
- Future tabs will be added here

### Styling (`gui/styles.py`)
- Dark theme with modern color scheme
- Consistent styling across all components
- Custom stylesheet for enhanced appearance

## Usage

### Trade Identification

1. **Load a Model**:
   - Select a model from the dropdown or browse for a model file
   - Click "Load Model"
   - Wait for confirmation message

2. **Configure Parameters**:
   - Set minimum probability threshold (0.0 - 1.0)
   - Set number of top results to return
   - Optionally enable recommended filters
   - Optionally configure stop-loss settings

3. **Identify Opportunities**:
   - Click "Identify Opportunities"
   - Wait for processing (progress bar will show)
   - Results will appear in the table below

4. **Export Results**:
   - Click "Export to CSV" to save results to a file

## Design Philosophy

- **Non-blocking**: All long-running operations run in background threads
- **User-friendly**: Clear labels, tooltips, and status messages
- **Modern UI**: Dark theme with accent colors
- **Extensible**: Easy to add new tabs and functionality

## Future Enhancements

- Complete implementation of all placeholder tabs
- Real-time data updates
- Advanced visualizations (charts, graphs)
- Configuration persistence
- Keyboard shortcuts
- Drag-and-drop file support
- Help/documentation panel

