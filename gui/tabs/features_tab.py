"""
Feature Engineering Tab

Allows users to build features from cleaned data.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QTextEdit, QLineEdit, QFileDialog,
    QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import os
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd

from gui.services import FeatureService, DataService

# Project root for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


class FeatureWorker(QThread):
    """Worker thread for feature building."""
    
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, int)  # completed, total (for progress bar)
    progress_message = pyqtSignal(str)  # progress message text
    
    def __init__(self, service: FeatureService, **kwargs):
        super().__init__()
        self.service = service
        self.kwargs = kwargs
    
    def feature_progress_callback(self, completed: int, total: int):
        """Callback for feature building progress updates."""
        self.progress.emit(completed, total)
        if total > 0:
            pct = (completed / total) * 100
            self.progress_message.emit(f"Processed {completed}/{total} files ({pct:.1f}%)")
    
    def run(self):
        """Run feature building in background thread."""
        try:
            # Add progress callback
            self.kwargs['progress_callback'] = self.feature_progress_callback
            success, message = self.service.build_features(**self.kwargs)
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class FeaturesTab(QWidget):
    """Tab for feature engineering."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = FeatureService()
        self.data_service = DataService()
        self.worker = None
        self.current_feature_set = "v1"  # Default
        self.feature_set_selector = None  # Will be set by main window
        self.dev_tools_enabled = False  # Controlled by MainWindow toggle
        self.test_run_context = None  # Holds info for the dev test run
        
        self.init_ui()
    
    def set_feature_set_selector(self, selector):
        """Set the feature set selector widget (called by main window)."""
        self.feature_set_selector = selector
        # Add it to the UI - replace the display label with the selector
        if hasattr(self, 'featureset_row') and hasattr(self, 'featureset_display'):
            # Find the display label in the layout
            for i in range(self.featureset_row.count()):
                item = self.featureset_row.itemAt(i)
                if item and item.widget() == self.featureset_display:
                    # Remove the display label
                    self.featureset_display.setParent(None)
                    # Insert the selector at the same position
                    self.featureset_row.insertWidget(i, selector)
                    break
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout for the tab
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Content widget
        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Feature Engineering")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Parameters group
        params_group = QGroupBox("Feature Parameters")
        params_layout = QVBoxLayout()
        
        # Note about labels
        note_label = QLabel("Note: Labels are now calculated during training based on your horizon and return threshold settings.")
        note_label.setStyleSheet("color: #b0b0b0; font-style: italic; padding: 10px;")
        note_label.setWordWrap(True)
        params_layout.addWidget(note_label)
        
        # Feature set selector (will be populated by main window)
        # Note: The selector widget already has its own "Feature Set:" label, so we don't add another one
        self.featureset_row = QHBoxLayout()
        # Placeholder - will be replaced with actual selector from main window
        self.featureset_display = QLabel("v1 (default)")
        self.featureset_display.setStyleSheet("color: #00d4aa; font-weight: bold;")
        self.featureset_row.addWidget(self.featureset_display)
        self.featureset_row.addStretch()
        params_layout.addLayout(self.featureset_row)
        
        # Options
        options_row = QHBoxLayout()
        self.full_features_check = QCheckBox("Force Full Recompute")
        self.full_features_check.setChecked(True)
        options_row.addWidget(self.full_features_check)
        options_row.addStretch()
        params_layout.addLayout(options_row)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Paths group (optional)
        paths_group = QGroupBox("Paths (Optional - Leave empty for defaults)")
        paths_layout = QVBoxLayout()
        
        # Input dir
        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input Directory:"))
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Leave empty for default (data/clean)")
        input_row.addWidget(self.input_dir_edit)
        browse_input_btn = QPushButton("Browse...")
        browse_input_btn.clicked.connect(self.browse_input_dir)
        input_row.addWidget(browse_input_btn)
        paths_layout.addLayout(input_row)
        
        # Output dir
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output Directory:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Leave empty for default (data/features_labeled)")
        output_row.addWidget(self.output_dir_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        output_row.addWidget(browse_output_btn)
        paths_layout.addLayout(output_row)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Build button
        self.build_btn = QPushButton("Build Features")
        self.build_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #000000;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #00c896;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        self.build_btn.clicked.connect(self.build_features)
        layout.addWidget(self.build_btn)
        
        # Dev-only test button (single ticker build + validation) - visibility controlled by toggle
        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test")
        self.test_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: #ffffff;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        self.test_btn.clicked.connect(self.run_feature_test)
        test_row.addWidget(self.test_btn)
        test_row.addStretch()
        layout.addLayout(test_row)
        
        # Delete feature data group
        delete_group = QGroupBox("Delete Feature Data")
        delete_layout = QVBoxLayout()
        
        delete_info = QLabel("Delete all feature data from the output directory.")
        delete_info.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        delete_layout.addWidget(delete_info)
        
        self.delete_btn = QPushButton("Delete Feature Data")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: #ffffff;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_feature_data)
        delete_layout.addWidget(self.delete_btn)
        
        delete_group.setLayout(delete_layout)
        layout.addWidget(delete_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setFormat("%p% (%v/%m)")
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Log output
        log_label = QLabel("Output Log")
        log_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setMaximumHeight(300)
        layout.addWidget(self.log_text)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def browse_input_dir(self):
        """Browse for input directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            str(Path(self.data_service.get_clean_dir()))
        )
        if dir_path:
            self.input_dir_edit.setText(dir_path)
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            str(Path(self.data_service.get_data_dir()))
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def build_features(self):
        """Start feature building."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "Feature building is already in progress.")
            return
        
        # Disable button
        self.build_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing feature building...")
        self.status_label.setStyleSheet("color: #ff9800;")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting feature building...")
        
        # Get paths
        input_dir = self.input_dir_edit.text().strip()
        if not input_dir:
            input_dir = self.data_service.get_clean_dir()
        
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            output_dir = self.service.get_output_dir()
        
        # Get total files for progress bar
        total_files = self.service.get_total_input_files(input_dir)
        if total_files > 0:
            self.progress_bar.setRange(0, total_files)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate if we can't get total
        
        # Get parameters
        # Convert percentage to decimal (e.g., 5 -> 0.05)
        kwargs = {
            "full": self.full_features_check.isChecked()
        }
        
        # Feature set - get from main window's global selector
        feature_set = self.get_current_feature_set()
        if feature_set:
            kwargs["feature_set"] = feature_set
        
        # Paths (only if specified)
        if self.input_dir_edit.text().strip():
            kwargs["input_dir"] = input_dir
        if self.output_dir_edit.text().strip():
            kwargs["output_dir"] = output_dir
        
        # Create worker
        self.worker = FeatureWorker(self.service, **kwargs)
        self.worker.finished.connect(self.on_build_finished)
        self.worker.progress.connect(self.on_feature_progress)
        self.worker.progress_message.connect(self.on_progress_message)
        self.worker.start()
    
    def on_feature_progress(self, completed: int, total: int):
        """Handle feature building progress update."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)
    
    def on_progress_message(self, message: str):
        """Handle progress message update."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #ff9800;")
    
    def delete_feature_data(self):
        """Delete all feature data from the output directory."""
        from pathlib import Path
        
        # Get output directory
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            output_dir = self.service.get_output_dir()
        
        output_path = Path(output_dir)
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Feature Data",
            f"Are you sure you want to delete ALL feature data from:\n{output_path}\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if not output_path.exists():
                    QMessageBox.information(self, "No Files", "Feature data directory does not exist or is already empty.")
                    return
                
                # Count files before deletion
                parquet_files = list(output_path.glob("*.parquet"))
                file_count = len(parquet_files)
                
                if file_count == 0:
                    QMessageBox.information(self, "No Files", "Feature data directory is already empty.")
                    return
                
                # Delete all Parquet files
                deleted_count = 0
                for parquet_file in parquet_files:
                    try:
                        parquet_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error deleting {parquet_file.name}: {str(e)}")
                
                # Log result
                timestamp = datetime.now().strftime('%H:%M:%S')
                if deleted_count == file_count:
                    self.log_text.append(f"[{timestamp}] ✓ Deleted {deleted_count} feature file(s) from {output_path}")
                    QMessageBox.information(self, "Success", f"Successfully deleted {deleted_count} feature file(s).")
                else:
                    self.log_text.append(f"[{timestamp}] ⚠ Deleted {deleted_count} of {file_count} file(s) (some errors occurred)")
                    QMessageBox.warning(self, "Partial Success", f"Deleted {deleted_count} of {file_count} file(s). Some files could not be deleted.")
                
            except Exception as e:
                timestamp = datetime.now().strftime('%H:%M:%S')
                error_msg = f"Error deleting feature data: {str(e)}"
                self.log_text.append(f"[{timestamp}] ✗ {error_msg}")
                QMessageBox.critical(self, "Error", error_msg)

    # --- Developer test helper ---
    def run_feature_test(self):
        """Build and validate a single ticker into a temp folder (dev-only)."""
        if not self.dev_tools_enabled:
            QMessageBox.information(self, "Developer Tools Disabled", "Enable Developer Tools in Settings to use the test run.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "A feature task is already in progress.")
            return
        
        # Resolve clean input directory
        input_dir = self.input_dir_edit.text().strip() or self.data_service.get_clean_dir()
        input_path = Path(input_dir)
        if not input_path.exists():
            QMessageBox.warning(self, "Invalid Input", f"Input directory does not exist:\n{input_path}")
            return
        
        # Auto-select ticker
        ticker, source_file = self._select_test_ticker(input_path)
        if not ticker or not source_file:
            QMessageBox.warning(self, "No Tickers Found", "Could not find any parquet files to test.")
            return
        
        # Prep temp input/output (overwrite each run)
        temp_input = PROJECT_ROOT / "data" / "temp_feature_test_input"
        temp_output = PROJECT_ROOT / "data" / "temp_feature_test_output"
        shutil.rmtree(temp_input, ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)
        temp_input.mkdir(parents=True, exist_ok=True)
        temp_output.mkdir(parents=True, exist_ok=True)
        
        # Copy single ticker file
        try:
            shutil.copy2(source_file, temp_input / source_file.name)
        except Exception as e:
            QMessageBox.critical(self, "Copy Failed", f"Could not prepare test file:\n{e}")
            return
        
        # Determine config path for current feature set
        feature_set = self.get_current_feature_set()
        config_path = None
        try:
            from feature_set_manager import get_feature_set_config_path
            config_path = get_feature_set_config_path(feature_set)
        except Exception:
            # Fallback to conventional path
            cfg_candidate = PROJECT_ROOT / "config" / f"features_{feature_set}.yaml"
            if cfg_candidate.exists():
                config_path = cfg_candidate
            else:
                config_path = PROJECT_ROOT / "config" / "features.yaml"
        
        # Force full recompute for the single ticker into temp output
        kwargs = {
            "input_dir": str(temp_input),
            "output_dir": str(temp_output),
            "config": str(config_path),
            "full": True
        }
        
        # Track context for validation after build completes
        self.test_run_context = {
            "ticker": ticker,
            "output_dir": temp_output,
            "feature_set": feature_set
        }
        
        # Configure progress UI
        self.build_btn.setEnabled(False)
        if self.dev_tools_enabled and hasattr(self, "test_btn"):
            self.test_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Testing {ticker} ({feature_set})...")
        self.status_label.setStyleSheet("color: #ff9800;")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting test build for {ticker} ({feature_set})")
        
        # Run via worker to avoid UI freeze
        self.worker = FeatureWorker(self.service, **kwargs)
        self.worker.finished.connect(self.on_build_finished)
        self.worker.progress.connect(self.on_feature_progress)
        self.worker.progress_message.connect(self.on_progress_message)
        self.worker.start()

    def _select_test_ticker(self, input_path: Path):
        """Pick ticker in priority order, then first file."""
        priority = ["AAPL", "MSFT", "GOOG", "TSLA"]
        for ticker in priority:
            candidate = input_path / f"{ticker}.parquet"
            if candidate.exists():
                return ticker, candidate
        # Fallback: first parquet file
        parquet_files = sorted(input_path.glob("*.parquet"))
        if parquet_files:
            return parquet_files[0].stem, parquet_files[0]
        return None, None

    def _run_test_validation(self, context: dict, build_success: bool, build_message: str):
        """Validate the test output and show popup summary."""
        ticker = context.get("ticker")
        output_dir: Path = context.get("output_dir")
        feature_set = context.get("feature_set")
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if not build_success:
            QMessageBox.critical(self, "Test Failed", f"Build failed for {ticker} ({feature_set}).\n\n{build_message}")
            self.log_text.append(f"[{timestamp}] ✗ Test build failed: {build_message}")
            return
        
        target_file = output_dir / f"{ticker}.parquet"
        if not target_file.exists():
            QMessageBox.critical(self, "Test Failed", f"Expected output file not found:\n{target_file}")
            self.log_text.append(f"[{timestamp}] ✗ Test output missing: {target_file}")
            return
        
        try:
            df = pd.read_parquet(target_file)
            row_count = len(df)
            cols = list(df.columns)
            
            # Validate structure: date should be index, ticker from filename
            has_date_index = df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex)
            ticker_from_file = target_file.stem  # Filename without extension
            
            # Feature columns = exclude common metadata/label columns (ticker/date are not columns)
            exclude = {"label", "entry_date", "exit_date"}
            feature_cols = [c for c in cols if c not in exclude]
            feature_count = len(feature_cols)
            
            nan_counts = df.isna().sum()
            total_nans = int(nan_counts.sum())
            # Top 5 feature NaNs
            feature_nan_counts = nan_counts[feature_cols] if feature_cols else pd.Series(dtype=int)
            top_nan = feature_nan_counts.sort_values(ascending=False).head(5)
            
            summary_lines = [
                f"Ticker: {ticker} (from filename: {ticker_from_file})",
                f"Feature Set: {feature_set}",
                f"Rows: {row_count}",
                f"Columns: {len(cols)} (features: {feature_count})",
                f"Date Index: {'✓' if has_date_index else '✗'}",
                f"Total NaNs: {total_nans}"
            ]
            if not has_date_index:
                summary_lines.append("⚠ Warning: Date should be the index")
            if not feature_cols:
                summary_lines.append("⚠ Warning: No feature columns found.")
            else:
                formatted_top = "\n".join([f"  {k}: {int(v)}" for k, v in top_nan.items()])
                summary_lines.append("Top NaN counts (features):")
                summary_lines.append(formatted_top if formatted_top else "  None")
            
            message = "\n".join(summary_lines)
            QMessageBox.information(self, "Test Complete", message)
            self.log_text.append(f"[{timestamp}] ✓ Test complete for {ticker} ({feature_set}). Rows={row_count}, Features={feature_count}, NaNs={total_nans}")
        except Exception as e:
            err_msg = f"Validation failed for {ticker}: {e}"
            QMessageBox.critical(self, "Test Failed", err_msg)
            self.log_text.append(f"[{timestamp}] ✗ {err_msg}")

    def set_dev_tools_enabled(self, enabled: bool):
        """Toggle visibility and availability of dev tools."""
        self.dev_tools_enabled = enabled
        if hasattr(self, "test_btn"):
            self.test_btn.setVisible(enabled)
            self.test_btn.setEnabled(enabled and not (self.worker and self.worker.isRunning()))
    
    def get_current_feature_set(self) -> str:
        """Get the current feature set from selector or use default."""
        # Try to get from local selector first
        if self.feature_set_selector:
            try:
                return self.feature_set_selector.get_current_feature_set()
            except Exception:
                pass
        # Fall back to stored value
        return getattr(self, 'current_feature_set', 'v1')
    
    def on_feature_set_changed(self, feature_set: str):
        """Handle feature set change from selector."""
        self.current_feature_set = feature_set
        # Update display if selector hasn't been added yet
        if hasattr(self, 'featureset_display') and self.featureset_display.parent():
            self.featureset_display.setText(f"{feature_set} {'(default)' if feature_set == 'v1' else ''}")
            self.featureset_display.setStyleSheet("color: #00d4aa; font-weight: bold;")
    
    def on_build_finished(self, success: bool, message: str):
        """Handle completion of feature building."""
        # Re-enable button
        self.build_btn.setEnabled(True)
        if hasattr(self, "test_btn"):
            self.test_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if success:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: #4caf50;")
            self.log_text.append(f"[{timestamp}] ✓ {message}")
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            self.log_text.append(f"[{timestamp}] ✗ Error: {message}")
            QMessageBox.critical(self, "Feature Building Failed", message)

        # If this was a dev test run, run validation and show popup
        if self.test_run_context:
            context = self.test_run_context
            self.test_run_context = None
            self._run_test_validation(context, success, message)

