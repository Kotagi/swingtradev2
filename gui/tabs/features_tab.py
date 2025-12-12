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
from pathlib import Path
from datetime import datetime

from gui.services import FeatureService, DataService


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
        
        self.init_ui()
    
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
        
        # Horizon
        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel("Trade Horizon (days):"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 365)
        self.horizon_spin.setValue(5)
        horizon_row.addWidget(self.horizon_spin)
        horizon_row.addStretch()
        params_layout.addLayout(horizon_row)
        
        # Return threshold (as percentage)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Return Threshold:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setSingleStep(1)
        self.threshold_spin.setValue(0)
        self.threshold_spin.setSuffix("%")
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        params_layout.addLayout(threshold_row)
        
        # Feature set
        featureset_row = QHBoxLayout()
        featureset_row.addWidget(QLabel("Feature Set:"))
        self.featureset_combo = QComboBox()
        self.featureset_combo.setEditable(True)
        self.featureset_combo.setMinimumWidth(200)
        self.featureset_combo.lineEdit().setPlaceholderText("Leave empty for default")
        # Only one feature set now; keep empty for default
        self.featureset_combo.addItems([""])
        featureset_row.addWidget(self.featureset_combo)
        featureset_row.addStretch()
        params_layout.addLayout(featureset_row)
        
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
        threshold_percent = self.threshold_spin.value()
        threshold_decimal = threshold_percent / 100.0
        
        kwargs = {
            "horizon": self.horizon_spin.value(),
            "threshold": threshold_decimal,
            "full": self.full_features_check.isChecked()
        }
        
        # Feature set
        feature_set = self.featureset_combo.currentText().strip()
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
    
    def on_build_finished(self, success: bool, message: str):
        """Handle completion of feature building."""
        # Re-enable button
        self.build_btn.setEnabled(True)
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

