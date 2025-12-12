"""
Data Management Tab

Allows users to download and clean stock data.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QMessageBox, QProgressBar, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
from datetime import datetime

from gui.services import DataService


class DataWorker(QThread):
    """Worker thread for data operations."""
    
    finished = pyqtSignal(bool, str)  # success, message
    progress = pyqtSignal(int, int)  # completed, total (for progress bar)
    progress_message = pyqtSignal(str)  # progress message text
    
    def __init__(self, operation: str, service: DataService, **kwargs):
        super().__init__()
        self.operation = operation
        self.service = service
        self.kwargs = kwargs
    
    def download_progress_callback(self, completed: int, total: int):
        """Callback for download progress updates."""
        self.progress.emit(completed, total)
        if total > 0:
            pct = (completed / total) * 100
            self.progress_message.emit(f"Downloaded {completed}/{total} tickers ({pct:.1f}%)")
    
    def clean_progress_callback(self, completed: int, total: int):
        """Callback for cleaning progress updates."""
        self.progress.emit(completed, total)
        if total > 0:
            pct = (completed / total) * 100
            self.progress_message.emit(f"Cleaned {completed}/{total} files ({pct:.1f}%)")
    
    def run(self):
        """Run the operation in background thread."""
        try:
            if self.operation == "download":
                # Add progress callback for download
                self.kwargs['progress_callback'] = self.download_progress_callback
                success, message = self.service.download_data(**self.kwargs)
            elif self.operation == "clean":
                # Add progress callback for cleaning
                self.kwargs['progress_callback'] = self.clean_progress_callback
                success, message = self.service.clean_data(**self.kwargs)
            else:
                success, message = False, f"Unknown operation: {self.operation}"
            
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class DataTab(QWidget):
    """Tab for data management operations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = DataService()
        self.worker = None
        self.initial_download_count = 0
        
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
        title = QLabel("Data Management")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Download group
        download_group = QGroupBox("Download Stock Data")
        download_layout = QVBoxLayout()
        
        # Tickers file
        tickers_row = QHBoxLayout()
        tickers_row.addWidget(QLabel("Tickers File:"))
        self.tickers_edit = QLineEdit(self.service.get_tickers_file())
        self.tickers_edit.setPlaceholderText("Path to tickers CSV file")
        tickers_row.addWidget(self.tickers_edit)
        browse_tickers_btn = QPushButton("Browse...")
        browse_tickers_btn.clicked.connect(self.browse_tickers_file)
        tickers_row.addWidget(browse_tickers_btn)
        download_layout.addLayout(tickers_row)
        
        # Start date
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Start Date:"))
        self.start_date_edit = QLineEdit("2008-01-01")
        self.start_date_edit.setPlaceholderText("YYYY-MM-DD")
        date_row.addWidget(self.start_date_edit)
        date_row.addStretch()
        download_layout.addLayout(date_row)
        
        # Options
        options_row = QHBoxLayout()
        self.full_download_check = QCheckBox("Full Download (ignore existing)")
        self.full_download_check.setChecked(True)
        self.resume_check = QCheckBox("Resume from checkpoint")
        options_row.addWidget(self.full_download_check)
        options_row.addWidget(self.resume_check)
        options_row.addStretch()
        download_layout.addLayout(options_row)
        
        # Max retries
        retries_row = QHBoxLayout()
        retries_row.addWidget(QLabel("Max Retries:"))
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(1, 10)
        self.max_retries_spin.setValue(3)
        retries_row.addWidget(self.max_retries_spin)
        retries_row.addStretch()
        download_layout.addLayout(retries_row)
        
        # Download button
        self.download_btn = QPushButton("Download Data")
        self.download_btn.setStyleSheet("""
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
        self.download_btn.clicked.connect(self.download_data)
        download_layout.addWidget(self.download_btn)
        
        download_group.setLayout(download_layout)
        layout.addWidget(download_group)
        
        # Clean group
        clean_group = QGroupBox("Clean Data")
        clean_layout = QVBoxLayout()
        
        # Workers
        workers_row = QHBoxLayout()
        workers_row.addWidget(QLabel("Parallel Workers:"))
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(4)
        workers_row.addWidget(self.workers_spin)
        workers_row.addStretch()
        clean_layout.addLayout(workers_row)
        
        # Full clean option
        clean_full_row = QHBoxLayout()
        self.clean_full_check = QCheckBox("Full Clean (reclean all files, ignore existing)")
        self.clean_full_check.setChecked(True)
        clean_full_row.addWidget(self.clean_full_check)
        clean_full_row.addStretch()
        clean_layout.addLayout(clean_full_row)
        
        # Resume option
        clean_resume_row = QHBoxLayout()
        self.clean_resume_check = QCheckBox("Resume (skip already cleaned files)")
        clean_resume_row.addWidget(self.clean_resume_check)
        clean_resume_row.addStretch()
        clean_layout.addLayout(clean_resume_row)
        
        # Clean button
        self.clean_btn = QPushButton("Clean Data")
        self.clean_btn.setStyleSheet("""
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
        self.clean_btn.clicked.connect(self.clean_data)
        clean_layout.addWidget(self.clean_btn)
        
        clean_group.setLayout(clean_layout)
        layout.addWidget(clean_group)
        
        # Clear downloads group
        clear_group = QGroupBox("Clear Downloads")
        clear_layout = QVBoxLayout()
        
        clear_info = QLabel("Delete all downloaded CSV files from the raw data folder.")
        clear_info.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        clear_layout.addWidget(clear_info)
        
        self.clear_btn = QPushButton("Clear All Downloads")
        self.clear_btn.setStyleSheet("""
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
        self.clear_btn.clicked.connect(self.clear_downloads)
        clear_layout.addWidget(self.clear_btn)
        
        clear_group.setLayout(clear_layout)
        layout.addWidget(clear_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
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
    
    def browse_tickers_file(self):
        """Browse for tickers file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Tickers File",
            str(Path(self.service.get_tickers_file()).parent),
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.tickers_edit.setText(file_path)
    
    def download_data(self):
        """Start data download."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "A data operation is already in progress.")
            return
        
        # Disable buttons
        self.download_btn.setEnabled(False)
        self.clean_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing download...")
        self.status_label.setStyleSheet("color: #ff9800;")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting download...")
        
        # Get total tickers for progress bar
        total_tickers = self.service.get_total_tickers(self.tickers_edit.text())
        if total_tickers > 0:
            self.progress_bar.setRange(0, total_tickers)
            self.progress_bar.setValue(0)
            # Store initial count for progress calculation
            raw_dir = self.service.get_raw_dir()
            initial_count = 0
            from pathlib import Path
            raw_path = Path(raw_dir)
            if raw_path.exists():
                initial_count = len(list(raw_path.glob("*.csv")))
            self.initial_download_count = initial_count
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate if we can't get total
            self.initial_download_count = 0
        
        # Create worker
        self.worker = DataWorker(
            "download",
            self.service,
            tickers_file=self.tickers_edit.text(),
            start_date=self.start_date_edit.text(),
            full=self.full_download_check.isChecked(),
            resume=self.resume_check.isChecked(),
            max_retries=self.max_retries_spin.value()
        )
        self.worker.finished.connect(self.on_operation_finished)
        self.worker.progress.connect(self.on_download_progress)
        self.worker.progress_message.connect(self.on_progress_message)
        self.worker.start()
    
    def clean_data(self):
        """Start data cleaning."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "A data operation is already in progress.")
            return
        
        # Disable buttons
        self.download_btn.setEnabled(False)
        self.clean_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing cleaning...")
        self.status_label.setStyleSheet("color: #ff9800;")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting data cleaning...")
        
        # Get total files for progress bar
        total_files = self.service.get_total_raw_files(self.service.get_raw_dir())
        if total_files > 0:
            self.progress_bar.setRange(0, total_files)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 0)  # Indeterminate if we can't get total
        
        # Create worker
        self.worker = DataWorker(
            "clean",
            self.service,
            raw_dir=self.service.get_raw_dir(),
            clean_dir=self.service.get_clean_dir(),
            full=self.clean_full_check.isChecked(),
            resume=self.clean_resume_check.isChecked() if not self.clean_full_check.isChecked() else False,
            workers=self.workers_spin.value()
        )
        self.worker.finished.connect(self.on_operation_finished)
        self.worker.progress.connect(self.on_clean_progress)
        self.worker.progress_message.connect(self.on_progress_message)
        self.worker.start()
    
    def on_download_progress(self, completed: int, total: int):
        """Handle download progress update."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)
    
    def on_clean_progress(self, completed: int, total: int):
        """Handle cleaning progress update."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)
    
    def on_progress_message(self, message: str):
        """Handle progress message update."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #ff9800;")
    
    def clear_downloads(self):
        """Clear all files from the raw data folder."""
        from pathlib import Path
        import shutil
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Clear All Downloads",
            "Are you sure you want to delete ALL files in the raw data folder?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                raw_dir = Path(self.service.get_raw_dir())
                
                if not raw_dir.exists():
                    QMessageBox.information(self, "No Files", "Raw data folder does not exist or is already empty.")
                    return
                
                # Count files before deletion
                csv_files = list(raw_dir.glob("*.csv"))
                file_count = len(csv_files)
                
                if file_count == 0:
                    QMessageBox.information(self, "No Files", "Raw data folder is already empty.")
                    return
                
                # Delete all CSV files
                deleted_count = 0
                for csv_file in csv_files:
                    try:
                        csv_file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error deleting {csv_file.name}: {str(e)}")
                
                # Also delete checkpoint file if it exists
                checkpoint_file = raw_dir / '.download_checkpoint.json'
                if checkpoint_file.exists():
                    try:
                        checkpoint_file.unlink()
                    except Exception:
                        pass
                
                # Log result
                timestamp = datetime.now().strftime('%H:%M:%S')
                if deleted_count == file_count:
                    self.log_text.append(f"[{timestamp}] ✓ Cleared {deleted_count} file(s) from raw data folder")
                    QMessageBox.information(self, "Success", f"Successfully deleted {deleted_count} file(s) from raw data folder.")
                else:
                    self.log_text.append(f"[{timestamp}] ⚠ Deleted {deleted_count} of {file_count} file(s) (some errors occurred)")
                    QMessageBox.warning(self, "Partial Success", f"Deleted {deleted_count} of {file_count} file(s). Some files could not be deleted.")
                
            except Exception as e:
                timestamp = datetime.now().strftime('%H:%M:%S')
                error_msg = f"Error clearing downloads: {str(e)}"
                self.log_text.append(f"[{timestamp}] ✗ {error_msg}")
                QMessageBox.critical(self, "Error", error_msg)
    
    def on_operation_finished(self, success: bool, message: str):
        """Handle completion of data operation."""
        # Re-enable buttons
        self.download_btn.setEnabled(True)
        self.clean_btn.setEnabled(True)
        
        # Set progress to 100% if successful
        if success and self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(self.progress_bar.maximum())
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if success:
            self.status_label.setText(message)
            # Check if message indicates failures
            if "failure" in message.lower() or "failed" in message.lower():
                self.status_label.setStyleSheet("color: #ff9800;")  # Warning color for partial success
                # Log with warning indicator
                self.log_text.append(f"[{timestamp}] ⚠ {message}")
            else:
                self.status_label.setStyleSheet("color: #4caf50;")  # Success color
                self.log_text.append(f"[{timestamp}] ✓ {message}")
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            self.log_text.append(f"[{timestamp}] ✗ Error: {message}")
            QMessageBox.critical(self, "Operation Failed", message)

