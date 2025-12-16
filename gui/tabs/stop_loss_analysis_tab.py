"""
Stop-Loss Analysis Tab

Analyzes stop-loss trades to identify patterns and generate filter recommendations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
    QProgressBar, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from gui.services import StopLossAnalysisService, DataService

PROJECT_ROOT = Path(__file__).parent.parent.parent


class FeatureExtractionWorker(QThread):
    """Worker thread for extracting features at entry time."""
    
    finished = pyqtSignal(bool, pd.DataFrame, str)  # success, features_df, message
    progress = pyqtSignal(int, int)  # completed, total
    progress_message = pyqtSignal(str)  # message
    
    def __init__(self, service: StopLossAnalysisService, trades_df: pd.DataFrame, data_dir: Path):
        super().__init__()
        self.service = service
        self.trades_df = trades_df
        self.data_dir = data_dir
    
    def run(self):
        """Extract features in background thread."""
        try:
            def progress_callback(completed, total, message=None):
                self.progress.emit(completed, total)
                if message:
                    self.progress_message.emit(message)
            
            features_df = self.service.get_entry_features(
                self.trades_df,
                self.data_dir,
                progress_callback=progress_callback
            )
            
            if features_df.empty:
                self.finished.emit(False, pd.DataFrame(), 
                                 "No features extracted. Check that feature files exist.")
            else:
                self.finished.emit(True, features_df, 
                                 f"Extracted features for {len(features_df)} trades")
        except Exception as e:
            self.finished.emit(False, pd.DataFrame(), f"Error: {str(e)}")


class StopLossAnalysisTab(QWidget):
    """Tab for analyzing stop-loss trades and generating filter recommendations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = StopLossAnalysisService()
        self.data_service = DataService()
        self.current_csv_path = None
        self.current_trades_df = None
        self.current_features_df = None
        self.analysis_results = None
        self.feature_worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
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
        title = QLabel("Stop-Loss Analysis")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Backtest CSV:"))
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setReadOnly(True)
        self.csv_path_edit.setPlaceholderText("No backtest CSV loaded")
        csv_row.addWidget(self.csv_path_edit)
        change_csv_btn = QPushButton("Change...")
        change_csv_btn.clicked.connect(self.browse_csv)
        csv_row.addWidget(change_csv_btn)
        input_layout.addLayout(csv_row)
        
        self.analyze_btn = QPushButton("Analyze Stop-Losses")
        self.analyze_btn.setStyleSheet("""
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
        self.analyze_btn.clicked.connect(self.run_analysis)
        input_layout.addWidget(self.analyze_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Summary cards (placeholder - will be populated in Unit 1.4)
        summary_group = QGroupBox("Summary")
        summary_layout = QHBoxLayout()
        
        self.total_trades_card = self.create_summary_card("Total Trades", "0")
        self.stop_loss_card = self.create_summary_card("Stop-Losses", "0 (0%)")
        self.winners_card = self.create_summary_card("Winners", "0 (0%)")
        self.target_card = self.create_summary_card("Target Reached", "0 (0%)")
        
        summary_layout.addWidget(self.total_trades_card)
        summary_layout.addWidget(self.stop_loss_card)
        summary_layout.addWidget(self.winners_card)
        summary_layout.addWidget(self.target_card)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Analysis settings (collapsible - will be implemented in Phase 2)
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QVBoxLayout()
        
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Effect Size Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(0.3)
        self.threshold_spin.setDecimals(2)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        settings_layout.addLayout(threshold_row)
        
        settings_group.setLayout(settings_layout)
        settings_group.setVisible(False)  # Hide for now, will show in Phase 2
        layout.addWidget(settings_group)
        
        # Feature comparison table (placeholder - will be populated in Unit 1.5)
        table_group = QGroupBox("Top Differentiating Features")
        table_layout = QVBoxLayout()
        
        self.features_table = QTableWidget()
        self.features_table.setColumnCount(5)
        self.features_table.setHorizontalHeaderLabels([
            "Feature", "Stop-Loss Mean", "Winner Mean", "Difference", "Effect Size"
        ])
        self.features_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.features_table.setAlternatingRowColors(True)
        self.features_table.setMinimumHeight(300)
        table_layout.addWidget(self.features_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Recommendations panel (placeholder - will be implemented in Phase 2)
        recommendations_group = QGroupBox("Recommendations")
        recommendations_layout = QVBoxLayout()
        
        self.recommendations_label = QLabel("Run analysis to generate recommendations")
        self.recommendations_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        recommendations_layout.addWidget(self.recommendations_label)
        
        recommendations_group.setLayout(recommendations_layout)
        layout.addWidget(recommendations_group)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setFormat("%p% (%v/%m)")
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("Ready - Load a backtest CSV to begin")
        self.status_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def create_summary_card(self, title: str, value: str) -> QWidget:
        """Create a summary card widget."""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        card_layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
        card_layout.addWidget(value_label)
        
        card.setLayout(card_layout)
        return card
    
    def browse_csv(self):
        """Browse for backtest CSV file."""
        backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backtest CSV",
            str(backtest_dir),
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.current_csv_path = file_path
            self.csv_path_edit.setText(Path(file_path).name)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            self.status_label.setStyleSheet("color: #4caf50;")
    
    def run_analysis(self):
        """Run stop-loss analysis."""
        if not self.current_csv_path:
            QMessageBox.warning(self, "No CSV Loaded", "Please load a backtest CSV file first.")
            return
        
        # Disable analyze button
        self.analyze_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Loading backtest CSV...")
        self.status_label.setStyleSheet("color: #ff9800;")
        
        try:
            # Load trades from CSV
            self.current_trades_df = pd.read_csv(self.current_csv_path)
            
            # Handle entry_date column (convert to datetime if needed)
            if 'entry_date' in self.current_trades_df.columns:
                self.current_trades_df['entry_date'] = pd.to_datetime(
                    self.current_trades_df['entry_date'], errors='coerce'
                )
                # Set entry_date as index for easier lookup
                self.current_trades_df = self.current_trades_df.set_index('entry_date')
            
            self.status_label.setText(f"Loaded {len(self.current_trades_df)} trades. Extracting features...")
            
            # Get data directory for features
            data_dir = PROJECT_ROOT / "data" / "features_labeled"
            
            # Start feature extraction in background thread
            self.feature_worker = FeatureExtractionWorker(
                self.service,
                self.current_trades_df,
                data_dir
            )
            self.feature_worker.finished.connect(self.on_feature_extraction_finished)
            self.feature_worker.progress.connect(self.on_feature_extraction_progress)
            self.feature_worker.progress_message.connect(self.on_feature_extraction_message)
            self.feature_worker.start()
            
        except Exception as e:
            self.status_label.setText(f"Error loading CSV: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336;")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
            self.analyze_btn.setEnabled(True)
    
    def on_feature_extraction_progress(self, completed: int, total: int):
        """Handle feature extraction progress updates."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(completed)
    
    def on_feature_extraction_message(self, message: str):
        """Handle feature extraction progress messages."""
        self.status_label.setText(message)
    
    def on_feature_extraction_finished(self, success: bool, features_df: pd.DataFrame, message: str):
        """Handle completion of feature extraction."""
        self.progress_bar.setVisible(False)
        
        # Re-enable analyze button
        self.analyze_btn.setEnabled(True)
        
        if success:
            self.current_features_df = features_df
            self.status_label.setText(f"✓ {message} - Analysis will be implemented in Unit 1.4")
            self.status_label.setStyleSheet("color: #4caf50;")
        else:
            self.status_label.setText(f"✗ {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            QMessageBox.warning(self, "Feature Extraction Failed", message)

