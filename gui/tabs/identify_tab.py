"""
Trade Identification Tab

This tab allows users to identify current trading opportunities using the trained model.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QGroupBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QMessageBox, QCheckBox, QProgressBar,
    QTextEdit, QHeaderView, QLineEdit, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Tuple

from gui.services import TradeIdentificationService, DataService
from pathlib import Path
from datetime import datetime

# Project root for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


class IdentifyWorker(QThread):
    """Worker thread for identifying opportunities (prevents UI freezing)."""
    
    finished = pyqtSignal(bool, object, str)  # success, dataframe, message
    
    def __init__(self, service: TradeIdentificationService, **kwargs):
        super().__init__()
        self.service = service
        self.kwargs = kwargs
    
    def run(self):
        """Run the identification in background thread."""
        success, df, message = self.service.identify_opportunities(**self.kwargs)
        self.finished.emit(success, df, message)


class IdentifyTab(QWidget):
    """Tab for identifying trading opportunities."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = TradeIdentificationService()
        self.data_service = DataService()
        self.current_opportunities = None
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
        title = QLabel("Trade Identification")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Model selection group
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model File:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(400)
        self.model_combo.lineEdit().setPlaceholderText("Select or enter model path...")
        model_row.addWidget(self.model_combo)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_model)
        model_row.addWidget(browse_btn)
        
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        model_row.addWidget(load_btn)
        
        model_layout.addLayout(model_row)
        
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setStyleSheet("color: #ff9800;")
        model_layout.addWidget(self.model_status_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Parameters group
        params_group = QGroupBox("Identification Parameters")
        params_layout = QVBoxLayout()
        
        # Probability threshold
        prob_row = QHBoxLayout()
        prob_row.addWidget(QLabel("Minimum Probability:"))
        self.min_prob_spin = QSpinBox()
        self.min_prob_spin.setRange(0, 100)
        self.min_prob_spin.setSingleStep(1)
        self.min_prob_spin.setValue(50)
        self.min_prob_spin.setSuffix("%")
        prob_row.addWidget(self.min_prob_spin)
        prob_row.addStretch()
        params_layout.addLayout(prob_row)
        
        # Top N
        topn_row = QHBoxLayout()
        topn_row.addWidget(QLabel("Top N Results:"))
        self.topn_spin = QSpinBox()
        self.topn_spin.setRange(1, 100)
        self.topn_spin.setValue(20)
        topn_row.addWidget(self.topn_spin)
        topn_row.addStretch()
        params_layout.addLayout(topn_row)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Filters group
        filters_group = QGroupBox("Entry Filters")
        filters_layout = QVBoxLayout()
        
        self.use_recommended_check = QCheckBox("Use Recommended Filters")
        self.use_recommended_check.setToolTip(
            "Apply recommended filters from stop-loss analysis to reduce false positives"
        )
        filters_layout.addWidget(self.use_recommended_check)
        
        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)
        
        # Stop-loss group
        stoploss_group = QGroupBox("Stop-Loss Configuration (Optional)")
        stoploss_layout = QVBoxLayout()
        
        stoploss_mode_row = QHBoxLayout()
        stoploss_mode_row.addWidget(QLabel("Stop-Loss Mode:"))
        self.stoploss_combo = QComboBox()
        self.stoploss_combo.addItems(["None", "adaptive_atr", "swing_atr"])
        self.stoploss_combo.setCurrentIndex(0)
        stoploss_mode_row.addWidget(self.stoploss_combo)
        stoploss_mode_row.addStretch()
        stoploss_layout.addLayout(stoploss_mode_row)
        
        # ATR parameters (shown when adaptive_atr or swing_atr is selected)
        atr_row = QHBoxLayout()
        atr_row.addWidget(QLabel("ATR Multiplier (K):"))
        self.atr_k_spin = QDoubleSpinBox()
        self.atr_k_spin.setRange(0.5, 5.0)
        self.atr_k_spin.setSingleStep(0.1)
        self.atr_k_spin.setValue(1.8)
        self.atr_k_spin.setDecimals(2)
        atr_row.addWidget(self.atr_k_spin)
        
        atr_row.addWidget(QLabel("Min Stop %:"))
        self.atr_min_spin = QSpinBox()
        self.atr_min_spin.setRange(1, 20)
        self.atr_min_spin.setSingleStep(1)
        self.atr_min_spin.setValue(4)
        self.atr_min_spin.setSuffix("%")
        atr_row.addWidget(self.atr_min_spin)
        
        atr_row.addWidget(QLabel("Max Stop %:"))
        self.atr_max_spin = QSpinBox()
        self.atr_max_spin.setRange(1, 30)
        self.atr_max_spin.setSingleStep(1)
        self.atr_max_spin.setValue(10)
        self.atr_max_spin.setSuffix("%")
        atr_row.addWidget(self.atr_max_spin)
        atr_row.addStretch()
        stoploss_layout.addLayout(atr_row)
        
        stoploss_group.setLayout(stoploss_layout)
        layout.addWidget(stoploss_group)
        
        # Action buttons
        action_row = QHBoxLayout()
        
        self.identify_btn = QPushButton("Identify Opportunities")
        self.identify_btn.setStyleSheet("""
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
        self.identify_btn.clicked.connect(self.identify_opportunities)
        self.identify_btn.setEnabled(False)  # Disabled until model is loaded
        action_row.addWidget(self.identify_btn)
        
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        action_row.addWidget(self.export_btn)
        
        action_row.addStretch()
        layout.addLayout(action_row)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Results table
        results_label = QLabel("Results")
        results_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(results_label)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Ticker", "Probability", "Current Price", "Stop Loss %", "Date"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.results_table)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
        
        # Load available models
        self.load_available_models()
    
    def load_available_models(self):
        """Load list of available models into combo box."""
        models = self.service.get_available_models()
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if models:
            # Set default model
            default = "models/xgb_classifier_selected_features.pkl"
            if default in models:
                self.model_combo.setCurrentText(default)
    
    def browse_model(self):
        """Open file dialog to browse for model file."""
        models_dir = Path(self.data_service.get_models_dir())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            str(models_dir),
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if file_path:
            self.model_combo.setCurrentText(file_path)
    
    def load_model(self):
        """Load the selected model."""
        model_path = self.model_combo.currentText().strip()
        if not model_path:
            QMessageBox.warning(self, "No Model Selected", "Please select or enter a model path.")
            return
        
        # Convert relative path to absolute if needed
        if not Path(model_path).is_absolute():
            model_path = str(PROJECT_ROOT / model_path)
        
        self.model_status_label.setText("Loading model...")
        self.model_status_label.setStyleSheet("color: #ff9800;")
        
        success, message = self.service.load_model(model_path)
        
        if success:
            self.model_status_label.setText(f"✓ {message}")
            self.model_status_label.setStyleSheet("color: #4caf50;")
            self.identify_btn.setEnabled(True)
        else:
            self.model_status_label.setText(f"✗ {message}")
            self.model_status_label.setStyleSheet("color: #f44336;")
            QMessageBox.critical(self, "Model Load Error", message)
    
    def identify_opportunities(self):
        """Identify trading opportunities."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "Identification is already in progress.")
            return
        
        # Disable button and show progress
        self.identify_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Identifying opportunities...")
        self.status_label.setStyleSheet("color: #ff9800;")
        
        # Get parameters
        min_prob = self.min_prob_spin.value() / 100.0  # Convert % to decimal
        top_n = self.topn_spin.value()
        use_recommended = self.use_recommended_check.isChecked()
        
        # Stop-loss configuration
        stoploss_mode = self.stoploss_combo.currentText()
        if stoploss_mode == "None":
            stoploss_mode = None
        
        # Create worker thread
        self.worker = IdentifyWorker(
            self.service,
            data_dir=self.data_service.get_data_dir(),
            tickers_file=self.data_service.get_tickers_file(),
            min_probability=min_prob,
            top_n=top_n,
            use_recommended_filters=use_recommended,
            custom_filters=None,  # TODO: Add custom filters UI
            stop_loss_mode=stoploss_mode,
            atr_stop_k=self.atr_k_spin.value(),
            atr_stop_min_pct=self.atr_min_spin.value() / 100.0,  # Convert % to decimal
            atr_stop_max_pct=self.atr_max_spin.value() / 100.0,
            swing_lookback_days=10,
            swing_atr_buffer_k=0.75
        )
        self.worker.finished.connect(self.on_identification_finished)
        self.worker.start()
    
    def on_identification_finished(self, success: bool, df: Optional[pd.DataFrame], message: str):
        """Handle completion of opportunity identification."""
        # Re-enable button and hide progress
        self.identify_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.current_opportunities = df
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: #4caf50;")
            
            if df is not None and not df.empty:
                self.populate_results_table(df)
                self.export_btn.setEnabled(True)
                
                # Auto-save opportunities for dashboard display
                self._auto_save_opportunities(df)
            else:
                self.results_table.setRowCount(0)
                self.export_btn.setEnabled(False)
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            QMessageBox.critical(self, "Identification Error", message)
            self.results_table.setRowCount(0)
            self.export_btn.setEnabled(False)
    
    def _auto_save_opportunities(self, df: pd.DataFrame):
        """Auto-save opportunities to a standard location for dashboard display."""
        try:
            # Create opportunities directory if it doesn't exist
            opps_dir = PROJECT_ROOT / "data" / "opportunities"
            opps_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = opps_dir / f"opportunities_{timestamp}.csv"
            df.to_csv(file_path, index=False)
            
            # Also save as "latest" for easy access
            latest_path = opps_dir / "latest_opportunities.csv"
            df.to_csv(latest_path, index=False)
        except Exception as e:
            # Don't show error to user, just log it
            print(f"Warning: Could not auto-save opportunities: {e}")
    
    def populate_results_table(self, df: pd.DataFrame):
        """Populate the results table with opportunities."""
        self.results_table.setRowCount(len(df))
        
        for row_idx, (_, row) in enumerate(df.iterrows()):
            # Ticker
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(str(row.get('ticker', ''))))
            
            # Probability (format as percentage)
            prob = row.get('probability', 0)
            prob_item = QTableWidgetItem(f"{prob:.1%}")
            prob_item.setData(Qt.ItemDataRole.UserRole, prob)  # Store raw value for sorting
            self.results_table.setItem(row_idx, 1, prob_item)
            
            # Current Price
            price = row.get('current_price', None)
            if pd.notna(price):
                price_item = QTableWidgetItem(f"${price:.2f}")
                price_item.setData(Qt.ItemDataRole.UserRole, price)
            else:
                price_item = QTableWidgetItem("N/A")
            self.results_table.setItem(row_idx, 2, price_item)
            
            # Stop Loss %
            stop_loss = row.get('stop_loss_pct', None)
            if pd.notna(stop_loss) and stop_loss is not None:
                stop_item = QTableWidgetItem(f"{stop_loss:.2%}")
                stop_item.setData(Qt.ItemDataRole.UserRole, stop_loss)
            else:
                stop_item = QTableWidgetItem("N/A")
            self.results_table.setItem(row_idx, 3, stop_item)
            
            # Date
            date = row.get('date', '')
            self.results_table.setItem(row_idx, 4, QTableWidgetItem(str(date)))
        
        # Enable sorting
        self.results_table.setSortingEnabled(True)
        # Sort by probability descending by default
        self.results_table.sortItems(1, Qt.SortOrder.DescendingOrder)
    
    def export_results(self):
        """Export results to CSV file."""
        if self.current_opportunities is None or self.current_opportunities.empty:
            QMessageBox.warning(self, "No Data", "No results to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Opportunities",
            f"opportunities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_opportunities.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Successful", f"Results exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{str(e)}")



