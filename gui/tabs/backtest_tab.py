"""
Backtesting Tab

Allows users to run backtests with various strategies and parameters.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QTextEdit, QLineEdit, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QDialog, QScrollArea as QDialogScrollArea, QFrame, QMenu
)
from PyQt6.QtGui import QIcon, QPalette, QColor
from PyQt6.QtCore import QSize
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
from datetime import datetime
import pandas as pd

from gui.services import BacktestService, DataService, StopLossAnalysisService
from gui.widgets import PresetManagerWidget, EquityCurveWidget, ReturnsDistributionWidget
from gui.utils.feature_descriptions import get_feature_description, get_feature_display_name

PROJECT_ROOT = Path(__file__).parent.parent.parent


class FeatureInfoDialog(QDialog):
    """Dialog showing feature description and interpretation guidance."""
    
    def __init__(self, feature_name: str, parent=None):
        super().__init__(parent)
        self.feature_name = feature_name
        self.setWindowTitle("Feature Information")
        self.setMinimumSize(500, 400)
        self.setModal(False)  # Non-modal - can keep open
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Get feature description
        desc = get_feature_description(self.feature_name)
        
        if desc:
            # Feature name (title)
            title = QLabel(desc["name"])
            title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00d4aa;")
            layout.addWidget(title)
            
            # Description section
            desc_label = QLabel("Description:")
            desc_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff; margin-top: 10px;")
            layout.addWidget(desc_label)
            
            desc_text = QTextEdit()
            desc_text.setReadOnly(True)
            desc_text.setPlainText(desc["description"])
            desc_text.setStyleSheet("background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; border-radius: 5px; padding: 10px;")
            desc_text.setMinimumHeight(100)
            layout.addWidget(desc_text)
            
            # Interpretation section
            interp_label = QLabel("How to Interpret Normalized Values:")
            interp_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff; margin-top: 10px;")
            layout.addWidget(interp_label)
            
            interp_text = QTextEdit()
            interp_text.setReadOnly(True)
            interp_text.setPlainText(desc["interpretation"])
            interp_text.setStyleSheet("background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; border-radius: 5px; padding: 10px;")
            interp_text.setMinimumHeight(120)
            layout.addWidget(interp_text)
        else:
            # Generic message for unknown features
            title = QLabel(get_feature_display_name(self.feature_name))
            title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00d4aa;")
            layout.addWidget(title)
            
            generic_text = QTextEdit()
            generic_text.setReadOnly(True)
            generic_text.setPlainText(
                "No detailed description available for this feature.\n\n"
                "Normalized values are typically z-scores (mean=0, std=1) or min-max scaled values. "
                "Positive values generally indicate above-average conditions, "
                "while negative values indicate below-average conditions. "
                "Use your knowledge of the feature to set appropriate filter thresholds."
            )
            generic_text.setStyleSheet("background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; border-radius: 5px; padding: 10px;")
            generic_text.setMinimumHeight(200)
            layout.addWidget(generic_text)
        
        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        
        self.setLayout(layout)
        
        # Center the dialog on parent
        if self.parent():
            parent_geometry = self.parent().geometry()
            dialog_geometry = self.geometry()
            x = parent_geometry.x() + (parent_geometry.width() - dialog_geometry.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - dialog_geometry.height()) // 2
            self.move(x, y)


class ApplyFeaturesDialog(QDialog):
    """Dialog to select and configure feature filters."""
    
    def __init__(self, features: list, existing_filters=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Apply Features / Filters")
        self.setMinimumSize(450, 500)
        self.rows = []
        self.existing_filters = existing_filters or []
        self.features = features
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # Top row with info and load preset button
        top_row = QHBoxLayout()
        info = QLabel("Select features to filter trades. Enable/disable and set a threshold value (normalized values).")
        info.setWordWrap(True)
        info.setStyleSheet("color: #b0b0b0;")
        top_row.addWidget(info, 1)
        
        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #00d4aa;
                color: #000000;
            }
        """)
        load_preset_btn.clicked.connect(self.load_preset_in_dialog)
        top_row.addWidget(load_preset_btn)
        layout.addLayout(top_row)
        
        # Preset status label
        self.preset_status_label = QLabel("")
        self.preset_status_label.setStyleSheet("color: #4caf50; font-style: italic;")
        self.preset_status_label.setWordWrap(True)
        layout.addWidget(self.preset_status_label)
        
        # Scroll area for feature list
        scroll = QDialogScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner_layout = QVBoxLayout()
        inner_layout.setSpacing(8)
        
        for feature_name in self.features:
            row_widget = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            
            # Info button - use QWidget container with QLabel (exact same pattern as help button in model comparison)
            info_btn_container = QWidget()
            info_btn_container.setFixedSize(30, 30)
            info_btn_container.setToolTip(f"Show information about {get_feature_display_name(feature_name)}")
            info_btn_container.setStyleSheet("""
                QWidget {
                    background-color: #2d2d2d;
                    border: 2px solid #00d4aa;
                    border-radius: 15px;
                }
                QWidget:hover {
                    background-color: #3d3d3d;
                    border-color: #00ffcc;
                }
            """)
            info_btn_layout = QHBoxLayout(info_btn_container)
            info_btn_layout.setContentsMargins(0, 0, 0, 0)
            info_label = QLabel("i")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info_label.setStyleSheet("""
                QLabel {
                    color: #00d4aa;
                    font-size: 20px;
                    font-weight: bold;
                    background-color: transparent;
                    border: none;
                }
            """)
            info_btn_layout.addWidget(info_label)
            # Make it clickable
            info_btn_container.mousePressEvent = lambda e, fn=feature_name: self.show_feature_info(fn)
            row_layout.addWidget(info_btn_container)
            
            # Feature name label (readable format)
            feature_label = QLabel(get_feature_display_name(feature_name))
            feature_label.setMinimumWidth(150)
            feature_label.setStyleSheet("color: #e0e0e0;")
            row_layout.addWidget(feature_label)
            
            checkbox = QCheckBox()
            op_combo = QComboBox()
            op_combo.addItems([">", ">=", "<", "<="])
            op_combo.setCurrentText(">")
            value_edit = QLineEdit()
            value_edit.setPlaceholderText("Enter value (normalized)")
            
            # Pre-select if existing filters contain this feature
            for f_name, f_op, f_val in self.existing_filters:
                if f_name == feature_name:
                    checkbox.setChecked(True)
                    op_combo.setCurrentText(f_op)
                    value_edit.setText(str(f_val))
                    break
            
            row_layout.addWidget(checkbox)
            row_layout.addWidget(op_combo)
            row_layout.addWidget(value_edit)
            row_layout.addStretch()
            row_widget.setLayout(row_layout)
            inner_layout.addWidget(row_widget)
            
            self.rows.append((feature_name, checkbox, op_combo, value_edit))
        
        inner_layout.addStretch()
        inner.setLayout(inner_layout)
        scroll.setWidget(inner)
        layout.addWidget(scroll)
        
        # Buttons
        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_selection)
        apply_btn = QPushButton("Apply Features")
        apply_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)
        
        self.setLayout(layout)
    
    def clear_selection(self):
        """Uncheck all features."""
        for _, cb, _, val_edit in self.rows:
            cb.setChecked(False)
            val_edit.clear()
    
    def show_feature_info(self, feature_name: str):
        """Show feature information dialog."""
        dialog = FeatureInfoDialog(feature_name, parent=self)
        dialog.show()  # Use show() instead of exec() for non-modal
    
    def load_preset_in_dialog(self):
        """Load a preset in the dialog."""
        # Get parent (BacktestTab) to access sl_analysis_service
        parent = self.parent()
        if not hasattr(parent, 'sl_analysis_service'):
            QMessageBox.warning(self, "Error", "Cannot access preset service.")
            return
        
        try:
            presets = parent.sl_analysis_service.list_presets()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load presets: {str(e)}")
            return
        
        if not presets:
            QMessageBox.information(self, "No Presets", "No filter presets found.")
            return
        
        # Create menu
        menu = QMenu(self)
        
        # Recent stop-loss filters
        recent_sl_presets = [p for p in presets if p.get('source_backtest')][:5]
        if recent_sl_presets:
            recent_menu = menu.addMenu("Recent Stop-Loss Filters")
            for preset in recent_sl_presets:
                sl_rate_before = preset.get('stop_loss_rate_before', 0.0) * 100
                sl_rate_after = preset.get('stop_loss_rate_after', 0.0) * 100
                action_text = f"{preset['name']} (SL: {sl_rate_before:.1f}% → {sl_rate_after:.1f}%)"
                action = recent_menu.addAction(action_text)
                action.setData(preset['filename'])
        
        menu.addSeparator()
        
        # All presets
        all_menu = menu.addMenu("All Saved Presets")
        for preset in presets:
            action = all_menu.addAction(preset['name'])
            action.setData(preset['filename'])
        
        # Show menu
        action = menu.exec(self.sender().mapToGlobal(self.sender().rect().bottomLeft()))
        
        if action and action.data():
            try:
                preset_data = parent.sl_analysis_service.load_preset(action.data())
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load preset: {str(e)}")
                return
            
            filters = preset_data.get('filters', [])
            if not filters:
                QMessageBox.information(self, "Empty Preset", "This preset contains no filters.")
                return
            
            # Update existing_filters
            self.existing_filters = [
                (f['feature'], f['operator'], f['value'])
                for f in filters
            ]
            
            # Refresh dialog to show loaded filters
            self.refresh_dialog()
            
            # Update status label
            preset_name = preset_data.get('name', action.data())
            self.preset_status_label.setText(f"Loaded preset: {preset_name} ({len(filters)} filters)")
    
    def refresh_dialog(self):
        """Refresh dialog to show existing filters."""
        # Update checkboxes and values based on existing_filters
        for feature_name, cb, op_combo, val_edit in self.rows:
            cb.setChecked(False)
            for f_name, f_op, f_val in self.existing_filters:
                if f_name == feature_name:
                    cb.setChecked(True)
                    op_combo.setCurrentText(f_op)
                    val_edit.setText(str(f_val))
                    break
    
    def get_filters(self):
        """Return list of (feature, operator, value)."""
        filters = []
        for feature_name, cb, op_combo, val_edit in self.rows:
            if cb.isChecked():
                text_val = val_edit.text().strip()
                if text_val == "":
                    # Require a value if checked
                    QMessageBox.warning(self, "Missing Value", f"Please enter a value for {feature_name}.")
                    return None
                try:
                    value = float(text_val)
                except ValueError:
                    QMessageBox.warning(self, "Invalid Value", f"Value for {feature_name} must be numeric.")
                    return None
                filters.append((feature_name, op_combo.currentText(), value))
        return filters


class BacktestWorker(QThread):
    """Worker thread for backtesting."""
    
    finished = pyqtSignal(bool, str, str)  # success, output_file, message
    progress = pyqtSignal(int, int)  # completed, total (for progress bar)
    progress_message = pyqtSignal(str)  # progress message text
    
    def __init__(self, service: BacktestService, **kwargs):
        super().__init__()
        self.service = service
        self.kwargs = kwargs
    
    def backtest_progress_callback(self, completed: int, total: int, message: str = None):
        """Callback for backtest progress updates."""
        self.progress.emit(completed, total)
        if message:
            self.progress_message.emit(message)
        elif total > 0:
            pct = (completed / total) * 100
            self.progress_message.emit(f"Processing {completed}/{total} ({pct:.1f}%)")
    
    def run(self):
        """Run backtest in background thread."""
        try:
            self.kwargs['progress_callback'] = self.backtest_progress_callback
            success, output_file, message = self.service.run_backtest(**self.kwargs)
            self.finished.emit(success, output_file or "", message)
        except Exception as e:
            self.finished.emit(False, "", f"Error: {str(e)}")


class BacktestTab(QWidget):
    """Tab for backtesting."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = BacktestService()
        self.data_service = DataService()
        self.sl_analysis_service = StopLossAnalysisService()
        self.worker = None
        self.entry_filters = []  # List of (feature, operator, value)
        
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
        title = QLabel("Backtesting")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Strategy group
        strategy_group = QGroupBox("Strategy Configuration")
        strategy_layout = QVBoxLayout()
        
        # Strategy selection
        strategy_row = QHBoxLayout()
        strategy_row.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["model", "oracle", "rsi"])
        strategy_row.addWidget(self.strategy_combo)
        strategy_row.addStretch()
        strategy_layout.addLayout(strategy_row)
        
        # Model path (for model strategy)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model File:"))
        self.model_edit = QLineEdit("models/xgb_classifier_selected_features.pkl")
        model_row.addWidget(self.model_edit)
        
        # Create Browse button first to get its height
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.clicked.connect(self.browse_model)
        
        # Dropdown button for model selection - match Browse button height
        # Use QPushButton with custom styling to ensure clicks work
        model_dropdown_btn = QPushButton("▼")
        model_dropdown_btn.setFixedWidth(35)
        browse_height = browse_model_btn.sizeHint().height()
        model_dropdown_btn.setFixedHeight(browse_height)
        model_dropdown_btn.setToolTip("Select from available models")
        model_dropdown_btn.clicked.connect(self.show_model_menu)
        # Style the button to match GUI with rounded corners and white arrow
        model_dropdown_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 5px;
                color: white;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
            }
            QPushButton:pressed {
                background-color: #1d1d1d;
            }
        """)
        
        model_row.addWidget(model_dropdown_btn)
        model_row.addWidget(browse_model_btn)
        strategy_layout.addLayout(model_row)
        
        # Model threshold
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Model Threshold:"))
        self.model_threshold_spin = QSpinBox()
        self.model_threshold_spin.setRange(0, 100)
        self.model_threshold_spin.setSingleStep(1)
        self.model_threshold_spin.setValue(50)
        self.model_threshold_spin.setSuffix("%")
        threshold_row.addWidget(self.model_threshold_spin)
        threshold_row.addStretch()
        strategy_layout.addLayout(threshold_row)
        
        strategy_group.setLayout(strategy_layout)
        layout.addWidget(strategy_group)

        # Filters / feature gating
        filters_row = QHBoxLayout()
        self.apply_features_btn = QPushButton("Apply Features")
        self.apply_features_btn.clicked.connect(self.open_features_dialog)
        filters_row.addWidget(self.apply_features_btn)
        
        # Load Preset button with dropdown
        self.load_preset_btn = QPushButton("Load Preset ▼")
        self.load_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #00d4aa;
                color: #000000;
            }
        """)
        self.load_preset_btn.clicked.connect(self.show_preset_menu)
        filters_row.addWidget(self.load_preset_btn)
        
        self.filters_status_label = QLabel("No filters applied")
        self.filters_status_label.setStyleSheet("color: #b0b0b0;")
        filters_row.addWidget(self.filters_status_label)
        filters_row.addStretch()
        layout.addLayout(filters_row)
        
        # Parameters group
        params_group = QGroupBox("Backtest Parameters")
        params_layout = QVBoxLayout()
        
        # Horizon
        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel("Trade Horizon (days):"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 365)
        self.horizon_spin.setValue(30)
        horizon_row.addWidget(self.horizon_spin)
        horizon_row.addStretch()
        params_layout.addLayout(horizon_row)
        
        # Return threshold
        return_row = QHBoxLayout()
        return_row.addWidget(QLabel("Return Threshold:"))
        self.return_threshold_spin = QSpinBox()
        self.return_threshold_spin.setRange(0, 100)
        self.return_threshold_spin.setSingleStep(1)
        self.return_threshold_spin.setValue(15)
        self.return_threshold_spin.setSuffix("%")
        return_row.addWidget(self.return_threshold_spin)
        return_row.addStretch()
        params_layout.addLayout(return_row)
        
        # Position size
        position_row = QHBoxLayout()
        position_row.addWidget(QLabel("Position Size ($):"))
        self.position_spin = QDoubleSpinBox()
        self.position_spin.setRange(100.0, 100000.0)
        self.position_spin.setSingleStep(100.0)
        self.position_spin.setValue(1000.0)
        self.position_spin.setDecimals(0)
        position_row.addWidget(self.position_spin)
        position_row.addStretch()
        params_layout.addLayout(position_row)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Stop-loss group
        stoploss_group = QGroupBox("Stop-Loss Configuration")
        stoploss_layout = QVBoxLayout()
        
        # Stop-loss mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Stop-Loss Mode:"))
        self.stoploss_combo = QComboBox()
        self.stoploss_combo.addItems(["None", "constant", "adaptive_atr", "swing_atr"])
        self.stoploss_combo.setCurrentIndex(0)
        mode_row.addWidget(self.stoploss_combo)
        mode_row.addStretch()
        stoploss_layout.addLayout(mode_row)
        
        # Constant stop-loss
        constant_row = QHBoxLayout()
        constant_row.addWidget(QLabel("Stop-Loss % (constant mode):"))
        self.stop_loss_spin = QSpinBox()
        self.stop_loss_spin.setRange(-50, 0)
        self.stop_loss_spin.setSingleStep(1)
        self.stop_loss_spin.setValue(-7)
        self.stop_loss_spin.setSuffix("%")
        constant_row.addWidget(self.stop_loss_spin)
        constant_row.addStretch()
        stoploss_layout.addLayout(constant_row)
        
        # ATR parameters
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
        
        # Output group
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Save Results To:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Filename (.csv added automatically) or full path")
        output_row.addWidget(self.output_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output)
        output_row.addWidget(browse_output_btn)
        output_layout.addLayout(output_row)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Preset manager
        preset_row = QHBoxLayout()
        self.preset_manager = PresetManagerWidget("backtest", self)
        self.preset_manager.set_callbacks(self.load_config, self.save_config)
        preset_row.addWidget(self.preset_manager)
        preset_row.addStretch()
        layout.addLayout(preset_row)
        
        # Run button
        self.run_btn = QPushButton("Run Backtest")
        self.run_btn.setStyleSheet("""
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
        self.run_btn.clicked.connect(self.run_backtest)
        layout.addWidget(self.run_btn)
        
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
        
        # Visualizations group
        viz_group = QGroupBox("Visualizations")
        viz_layout = QVBoxLayout()
        
        # Equity curve chart
        equity_label = QLabel("Equity Curve")
        equity_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        viz_layout.addWidget(equity_label)
        self.equity_chart = EquityCurveWidget(self, width=10, height=4)
        self.equity_chart.setMinimumHeight(200)
        viz_layout.addWidget(self.equity_chart)
        
        # Returns distribution chart
        returns_label = QLabel("Returns Distribution")
        returns_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        viz_layout.addWidget(returns_label)
        self.returns_chart = ReturnsDistributionWidget(self, width=10, height=4)
        self.returns_chart.setMinimumHeight(200)
        viz_layout.addWidget(self.returns_chart)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
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
        
        # Track if output field was manually edited (to avoid overwriting user input)
        self.output_manually_edited = False
        self._updating_auto_name = False  # Flag to prevent textChanged from marking as manual during auto-update
        def on_output_text_changed(text):
            # Don't mark as manual if we're programmatically updating
            if self._updating_auto_name:
                return
            # Reset flag if field is cleared, otherwise mark as manually edited
            if not text.strip():
                self.output_manually_edited = False
            else:
                self.output_manually_edited = True
        self.output_edit.textChanged.connect(on_output_text_changed)
        
        # Connect parameter changes to auto-name generation
        self.strategy_combo.currentTextChanged.connect(self.update_auto_name)
        self.horizon_spin.valueChanged.connect(self.update_auto_name)
        self.return_threshold_spin.valueChanged.connect(self.update_auto_name)
        self.stoploss_combo.currentTextChanged.connect(self.update_auto_name)
        self.stop_loss_spin.valueChanged.connect(self.update_auto_name)
        self.atr_k_spin.valueChanged.connect(self.update_auto_name)
        self.atr_min_spin.valueChanged.connect(self.update_auto_name)
        self.atr_max_spin.valueChanged.connect(self.update_auto_name)
        
        # Generate initial auto-name
        self.update_auto_name()
    
    def generate_auto_name(self) -> str:
        """
        Generate an auto-name for the backtest based on current parameters.
        Format: backtest_{strategy}_{horizon}d_{threshold}%_{stop_loss}_{timestamp}.csv
        """
        strategy = self.strategy_combo.currentText()
        horizon = self.horizon_spin.value()
        threshold = self.return_threshold_spin.value()
        
        # Build stop-loss description
        stoploss_mode = self.stoploss_combo.currentText()
        if stoploss_mode == "None":
            stop_loss_str = "none"
        elif stoploss_mode == "constant":
            stop_pct = abs(self.stop_loss_spin.value())  # Make positive for filename
            stop_loss_str = f"constant_{stop_pct}pct"
        elif stoploss_mode == "adaptive_atr":
            k = self.atr_k_spin.value()
            min_pct = self.atr_min_spin.value()
            max_pct = self.atr_max_spin.value()
            stop_loss_str = f"adaptive_atr_{k}_{min_pct}pct_{max_pct}pct"
        elif stoploss_mode == "swing_atr":
            k = self.atr_k_spin.value()
            min_pct = self.atr_min_spin.value()
            max_pct = self.atr_max_spin.value()
            stop_loss_str = f"swing_atr_{k}_{min_pct}pct_{max_pct}pct"
        else:
            stop_loss_str = "unknown"
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Construct filename
        filename = f"backtest_{strategy}_{horizon}d_{threshold}pct_{stop_loss_str}_{timestamp}.csv"
        
        return filename
    
    def update_auto_name(self):
        """
        Update the output field with auto-generated name, but only if:
        - The field is empty, OR
        - The field contains a previously auto-generated name (starts with "backtest_")
        """
        # Only auto-update if field is empty or contains an auto-generated name
        current_text = self.output_edit.text().strip()
        if not current_text or (current_text.startswith("backtest_") and not self.output_manually_edited):
            auto_name = self.generate_auto_name()
            # Only update if different to avoid unnecessary signals
            if current_text != auto_name:
                self._updating_auto_name = True
                self.output_edit.setText(auto_name)
                self.output_manually_edited = False
                self._updating_auto_name = False
    
    def show_model_menu(self):
        """Show a menu with available models from the models folder."""
        models_dir = Path(self.data_service.get_models_dir())
        
        # Create menu
        menu = QMenu(self)
        
        if models_dir.exists():
            # Get all .pkl files
            model_files = sorted(models_dir.glob("*.pkl"))
            
            if model_files:
                # Add each model as a menu item
                for model_file in model_files:
                    relative_path = str(model_file.relative_to(PROJECT_ROOT))
                    action = menu.addAction(model_file.name)
                    action.setData(relative_path)  # Store full relative path
                menu.triggered.connect(self.on_model_selected)
            else:
                # No models found
                no_models_action = menu.addAction("No models found")
                no_models_action.setEnabled(False)
        else:
            # Models directory doesn't exist
            no_dir_action = menu.addAction("Models directory not found")
            no_dir_action.setEnabled(False)
        
        # Show menu below the button
        button = self.sender()
        if button:
            menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
    
    def on_model_selected(self, action):
        """Handle model selection from menu."""
        model_path = action.data()
        if model_path:
            self.model_edit.setText(model_path)
    
    def browse_model(self):
        """Browse for model file."""
        models_dir = Path(self.data_service.get_models_dir())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            str(models_dir),
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if file_path:
            # Convert to relative path if it's in the project
            try:
                relative_path = str(Path(file_path).relative_to(PROJECT_ROOT))
                self.model_edit.setText(relative_path)
            except ValueError:
                # File is outside project, use absolute path
                self.model_edit.setText(file_path)
    
    def _normalize_output_path(self, output: str) -> str:
        """
        Normalize output path: if just a filename, prepend data/backtest_results/.
        Also automatically appends .csv extension if no extension is provided.
        
        Args:
            output: Output path or filename
            
        Returns:
            Full path with default directory and .csv extension if needed
        """
        if not output:
            return output
        
        output_path = Path(output)
        
        # Automatically add .csv extension if no extension is provided
        if not output_path.suffix:
            output_path = output_path.with_suffix('.csv')
        
        # If it's just a filename (no directory separators), use default directory
        if not output_path.parent or output_path.parent == Path('.'):
            backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
            backtest_dir.mkdir(parents=True, exist_ok=True)
            return str(backtest_dir / output_path.name)
        
        # If it's a relative path, make it relative to project root
        if not output_path.is_absolute():
            return str(PROJECT_ROOT / output_path)
        
        # Already an absolute path, return as-is
        return str(output_path)
    
    def browse_output(self):
        """Browse for output file."""
        # Default to data/backtest_results directory
        backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        default_filename = backtest_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Backtest Results",
            str(default_filename),
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.output_edit.setText(file_path)
            self.output_manually_edited = True  # User manually selected a file

    def open_features_dialog(self):
        """Open dialog to apply feature filters."""
        features = self.get_available_features()
        if not features:
            QMessageBox.warning(self, "No Features Found", "Could not load feature columns from the feature data. Build features first.")
            return

        dialog = ApplyFeaturesDialog(features=features, existing_filters=self.entry_filters, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            filters = dialog.get_filters()
            if filters is None:
                return  # validation failed
            self.entry_filters = filters
            if self.entry_filters:
                self.filters_status_label.setText(f"Filters applied ({len(self.entry_filters)})")
                self.filters_status_label.setStyleSheet("color: #4caf50;")
            else:
                self.filters_status_label.setText("No filters applied")
                self.filters_status_label.setStyleSheet("color: #b0b0b0;")

    def show_preset_menu(self):
        """Show preset menu with recent and all presets."""
        try:
            presets = self.sl_analysis_service.list_presets()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load presets: {str(e)}")
            return
        
        if not presets:
            QMessageBox.information(self, "No Presets", "No filter presets found. Create one from the Stop-Loss Analysis tab.")
            return
        
        # Separate recent stop-loss filters (last 5) from all presets
        recent_sl_presets = [p for p in presets if p.get('source_backtest')][:5]
        all_presets = presets
        
        # Create menu
        menu = QMenu(self)
        
        # Recent stop-loss filters section
        if recent_sl_presets:
            recent_menu = menu.addMenu("Recent Stop-Loss Filters")
            for preset in recent_sl_presets:
                sl_rate_before = preset.get('stop_loss_rate_before', 0.0) * 100
                sl_rate_after = preset.get('stop_loss_rate_after', 0.0) * 100
                improvement = sl_rate_before - sl_rate_after
                
                action_text = f"{preset['name']} (SL: {sl_rate_before:.1f}% → {sl_rate_after:.1f}%)"
                action = recent_menu.addAction(action_text)
                action.setData(preset['filename'])
        
        menu.addSeparator()
        
        # All saved presets section
        all_menu = menu.addMenu("All Saved Presets")
        for preset in all_presets:
            action = all_menu.addAction(preset['name'])
            action.setData(preset['filename'])
        
        menu.addSeparator()
        
        # Manage presets option
        manage_action = menu.addAction("Manage Presets...")
        manage_action.triggered.connect(self.manage_presets)
        
        # Show menu and get selection
        action = menu.exec(self.load_preset_btn.mapToGlobal(self.load_preset_btn.rect().bottomLeft()))
        
        if action and action.data():
            self.load_preset(action.data())
    
    def load_preset(self, preset_filename: str):
        """Load a preset and populate the Apply Features dialog."""
        try:
            preset_data = self.sl_analysis_service.load_preset(preset_filename)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load preset: {str(e)}")
            return
        
        filters = preset_data.get('filters', [])
        if not filters:
            QMessageBox.information(self, "Empty Preset", "This preset contains no filters.")
            return
        
        # Convert preset filters to entry_filters format (feature, operator, value)
        self.entry_filters = [
            (f['feature'], f['operator'], f['value'])
            for f in filters
        ]
        
        # Update status label
        preset_name = preset_data.get('name', preset_filename)
        self.filters_status_label.setText(f"Preset loaded: {preset_name} ({len(self.entry_filters)} filters)")
        self.filters_status_label.setStyleSheet("color: #4caf50;")
        
        QMessageBox.information(
            self,
            "Preset Loaded",
            f"Preset '{preset_name}' loaded successfully!\n\n"
            f"Filters: {len(self.entry_filters)}\n"
            f"Source: {preset_data.get('source_backtest', 'N/A')}\n\n"
            f"Click 'Apply Features' to view or modify the filters."
        )
    
    def manage_presets(self):
        """Open preset management dialog."""
        from gui.tabs.preset_management_dialog import PresetManagementDialog
        
        dialog = PresetManagementDialog(self.sl_analysis_service, parent=self)
        dialog.exec()
    
    def get_available_features(self):
        """Get list of feature columns from the feature data (single feature set)."""
        try:
            # Use default feature directory; only one feature set now
            features_dir = Path(self.data_service.get_data_dir())
            if not features_dir.exists():
                return []
            parquet_files = sorted(features_dir.glob("*.parquet"))
            if not parquet_files:
                return []
            # Read columns from first parquet file (read minimal data)
            df = pd.read_parquet(parquet_files[0]).head(1)
            cols = list(df.columns)
            # Exclude label/metadata columns and raw price columns
            excluded = {
                "label", "ticker", "date", "entry_signal", "entry_date", "exit_date", 
                "pnl", "return", "holding_days",
                "open", "high", "low", "close", "adj close"
            }
            # Case-insensitive exclusion check
            excluded_lower = {col.lower() for col in excluded}
            features = [
                c for c in cols 
                if c.lower() not in excluded_lower 
                and not c.lower().startswith("label_")
            ]
            return features
        except Exception as e:
            print(f"Error loading features: {e}")
            return []
    
    def run_backtest(self):
        """Start backtest."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "Backtest is already in progress.")
            return
        
        # Disable button
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Running backtest...")
        self.status_label.setStyleSheet("color: #ff9800;")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting backtest...")
        
        # Get parameters
        kwargs = {
            "horizon": self.horizon_spin.value(),
            "return_threshold": self.return_threshold_spin.value() / 100.0,  # Convert % to decimal
            "position_size": self.position_spin.value(),
            "strategy": self.strategy_combo.currentText(),
            "model_path": self.model_edit.text(),
            "model_threshold": self.model_threshold_spin.value() / 100.0  # Convert % to decimal
        }
        
        # Stop-loss configuration
        stoploss_mode = self.stoploss_combo.currentText()
        if stoploss_mode != "None":
            kwargs["stop_loss_mode"] = stoploss_mode
            if stoploss_mode == "constant":
                kwargs["stop_loss"] = self.stop_loss_spin.value() / 100.0  # Convert % to decimal
            else:
                kwargs["atr_stop_k"] = self.atr_k_spin.value()
                kwargs["atr_stop_min_pct"] = self.atr_min_spin.value() / 100.0  # Convert % to decimal
                kwargs["atr_stop_max_pct"] = self.atr_max_spin.value() / 100.0  # Convert % to decimal
        
        # Output file
        output = self.output_edit.text().strip()
        if not output:
            # Generate auto-name if field is empty
            auto_name = self.generate_auto_name()
            backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
            backtest_dir.mkdir(parents=True, exist_ok=True)
            output = str(backtest_dir / auto_name)
            # Update UI to show the generated name
            self._updating_auto_name = True
            self.output_edit.setText(output)
            self.output_manually_edited = False
            self._updating_auto_name = False
        
        # Normalize path: if just filename, use data/backtest_results/
        kwargs["output"] = self._normalize_output_path(output)

        # Entry filters
        if self.entry_filters:
            kwargs["entry_filters"] = self.entry_filters
        
        # Create worker
        self.worker = BacktestWorker(self.service, **kwargs)
        self.worker.finished.connect(self.on_backtest_finished)
        self.worker.progress.connect(self.on_backtest_progress)
        self.worker.progress_message.connect(self.on_progress_message)
        self.worker.start()
    
    def on_backtest_progress(self, completed: int, total: int):
        """Handle progress updates."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(completed)
    
    def on_progress_message(self, message: str):
        """Handle progress message updates."""
        self.status_label.setText(message)
    
    def on_backtest_finished(self, success: bool, output_file: str, message: str):
        """Handle completion of backtest."""
        # Re-enable button
        self.run_btn.setEnabled(True)
        
        # Set progress to 100% if successful
        if success and self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(self.progress_bar.maximum())
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if success:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: #4caf50;")
            self.log_text.append(f"[{timestamp}] ✓ {message}")
            if self.entry_filters:
                self.log_text.append(f"[{timestamp}] Filters applied ({len(self.entry_filters)})")
                self.filters_status_label.setText(f"Filters applied ({len(self.entry_filters)})")
                self.filters_status_label.setStyleSheet("color: #4caf50;")
            
            # Load and visualize results - use output_file from service, or fall back to UI field
            if not output_file:
                output_file = self.output_edit.text().strip()
            
            # If still no output file, try to find the most recent backtest CSV
            if not output_file:
                try:
                    backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
                    if backtest_dir.exists():
                        csv_files = sorted(backtest_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                        if csv_files:
                            output_file = str(csv_files[0])
                            self.log_text.append(f"[{timestamp}] Using most recent backtest file: {output_file}")
                except Exception as e:
                    self.log_text.append(f"[{timestamp}] Could not find default backtest file: {e}")
            
            if output_file:
                self.log_text.append(f"[{timestamp}] Loading results from: {output_file}")
                self.load_and_visualize_results(output_file)
            else:
                self.log_text.append(f"[{timestamp}] Warning: No output file specified, cannot visualize results")
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            self.log_text.append(f"[{timestamp}] ✗ Error: {message}")
            QMessageBox.critical(self, "Backtest Failed", message)
        
        self.progress_bar.setVisible(False)
    
    def load_and_visualize_results(self, csv_file: str):
        """Load backtest results CSV and display visualizations."""
        try:
            from pathlib import Path
            import traceback
            
            file_path = Path(csv_file)
            if not file_path.exists():
                self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: CSV file not found: {csv_file}")
                return
            
            trades_df = pd.read_csv(file_path)
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(trades_df)} trades from CSV")
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] CSV columns: {', '.join(trades_df.columns)}")
            
            trades_df = self._normalize_trades_for_charts(trades_df)
            if trades_df.empty:
                self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: No trades after normalization")
                return
            
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Normalized trades: {len(trades_df)} rows")
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Normalized columns: {', '.join(trades_df.columns)}")
            
            # Check for required columns
            if 'pnl' not in trades_df.columns and 'return' not in trades_df.columns:
                self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Error: Missing both 'pnl' and 'return' columns")
                return
            
            # Plot cumulative P&L curve
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Plotting cumulative P&L curve...")
            self.equity_chart.plot_cumulative_pnl(trades_df)
            
            # Plot returns distribution
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Plotting returns distribution...")
            self.returns_chart.plot_returns_distribution(trades_df)
            
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Visualizations completed")
        except Exception as e:
            error_msg = f"Error visualizing results: {str(e)}\n{traceback.format_exc()}"
            self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✗ {error_msg}")
            print(f"ERROR in load_and_visualize_results: {error_msg}")
    
    def save_config(self):
        """Save current configuration as a dictionary."""
        config = {
            "strategy": self.strategy_combo.currentText(),
            "model_path": self.model_edit.text(),
            "model_threshold": self.model_threshold_spin.value(),
            "horizon": self.horizon_spin.value(),
            "return_threshold": self.return_threshold_spin.value(),
            "position_size": self.position_spin.value(),
            "stop_loss_mode": self.stoploss_combo.currentText(),
        }
        
        if self.stoploss_combo.currentText() == "constant":
            config["stop_loss"] = self.stop_loss_spin.value()
        elif self.stoploss_combo.currentText() in ["adaptive_atr", "swing_atr"]:
            config["atr_k"] = self.atr_k_spin.value()
            config["atr_min_pct"] = self.atr_min_spin.value()
            config["atr_max_pct"] = self.atr_max_spin.value()
        
        config["output"] = self.output_edit.text().strip()
        return config
    
    def load_config(self, config: dict):
        """Load configuration from a dictionary."""
        if "strategy" in config:
            index = self.strategy_combo.findText(config["strategy"])
            if index >= 0:
                self.strategy_combo.setCurrentIndex(index)
        if "model_path" in config:
            self.model_edit.setText(config["model_path"])
        if "model_threshold" in config:
            self.model_threshold_spin.setValue(config["model_threshold"])
        if "horizon" in config:
            self.horizon_spin.setValue(config["horizon"])
        if "return_threshold" in config:
            self.return_threshold_spin.setValue(config["return_threshold"])
        if "position_size" in config:
            self.position_spin.setValue(config["position_size"])
        if "stop_loss_mode" in config:
            index = self.stoploss_combo.findText(config["stop_loss_mode"])
            if index >= 0:
                self.stoploss_combo.setCurrentIndex(index)
        if "stop_loss" in config:
            self.stop_loss_spin.setValue(config["stop_loss"])
        if "atr_k" in config:
            self.atr_k_spin.setValue(config["atr_k"])
        if "atr_min_pct" in config:
            self.atr_min_spin.setValue(config["atr_min_pct"])
        if "atr_max_pct" in config:
            self.atr_max_spin.setValue(config["atr_max_pct"])
        if "output" in config:
            self.output_edit.setText(config["output"])

    def _normalize_trades_for_charts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare trades DataFrame for charting: ensure return column and dates."""
        if df is None or df.empty:
            return pd.DataFrame()
        trades = df.copy()
        # Ensure return column
        if 'return' not in trades.columns:
            if 'ret' in trades.columns:
                trades['return'] = trades['ret']
            elif 'pnl' in trades.columns:
                # approximate returns from pnl with a heuristic position size
                pos_size = trades['pnl'].abs().median()
                if pos_size <= 0:
                    pos_size = max(1e-9, trades['pnl'].abs().mean())
                if pos_size <= 0:
                    pos_size = 1.0
                trades['return'] = trades['pnl'] / pos_size
            else:
                return pd.DataFrame()
        # Ensure numeric returns
        trades['return'] = pd.to_numeric(trades['return'], errors='coerce').fillna(0.0)
        # Ensure pnl is numeric if it exists
        if 'pnl' in trades.columns:
            trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce').fillna(0.0)
        # Ensure exit_date for timing; fall back to entry_date or index
        if 'exit_date' in trades.columns:
            trades['exit_date'] = pd.to_datetime(trades['exit_date'], errors='coerce')
        elif 'entry_date' in trades.columns:
            trades['exit_date'] = pd.to_datetime(trades['entry_date'], errors='coerce')
        else:
            # create a synthetic date index if none provided
            trades['exit_date'] = pd.to_datetime(pd.Series(range(len(trades))), errors='coerce')
        return trades

