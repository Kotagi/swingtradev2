"""
Filter Editor Tab

Interactive filter editor for testing and optimizing entry filters.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
    QProgressBar, QComboBox, QScrollArea, QInputDialog, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

from gui.services import FilterEditorService, DataService, StopLossAnalysisService, SHAPService
from gui.tabs.backtest_tab import FeatureInfoDialog
from gui.utils.feature_descriptions import FEATURE_DESCRIPTIONS, get_feature_display_name
from gui.utils.model_registry import ModelRegistry

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


class ImpactCalculationWorker(QThread):
    """Worker thread for calculating filter impact."""
    
    finished = pyqtSignal(bool, dict, str)  # success, impact_data, message
    progress = pyqtSignal(int, int, str)  # completed, total, message
    
    def __init__(self, service: FilterEditorService, trades_df: pd.DataFrame,
                 features_df: pd.DataFrame, filters: List[Tuple[str, str, float]]):
        super().__init__()
        self.service = service
        self.trades_df = trades_df
        self.features_df = features_df
        self.filters = filters
    
    def run(self):
        """Calculate impact in background thread."""
        try:
            def progress_callback(completed, total, message=None):
                self.progress.emit(completed, total, message or "")
            
            impact_data = self.service.calculate_filter_impact(
                self.trades_df,
                self.features_df,
                self.filters,
                progress_callback=progress_callback
            )
            
            self.finished.emit(True, impact_data, "Impact calculation complete")
        except Exception as e:
            self.finished.emit(False, {}, f"Error: {str(e)}")


class FilterEditorTab(QWidget):
    """Tab for interactive filter editing and testing."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = FilterEditorService()
        self.data_service = DataService()
        self.sl_analysis_service = StopLossAnalysisService()
        self.shap_service = SHAPService()
        self.model_registry = ModelRegistry()
        self.current_csv_path = None
        self.current_trades_df = None
        self.current_features_df = None
        self.current_model_id = None
        self.shap_importance = {}  # Dict of {feature: importance_score}
        self.active_filters = {}  # Dict of {feature: {"enabled": bool, "operator": str, "value": float, "increment": float}}
        self.impact_data = None
        self.feature_worker = None
        self.impact_worker = None
        self.impact_timer = QTimer()  # For debouncing impact calculation
        self.impact_timer.setSingleShot(True)
        self.impact_timer.timeout.connect(self.calculate_impact)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Title
        title = QLabel("Filter Editor")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # Load Backtest section
        load_group = QGroupBox("Load Backtest")
        load_layout = QHBoxLayout()
        
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setReadOnly(True)
        self.csv_path_edit.setPlaceholderText("No backtest CSV loaded")
        load_layout.addWidget(self.csv_path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_csv)
        load_layout.addWidget(browse_btn)
        
        load_group.setLayout(load_layout)
        main_layout.addWidget(load_group)
        
        # Filters section
        filters_group = QGroupBox("Filters")
        filters_layout = QVBoxLayout()
        
        # Search and controls
        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search features...")
        self.search_edit.textChanged.connect(self.filter_feature_list)
        controls_row.addWidget(self.search_edit)
        
        controls_row.addWidget(QLabel("Group by:"))
        self.group_combo = QComboBox()
        self.group_combo.addItems(["Category", "None", "SHAP Importance"])
        self.group_combo.currentTextChanged.connect(self.refresh_feature_list)
        controls_row.addWidget(self.group_combo)
        
        self.show_enabled_only_check = QCheckBox("Show only enabled")
        self.show_enabled_only_check.toggled.connect(self.refresh_feature_list)
        controls_row.addWidget(self.show_enabled_only_check)
        
        # SHAP status label
        self.shap_status_label = QLabel("")
        self.shap_status_label.setStyleSheet("color: #808080; font-style: italic;")
        controls_row.addWidget(self.shap_status_label)
        
        reset_all_btn = QPushButton("Reset All")
        reset_all_btn.clicked.connect(self.reset_all_filters)
        controls_row.addWidget(reset_all_btn)
        
        controls_row.addStretch()
        filters_layout.addLayout(controls_row)
        
        # Feature list (scrollable)
        self.feature_scroll = QScrollArea()
        self.feature_scroll.setWidgetResizable(True)
        self.feature_scroll.setMinimumHeight(400)
        self.feature_widget = QWidget()
        self.feature_layout = QVBoxLayout()
        self.feature_layout.setContentsMargins(0, 0, 0, 0)
        self.feature_widget.setLayout(self.feature_layout)
        self.feature_scroll.setWidget(self.feature_widget)
        filters_layout.addWidget(self.feature_scroll)
        
        filters_group.setLayout(filters_layout)
        main_layout.addWidget(filters_group)
        
        # Impact Preview section
        impact_group = QGroupBox("Impact Preview")
        impact_layout = QVBoxLayout()
        
        self.impact_text = QTextEdit()
        self.impact_text.setReadOnly(True)
        self.impact_text.setMaximumHeight(300)
        self.impact_text.setHtml("<p style='color: #808080;'>Load a backtest and enable filters to see impact preview.</p>")
        impact_layout.addWidget(self.impact_text)
        
        impact_group.setLayout(impact_layout)
        main_layout.addWidget(impact_group)
        
        # Actions section
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout()
        
        save_preset_btn = QPushButton("Save Preset")
        save_preset_btn.clicked.connect(self.save_preset)
        actions_layout.addWidget(save_preset_btn)
        
        load_preset_btn = QPushButton("Load Preset")
        load_preset_btn.clicked.connect(self.load_preset)
        actions_layout.addWidget(load_preset_btn)
        
        compare_presets_btn = QPushButton("Compare Presets")
        compare_presets_btn.clicked.connect(self.compare_presets)
        actions_layout.addWidget(compare_presets_btn)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_filters)
        actions_layout.addWidget(clear_all_btn)
        
        actions_layout.addStretch()
        actions_group.setLayout(actions_layout)
        main_layout.addWidget(actions_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p% (%v/%m)")
        main_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        main_layout.addWidget(self.status_label)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
    
    def browse_csv(self):
        """Browse for backtest CSV file."""
        default_dir = str(PROJECT_ROOT / "data" / "backtest_results")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Backtest CSV",
            default_dir,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.load_backtest(file_path)
    
    def load_backtest(self, file_path: str):
        """Load backtest CSV and extract features."""
        try:
            self.current_csv_path = file_path
            self.csv_path_edit.setText(file_path)
            
            # Load trades
            self.current_trades_df = pd.read_csv(file_path)
            
            # Normalize column names
            self.current_trades_df.columns = self.current_trades_df.columns.str.lower().str.strip()
            
            # Ensure required columns exist
            if "return" not in self.current_trades_df.columns:
                QMessageBox.warning(self, "Invalid File", "CSV must contain 'return' column.")
                return
            
            if "pnl" not in self.current_trades_df.columns:
                # Calculate P&L if missing
                if "entry_price" in self.current_trades_df.columns and "exit_price" in self.current_trades_df.columns:
                    position_size = 1000.0  # Default
                    self.current_trades_df["pnl"] = (
                        (self.current_trades_df["exit_price"] / self.current_trades_df["entry_price"] - 1) * position_size
                    )
                else:
                    self.current_trades_df["pnl"] = self.current_trades_df.get("return", 0) * 1000
            
            # Try to load model metadata and SHAP data
            self.load_model_metadata(file_path)
            
            self.status_label.setText("Extracting features...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            
            # Extract features in background
            data_dir = Path(DataService.get_data_dir())
            self.feature_worker = FeatureExtractionWorker(
                self.sl_analysis_service,
                self.current_trades_df,
                data_dir
            )
            self.feature_worker.progress.connect(self.on_feature_progress)
            self.feature_worker.finished.connect(self.on_features_extracted)
            self.feature_worker.start()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load backtest: {str(e)}")
            self.status_label.setText("Error loading backtest")
    
    def load_model_metadata(self, csv_path: str):
        """Load model metadata from backtest and attempt to load SHAP importance data."""
        try:
            csv_file = Path(csv_path)
            metadata_path = csv_file.parent / f"{csv_file.stem}_metadata.json"
            
            if not metadata_path.exists():
                self.shap_status_label.setText("No model metadata found")
                self.current_model_id = None
                self.shap_importance = {}
                return
            
            # Load metadata
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract model file/name
            model_file = metadata.get("model_file") or metadata.get("model_name")
            if not model_file:
                self.shap_status_label.setText("No model info in metadata")
                self.current_model_id = None
                self.shap_importance = {}
                return
            
            # Get model ID from model path
            model_path = Path(model_file)
            if not model_path.is_absolute():
                # Try relative to models directory
                model_path = PROJECT_ROOT / "models" / model_path.name
            
            model_id = model_path.stem  # Filename without extension
            
            # Look up model in registry to get SHAP artifacts path
            # Try to find model by path first, then by ID
            model_entry = self.model_registry.get_model_by_path(str(model_path))
            if not model_entry:
                model_entry = self.model_registry.get_model(model_id)
            
            # Try to load SHAP artifacts
            if self.shap_service.is_available():
                shap_artifacts = self.shap_service.load_artifacts(model_id)
                
                if shap_artifacts and "importance_ranking" in shap_artifacts:
                    # Build importance dictionary
                    self.shap_importance = {
                        item["feature"]: {
                            "importance": item.get("importance", 0),
                            "importance_pct": item.get("importance_pct", 0),
                            "rank": item.get("rank", 999)
                        }
                        for item in shap_artifacts["importance_ranking"]
                    }
                    self.current_model_id = model_id
                    self.shap_status_label.setText(f"SHAP: {model_id[:30]}...")
                    self.shap_status_label.setStyleSheet("color: #00d4aa; font-style: italic;")
                else:
                    # Check if SHAP artifacts path exists in registry
                    shap_path = None
                    if model_entry:
                        shap_path = model_entry.get("shap_artifacts_path") or model_entry.get("training_info", {}).get("shap_artifacts_path")
                    
                    if shap_path and Path(shap_path).exists():
                        # Try alternative model ID extraction
                        alt_model_id = Path(shap_path).name
                        shap_artifacts = self.shap_service.load_artifacts(alt_model_id)
                        if shap_artifacts and "importance_ranking" in shap_artifacts:
                            self.shap_importance = {
                                item["feature"]: {
                                    "importance": item.get("importance", 0),
                                    "importance_pct": item.get("importance_pct", 0),
                                    "rank": item.get("rank", 999)
                                }
                                for item in shap_artifacts["importance_ranking"]
                            }
                            self.current_model_id = alt_model_id
                            self.shap_status_label.setText(f"SHAP: {alt_model_id[:30]}...")
                            self.shap_status_label.setStyleSheet("color: #00d4aa; font-style: italic;")
                        else:
                            self.shap_status_label.setText("SHAP data not found")
                            self.current_model_id = None
                            self.shap_importance = {}
                    else:
                        self.shap_status_label.setText("SHAP not computed for model")
                        self.current_model_id = None
                        self.shap_importance = {}
            else:
                self.shap_status_label.setText("SHAP service unavailable")
                self.current_model_id = None
                self.shap_importance = {}
                
        except Exception as e:
            self.shap_status_label.setText(f"Error loading SHAP: {str(e)[:30]}")
            self.shap_status_label.setStyleSheet("color: #ff6b6b; font-style: italic;")
            self.current_model_id = None
            self.shap_importance = {}
    
    def on_feature_progress(self, completed: int, total: int):
        """Handle feature extraction progress."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)
    
    def on_features_extracted(self, success: bool, features_df: pd.DataFrame, message: str):
        """Handle feature extraction completion."""
        self.progress_bar.setVisible(False)
        
        if success:
            self.current_features_df = features_df
            self.status_label.setText(f"Loaded {len(self.current_trades_df)} trades, {len(features_df.columns)} features")
            
            # Initialize feature list
            self.initialize_feature_list()
            
            # Calculate initial impact if filters are active
            if self.active_filters:
                self.calculate_impact()
        else:
            QMessageBox.warning(self, "Error", message)
            self.status_label.setText("Feature extraction failed")
    
    def initialize_feature_list(self):
        """Initialize the feature list with all available features."""
        if self.current_features_df is None:
            return
        
        # Get all features
        all_features = list(self.current_features_df.columns)
        
        # Remove non-feature columns
        exclude_cols = ["ticker", "entry_date", "exit_date", "return", "pnl", "holding_days", "exit_reason"]
        features = [f for f in all_features if f.lower() not in exclude_cols]
        
        # Initialize active_filters with defaults
        for feature in features:
            if feature not in self.active_filters:
                defaults = self.service.get_feature_defaults(feature, self.current_features_df)
                self.active_filters[feature] = {
                    "enabled": False,
                    "operator": defaults["operator"],
                    "value": defaults["neutral"],
                    "increment": defaults["increment"]
                }
        
        self.refresh_feature_list()
    
    def refresh_feature_list(self):
        """Refresh the feature list display."""
        # Clear existing widgets
        while self.feature_layout.count():
            child = self.feature_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if self.current_features_df is None:
            return
        
        # Get features to display
        search_text = self.search_edit.text().lower()
        show_enabled_only = self.show_enabled_only_check.isChecked()
        
        features_to_show = []
        for feature in self.active_filters.keys():
            if show_enabled_only and not self.active_filters[feature]["enabled"]:
                continue
            if search_text and search_text not in feature.lower():
                continue
            features_to_show.append(feature)
        
        # Group features if requested
        group_by = self.group_combo.currentText()
        if group_by == "Category":
            # Group by category (simplified - use feature name patterns)
            grouped = {}
            for feature in features_to_show:
                category = self._get_feature_category(feature)
                if category not in grouped:
                    grouped[category] = []
                grouped[category].append(feature)
            
            # Create group boxes
            for category, feature_list in sorted(grouped.items()):
                group_box = QGroupBox(category)
                group_layout = QVBoxLayout()
                
                for feature in sorted(feature_list):
                    feature_widget = self.create_feature_control(feature)
                    group_layout.addWidget(feature_widget)
                
                group_box.setLayout(group_layout)
                self.feature_layout.addWidget(group_box)
        else:
            # No grouping - show all features
            for feature in sorted(features_to_show):
                feature_widget = self.create_feature_control(feature)
                self.feature_layout.addWidget(feature_widget)
        
        self.feature_layout.addStretch()
    
    def _get_feature_category(self, feature_name: str) -> str:
        """Get category for a feature based on name patterns."""
        name_lower = feature_name.lower()
        
        if "price" in name_lower or "ma" in name_lower:
            return "Price & Moving Averages"
        elif "return" in name_lower or "gap" in name_lower:
            return "Returns"
        elif "rsi" in name_lower or "macd" in name_lower or "stochastic" in name_lower or "momentum" in name_lower:
            return "Momentum"
        elif "volume" in name_lower or "obv" in name_lower:
            return "Volume"
        elif "volatility" in name_lower or "atr" in name_lower or "bollinger" in name_lower:
            return "Volatility"
        elif "trend" in name_lower or "adx" in name_lower or "aroon" in name_lower:
            return "Trend"
        elif "candle" in name_lower or "wick" in name_lower:
            return "Candlestick"
        elif "52w" in name_lower or "52" in name_lower:
            return "52-Week"
        elif "beta" in name_lower:
            return "Market Context"
        else:
            return "Other"
    
    def create_feature_control(self, feature: str) -> QWidget:
        """Create a control widget for a single feature."""
        widget = QWidget()
        widget.feature_name = feature  # Store feature name for reference
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Enable checkbox
        enabled_check = QCheckBox()
        enabled_check.setChecked(self.active_filters[feature]["enabled"])
        enabled_check.toggled.connect(lambda checked, f=feature: self.on_feature_enabled(f, checked))
        layout.addWidget(enabled_check)
        
        # Feature name
        display_name = get_feature_display_name(feature)
        name_label = QLabel(display_name)
        name_label.setMinimumWidth(200)
        
        # Add SHAP importance indicator if available
        if feature in self.shap_importance:
            importance_info = self.shap_importance[feature]
            rank = importance_info.get("rank", 999)
            importance_pct = importance_info.get("importance_pct", 0)
            
            # Color code by importance (top 10 = green, top 20 = yellow, rest = gray)
            if rank <= 10:
                color = "#00d4aa"  # Green for top 10
                badge = "★"
            elif rank <= 20:
                color = "#ffd700"  # Gold for top 20
                badge = "●"
            else:
                color = "#808080"  # Gray for others
                badge = "○"
            
            display_name = f"{display_name} {badge} (#{rank}, {importance_pct:.1f}%)"
            name_label.setText(display_name)
            name_label.setStyleSheet(f"color: {color};")
            name_label.setToolTip(f"SHAP Rank: {rank}\nSHAP Importance: {importance_pct:.2f}%")
        
        layout.addWidget(name_label)
        
        # Info button
        info_btn = QPushButton("ℹ")
        info_btn.setFixedSize(25, 25)
        info_btn.setToolTip("Feature information")
        info_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 2px solid #00d4aa;
                border-radius: 12px;
                color: #00d4aa;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-color: #00ffcc;
            }
        """)
        info_btn.clicked.connect(lambda checked=False, f=feature: self.show_feature_info(f))
        layout.addWidget(info_btn)
        
        # Operator selector
        operator_combo = QComboBox()
        operator_combo.addItems([">", ">=", "<", "<="])
        operator_combo.setCurrentText(self.active_filters[feature]["operator"])
        operator_combo.currentTextChanged.connect(lambda op, f=feature: self.on_operator_changed(f, op))
        operator_combo.setEnabled(self.active_filters[feature]["enabled"])
        layout.addWidget(operator_combo)
        
        # Value display and increment input
        value_layout = QHBoxLayout()
        
        # Decrease button
        decrease_btn = QPushButton("-")
        decrease_btn.setFixedWidth(30)
        decrease_btn.clicked.connect(lambda checked=False, f=feature: self.decrease_value(f))
        decrease_btn.setEnabled(self.active_filters[feature]["enabled"])
        value_layout.addWidget(decrease_btn)
        
        # Value display (editable)
        value_spin = QDoubleSpinBox()
        value_spin.setDecimals(4)
        value_spin.setMinimum(-10000.0)
        value_spin.setMaximum(10000.0)
        value_spin.setValue(self.active_filters[feature]["value"])
        value_spin.valueChanged.connect(lambda val, f=feature: self.on_value_changed(f, val))
        value_spin.setEnabled(self.active_filters[feature]["enabled"])
        value_layout.addWidget(value_spin)
        
        # Increase button
        increase_btn = QPushButton("+")
        increase_btn.setFixedWidth(30)
        increase_btn.clicked.connect(lambda checked=False, f=feature: self.increase_value(f))
        increase_btn.setEnabled(self.active_filters[feature]["enabled"])
        value_layout.addWidget(increase_btn)
        
        layout.addLayout(value_layout)
        
        # Increment input
        increment_label = QLabel("Increment:")
        layout.addWidget(increment_label)
        
        increment_spin = QDoubleSpinBox()
        increment_spin.setDecimals(4)
        increment_spin.setMinimum(0.0001)
        increment_spin.setMaximum(1000.0)
        increment_spin.setValue(self.active_filters[feature]["increment"])
        increment_spin.valueChanged.connect(lambda val, f=feature: self.on_increment_changed(f, val))
        increment_spin.setEnabled(self.active_filters[feature]["enabled"])
        increment_spin.setMaximumWidth(100)
        layout.addWidget(increment_spin)
        
        layout.addStretch()
        widget.setLayout(layout)
        
        # Store references for enabling/disabling
        widget.operator_combo = operator_combo
        widget.decrease_btn = decrease_btn
        widget.value_spin = value_spin
        widget.increase_btn = increase_btn
        widget.increment_spin = increment_spin
        
        return widget
    
    def on_feature_enabled(self, feature: str, enabled: bool):
        """Handle feature enable/disable."""
        self.active_filters[feature]["enabled"] = enabled
        
        # Enable/disable controls for this feature
        for i in range(self.feature_layout.count()):
            item = self.feature_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'feature_name') and widget.feature_name == feature:
                    widget.operator_combo.setEnabled(enabled)
                    widget.decrease_btn.setEnabled(enabled)
                    widget.value_spin.setEnabled(enabled)
                    widget.increase_btn.setEnabled(enabled)
                    widget.increment_spin.setEnabled(enabled)
                elif hasattr(widget, 'layout'):  # Group box
                    # Check children in group box
                    group_layout = widget.layout()
                    for j in range(group_layout.count()):
                        child_item = group_layout.itemAt(j)
                        if child_item and child_item.widget():
                            child_widget = child_item.widget()
                            if hasattr(child_widget, 'feature_name') and child_widget.feature_name == feature:
                                child_widget.operator_combo.setEnabled(enabled)
                                child_widget.decrease_btn.setEnabled(enabled)
                                child_widget.value_spin.setEnabled(enabled)
                                child_widget.increase_btn.setEnabled(enabled)
                                child_widget.increment_spin.setEnabled(enabled)
        
        # Trigger impact calculation (debounced)
        self.impact_timer.stop()
        self.impact_timer.start(500)  # 500ms debounce
    
    def on_operator_changed(self, feature: str, operator: str):
        """Handle operator change."""
        self.active_filters[feature]["operator"] = operator
        self.impact_timer.stop()
        self.impact_timer.start(500)
    
    def on_value_changed(self, feature: str, value: float):
        """Handle value change."""
        self.active_filters[feature]["value"] = value
        self.impact_timer.stop()
        self.impact_timer.start(500)
    
    def on_increment_changed(self, feature: str, increment: float):
        """Handle increment change."""
        self.active_filters[feature]["increment"] = increment
    
    def increase_value(self, feature: str):
        """Increase feature value by increment."""
        current = self.active_filters[feature]["value"]
        increment = self.active_filters[feature]["increment"]
        new_value = current + increment
        self.active_filters[feature]["value"] = new_value
        
        # Update value in UI (find and update the spin box)
        self._update_feature_value_ui(feature, new_value)
        
        self.impact_timer.stop()
        self.impact_timer.start(500)
    
    def decrease_value(self, feature: str):
        """Decrease feature value by increment."""
        current = self.active_filters[feature]["value"]
        increment = self.active_filters[feature]["increment"]
        new_value = current - increment
        self.active_filters[feature]["value"] = new_value
        
        # Update value in UI (find and update the spin box)
        self._update_feature_value_ui(feature, new_value)
        
        self.impact_timer.stop()
        self.impact_timer.start(500)
    
    def _update_feature_value_ui(self, feature: str, value: float):
        """Update the value spin box for a specific feature."""
        for i in range(self.feature_layout.count()):
            item = self.feature_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if hasattr(widget, 'feature_name') and widget.feature_name == feature:
                    widget.value_spin.setValue(value)
                    return
                elif hasattr(widget, 'layout'):  # Group box
                    group_layout = widget.layout()
                    for j in range(group_layout.count()):
                        child_item = group_layout.itemAt(j)
                        if child_item and child_item.widget():
                            child_widget = child_item.widget()
                            if hasattr(child_widget, 'feature_name') and child_widget.feature_name == feature:
                                child_widget.value_spin.setValue(value)
                                return
    
    def calculate_impact(self):
        """Calculate impact of current filters."""
        if self.current_trades_df is None or self.current_features_df is None:
            return
        
        # Get enabled filters
        enabled_filters = [
            (feature, self.active_filters[feature]["operator"], self.active_filters[feature]["value"])
            for feature in self.active_filters
            if self.active_filters[feature]["enabled"]
        ]
        
        if not enabled_filters:
            self.impact_text.setHtml("<p style='color: #808080;'>Enable filters to see impact preview.</p>")
            return
        
        # Calculate impact in background
        self.status_label.setText("Calculating impact...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.impact_worker = ImpactCalculationWorker(
            self.service,
            self.current_trades_df,
            self.current_features_df,
            enabled_filters
        )
        self.impact_worker.progress.connect(self.on_impact_progress)
        self.impact_worker.finished.connect(self.on_impact_calculated)
        self.impact_worker.start()
    
    def on_impact_progress(self, completed: int, total: int, message: str):
        """Handle impact calculation progress."""
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)
        if message:
            self.status_label.setText(message)
    
    def on_impact_calculated(self, success: bool, impact_data: dict, message: str):
        """Handle impact calculation completion."""
        self.progress_bar.setVisible(False)
        
        if success:
            self.impact_data = impact_data
            self.update_impact_display()
            self.status_label.setText("Impact calculation complete")
        else:
            QMessageBox.warning(self, "Error", message)
            self.status_label.setText("Impact calculation failed")
    
    def update_impact_display(self):
        """Update the impact preview display."""
        if self.impact_data is None:
            return
        
        combined = self.impact_data.get("combined", {})
        before = self.impact_data.get("before_metrics", {})
        after = self.impact_data.get("after_metrics", {})
        per_feature = self.impact_data.get("per_feature", {})
        
        html = "<h3>Combined Impact</h3>"
        html += f"<p><b>Selected Filters:</b> {len([f for f in self.active_filters.values() if f['enabled']])}</p>"
        
        html += "<h4>Trade Volume:</h4>"
        html += f"<ul>"
        html += f"<li>Total Trades: {before.get('n_trades', 0)} → {after.get('n_trades', 0)} "
        if before.get('n_trades', 0) > 0:
            change = after.get('n_trades', 0) - before.get('n_trades', 0)
            pct = (change / before.get('n_trades', 0)) * 100
            html += f"(<span style='color: {'#ff6b6b' if change < 0 else '#51cf66'};'>{change:+.0f}, {pct:+.1f}%)</span>"
        html += f"</li>"
        html += f"<li>Winners Excluded: {combined.get('winners_excluded', 0)} ({combined.get('winners_excluded_pct', 0):.1f}%)</li>"
        html += f"<li>Stop-Losses Excluded: {combined.get('stop_losses_excluded', 0)} ({combined.get('stop_losses_excluded_pct', 0):.1f}%)</li>"
        html += f"</ul>"
        
        html += "<h4>Performance:</h4>"
        html += "<ul>"
        html += f"<li>Total P&L: ${before.get('total_pnl', 0):,.2f} → ${after.get('total_pnl', 0):,.2f}"
        pnl_change = after.get('total_pnl', 0) - before.get('total_pnl', 0)
        if pnl_change != 0:
            html += f" <span style='color: {'#ff6b6b' if pnl_change < 0 else '#51cf66'};'>({pnl_change:+,.2f})</span>"
        html += "</li>"
        html += f"<li>Avg P&L/Trade: ${before.get('avg_pnl', 0):,.2f} → ${after.get('avg_pnl', 0):,.2f}</li>"
        html += f"<li>Win Rate: {before.get('win_rate', 0)*100:.2f}% → {after.get('win_rate', 0)*100:.2f}%</li>"
        html += f"<li>Sharpe Ratio: {before.get('sharpe_ratio', 0):.2f} → {after.get('sharpe_ratio', 0):.2f}</li>"
        html += f"<li>Max Drawdown: ${before.get('max_drawdown', 0):,.2f} → ${after.get('max_drawdown', 0):,.2f}</li>"
        html += f"<li>Profit Factor: {before.get('profit_factor', 0):.2f} → {after.get('profit_factor', 0):.2f}</li>"
        if 'annual_return' in before:
            html += f"<li>Annual Return: {before.get('annual_return', 0)*100:.2f}% → {after.get('annual_return', 0)*100:.2f}%</li>"
        html += "</ul>"
        
        # Warnings and validation
        warnings = []
        errors = []
        
        # Trade volume warnings
        if combined.get('winners_excluded_pct', 0) > 20:
            warnings.append(f"⚠ Excluding >20% of winners ({combined.get('winners_excluded_pct', 0):.1f}%)")
        if combined.get('total_excluded_pct', 0) > 50:
            warnings.append(f"⚠ Excluding >50% of total trades ({combined.get('total_excluded_pct', 0):.1f}%)")
        if combined.get('remaining_trades', 0) < 10:
            warnings.append(f"⚠ Very few trades remaining ({combined.get('remaining_trades', 0)}). Results may be unreliable.")
        
        # Performance warnings
        pnl_change = after.get('total_pnl', 0) - before.get('total_pnl', 0)
        if pnl_change < -1000:
            warnings.append(f"⚠ Significant P&L reduction: ${pnl_change:,.2f}")
        
        win_rate_change = (after.get('win_rate', 0) - before.get('win_rate', 0)) * 100
        if win_rate_change < -5:
            warnings.append(f"⚠ Win rate decreased by {abs(win_rate_change):.1f}%")
        
        sharpe_change = after.get('sharpe_ratio', 0) - before.get('sharpe_ratio', 0)
        if sharpe_change < -0.5:
            warnings.append(f"⚠ Sharpe ratio decreased by {abs(sharpe_change):.2f}")
        
        # Errors (critical issues)
        if after.get('n_trades', 0) == 0:
            errors.append("❌ All trades filtered out! No trades remain after applying filters.")
        if after.get('win_rate', 0) == 0 and after.get('n_trades', 0) > 0:
            errors.append("❌ Win rate is 0% - all remaining trades are losers.")
        
        # Display errors first (if any)
        if errors:
            html += "<h4 style='color: #f44336;'>Errors:</h4><ul>"
            for error in errors:
                html += f"<li>{error}</li>"
            html += "</ul>"
        
        # Then warnings
        if warnings:
            html += "<h4 style='color: #ff9800;'>Warnings:</h4><ul>"
            for warning in warnings:
                html += f"<li>{warning}</li>"
            html += "</ul>"
        
        # Success indicators (if improvements)
        improvements = []
        if pnl_change > 1000:
            improvements.append(f"✓ P&L improved by ${pnl_change:,.2f}")
        if win_rate_change > 2:
            improvements.append(f"✓ Win rate improved by {win_rate_change:.1f}%")
        if sharpe_change > 0.2:
            improvements.append(f"✓ Sharpe ratio improved by {sharpe_change:.2f}")
        if combined.get('stop_losses_excluded_pct', 0) > combined.get('winners_excluded_pct', 0) + 5:
            improvements.append(f"✓ Excluding more stop-losses ({combined.get('stop_losses_excluded_pct', 0):.1f}%) than winners ({combined.get('winners_excluded_pct', 0):.1f}%)")
        
        if improvements:
            html += "<h4 style='color: #4caf50;'>Improvements:</h4><ul>"
            for improvement in improvements:
                html += f"<li>{improvement}</li>"
            html += "</ul>"
        
        # Per-feature impact (collapsible or in separate section)
        if per_feature:
            html += "<h4>Per-Feature Impact:</h4>"
            html += "<ul>"
            for feature, impact in per_feature.items():
                if impact.get('total_excluded', 0) > 0:
                    display_name = get_feature_display_name(feature)
                    html += f"<li><b>{display_name}</b>: "
                    html += f"Excludes {impact.get('winners_excluded', 0)} winners ({impact.get('winners_excluded_pct', 0):.1f}%), "
                    html += f"{impact.get('stop_losses_excluded', 0)} stop-losses ({impact.get('stop_losses_excluded_pct', 0):.1f}%)</li>"
            html += "</ul>"
        
        self.impact_text.setHtml(html)
    
    def filter_feature_list(self):
        """Filter feature list based on search text."""
        self.refresh_feature_list()
    
    def reset_all_filters(self):
        """Reset all filters to neutral/default values."""
        if self.current_features_df is None:
            return
        
        reply = QMessageBox.question(
            self,
            "Reset All Filters",
            "Reset all filters to default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for feature in self.active_filters:
                defaults = self.service.get_feature_defaults(feature, self.current_features_df)
                self.active_filters[feature] = {
                    "enabled": False,
                    "operator": defaults["operator"],
                    "value": defaults["neutral"],
                    "increment": defaults["increment"]
                }
            
            self.refresh_feature_list()
            self.impact_text.setHtml("<p style='color: #808080;'>All filters reset to defaults.</p>")
    
    def clear_all_filters(self):
        """Clear all enabled filters."""
        for feature in self.active_filters:
            self.active_filters[feature]["enabled"] = False
        
        self.refresh_feature_list()
        self.impact_text.setHtml("<p style='color: #808080;'>All filters cleared.</p>")
    
    def show_feature_info(self, feature: str):
        """Show feature information dialog."""
        dialog = FeatureInfoDialog(feature, self)
        dialog.exec()
    
    def save_preset(self):
        """Save current filter configuration as a preset."""
        enabled_filters = [
            (feature, self.active_filters[feature]["operator"], self.active_filters[feature]["value"])
            for feature in self.active_filters
            if self.active_filters[feature]["enabled"]
        ]
        
        if not enabled_filters:
            QMessageBox.warning(self, "No Filters", "Enable at least one filter before saving.")
            return
        
        name, ok = QInputDialog.getText(
            self,
            "Save Filter Preset",
            "Preset name:"
        )
        
        if ok and name:
            description, ok = QInputDialog.getText(
                self,
                "Save Filter Preset",
                "Description (optional):"
            )
            
            if self.service.save_filter_preset(name, enabled_filters, description or ""):
                QMessageBox.information(self, "Success", f"Filter preset '{name}' saved successfully.")
            else:
                QMessageBox.warning(self, "Error", "Failed to save filter preset.")
    
    def load_preset(self):
        """Load a filter preset."""
        presets = self.service.list_filter_presets()
        
        if not presets:
            QMessageBox.information(self, "No Presets", "No filter presets found.")
            return
        
        # Show preset selection dialog
        preset_names = [p.get("name", p.get("filename", "Unknown")) for p in presets]
        preset_name, ok = QInputDialog.getItem(
            self,
            "Load Filter Preset",
            "Select preset:",
            preset_names,
            0,
            False
        )
        
        if ok and preset_name:
            # Find preset
            preset = next((p for p in presets if p.get("name") == preset_name), None)
            if not preset:
                QMessageBox.warning(self, "Error", "Preset not found.")
                return
            
            # Load filters
            filters = preset.get("filters", [])
            
            # Clear current filters
            for feature in self.active_filters:
                self.active_filters[feature]["enabled"] = False
            
            # Apply preset filters
            for feature, operator, value in filters:
                if feature in self.active_filters:
                    self.active_filters[feature]["enabled"] = True
                    self.active_filters[feature]["operator"] = operator
                    self.active_filters[feature]["value"] = value
            
            self.refresh_feature_list()
            self.calculate_impact()
            QMessageBox.information(self, "Success", f"Filter preset '{preset_name}' loaded.")
    
    def compare_presets(self):
        """Compare multiple filter presets."""
        if self.current_trades_df is None or self.current_features_df is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load a backtest CSV first before comparing presets."
            )
            return
        
        from gui.tabs.preset_comparison_dialog import PresetComparisonDialog
        
        dialog = PresetComparisonDialog(
            self.service,
            self.current_trades_df,
            self.current_features_df,
            self
        )
        dialog.exec()
    
    def get_active_filters(self) -> List[Tuple[str, str, float]]:
        """Get list of active filters as (feature, operator, value) tuples."""
        return [
            (feature, self.active_filters[feature]["operator"], self.active_filters[feature]["value"])
            for feature in self.active_filters
            if self.active_filters[feature]["enabled"]
        ]

