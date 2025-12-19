"""
Feature Selection Dialog

Allows users to enable/disable features for model training.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QCheckBox, QGroupBox, QScrollArea, QWidget,
    QMessageBox, QMenu, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
from datetime import datetime

from gui.utils.feature_descriptions import get_feature_description, get_feature_display_name


class FeatureSelectionDialog(QDialog):
    """Dialog for selecting features for training."""
    
    # Signal emitted when features are saved
    features_saved = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Features for Training")
        self.setMinimumSize(700, 600)
        self.setModal(True)
        
        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.train_config_path = self.project_root / "config" / "train_features.yaml"
        self.features_config_path = self.project_root / "config" / "features.yaml"
        self.presets_dir = self.project_root / "config" / "feature_presets"
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.all_features = {}  # All available features from features.yaml
        self.current_flags = {}  # Current enabled/disabled flags from train_features.yaml
        self.feature_checkboxes = {}  # Map feature name -> QCheckBox
        self.feature_groups = {}  # Map feature name -> category
        
        # Load data
        self._load_features()
        
        # Initialize UI
        self.init_ui()
        
        # Center dialog
        self._center_dialog()
    
    def _load_features(self):
        """Load features from config files."""
        # Load all available features from features.yaml
        if self.features_config_path.exists():
            with open(self.features_config_path, 'r', encoding='utf-8') as f:
                features_cfg = yaml.safe_load(f) or {}
                self.all_features = features_cfg.get("features", {})
        
        # Load current enabled/disabled flags from train_features.yaml
        if self.train_config_path.exists():
            with open(self.train_config_path, 'r', encoding='utf-8') as f:
                train_cfg = yaml.safe_load(f) or {}
                self.current_flags = train_cfg.get("features", {})
        else:
            # If train_features.yaml doesn't exist, enable all features by default
            self.current_flags = {name: 1 for name in self.all_features.keys()}
        
        # Categorize features
        self._categorize_features()
    
    def _categorize_features(self):
        """Categorize features into groups based on their names."""
        categories = {
            "Price Features": ["price", "price_log", "price_vs_ma200"],
            "Return Features": ["daily_return", "gap_pct", "weekly_return", "monthly_return", 
                               "quarterly_return", "ytd_return"],
            "52-Week Features": ["dist_52w_high", "dist_52w_low", "pos_52w"],
            "Market Context": ["mkt_spy", "beta_spy"],  # Check this BEFORE Moving Averages
            "Moving Averages": ["sma20", "sma50", "sma200", "sma20_sma50", "sma50_sma200", "sma50_slope", "sma200_slope"],
            "Volatility": ["volatility_5d", "volatility_21d", "volatility_ratio", "atr14", "volatility_of_volatility"],
            "Volume": ["log_volume", "log_avg_volume", "relative_volume", "obv_momentum"],
            "Momentum Indicators": ["rsi14", "roc10", "roc20", "stochastic_k14", "williams_r14", "kama_slope"],
            "Trend Indicators": ["adx14", "aroon_up", "aroon_down", "aroon_oscillator", "dpo", "trend_residual"],
            "Oscillators": ["macd_histogram", "ppo_histogram", "cci20", "bollinger_band_width"],
            "Pattern Recognition": ["candle_body_pct", "candle_upper_wick_pct", "candle_lower_wick_pct",
                                   "higher_high", "higher_low", "swing_low"],
            "Channel Indicators": ["donchian_position", "donchian_breakout", "ttm_squeeze"],
            "Advanced": ["fractal_dimension_index", "hurst_exponent", "price_curvature"]
        }
        
        # Map each feature to a category
        for feature_name in self.all_features.keys():
            category = "Other"
            for cat_name, keywords in categories.items():
                if any(keyword in feature_name.lower() for keyword in keywords):
                    category = cat_name
                    break
            self.feature_groups[feature_name] = category
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("Select Features for Training")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4aa;")
        layout.addWidget(title)
        
        # Search and filter row
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter features by name...")
        self.search_edit.textChanged.connect(self._filter_features)
        search_row.addWidget(self.search_edit)
        layout.addLayout(search_row)
        
        # Action buttons row
        action_row = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        action_row.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        action_row.addWidget(deselect_all_btn)
        
        reset_btn = QPushButton("Reset to Default")
        reset_btn.clicked.connect(self._reset_to_default)
        action_row.addWidget(reset_btn)
        
        action_row.addStretch()
        
        # Preset management
        preset_btn = QPushButton("Presets ▼")
        preset_menu = QMenu(self)
        preset_menu.addAction("Save Current as Preset...", self._save_preset)
        preset_menu.addAction("Load Preset...", self._load_preset)
        preset_btn.setMenu(preset_menu)
        action_row.addWidget(preset_btn)
        
        layout.addLayout(action_row)
        
        # Count label
        self.count_label = QLabel()
        self._update_count_label()
        self.count_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(self.count_label)
        
        # Scrollable feature list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        
        self.features_widget = QWidget()
        self.features_layout = QVBoxLayout()
        self.features_layout.setSpacing(10)
        self.features_widget.setLayout(self.features_layout)
        
        # Populate features by category
        self._populate_features()
        
        scroll_area.setWidget(self.features_widget)
        layout.addWidget(scroll_area)
        
        # Dialog buttons
        button_row = QHBoxLayout()
        button_row.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save & Close")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #000000;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #00c896;
            }
        """)
        save_btn.clicked.connect(self._save_and_close)
        button_row.addWidget(save_btn)
        
        layout.addLayout(button_row)
        
        self.setLayout(layout)
    
    def _populate_features(self):
        """Populate the feature list grouped by category."""
        # Clear existing layout
        while self.features_layout.count():
            child = self.features_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Group features by category
        categories = {}
        for feature_name, category in self.feature_groups.items():
            if category not in categories:
                categories[category] = []
            categories[category].append(feature_name)
        
        # Sort categories
        sorted_categories = sorted(categories.keys())
        
        # Create group boxes for each category
        for category in sorted_categories:
            group_box = QGroupBox(category)
            group_layout = QVBoxLayout()
            group_layout.setSpacing(5)
            
            features = sorted(categories[category])
            for feature_name in features:
                if feature_name not in self.all_features:
                    continue
                
                # Create row for feature
                feature_row = QHBoxLayout()
                
                # Checkbox
                checkbox = QCheckBox()
                # Get enabled state (default to enabled if not in current_flags)
                is_enabled = self.current_flags.get(feature_name, 1) == 1
                checkbox.setChecked(is_enabled)
                checkbox.stateChanged.connect(self._update_count_label)
                self.feature_checkboxes[feature_name] = checkbox
                feature_row.addWidget(checkbox)
                
                # Feature name (readable format)
                display_name = get_feature_display_name(feature_name)
                name_label = QLabel(display_name)
                name_label.setMinimumWidth(200)
                feature_row.addWidget(name_label)
                
                # Info button
                info_btn = self._create_info_button(feature_name)
                feature_row.addWidget(info_btn)
                
                feature_row.addStretch()
                
                group_layout.addLayout(feature_row)
            
            group_box.setLayout(group_layout)
            self.features_layout.addWidget(group_box)
        
        self.features_layout.addStretch()
    
    def _create_info_button(self, feature_name: str) -> QWidget:
        """Create an info button for a feature."""
        info_btn = QWidget()
        info_btn.setFixedSize(30, 30)
        info_btn.setToolTip("Click for feature description")
        info_btn.setStyleSheet("""
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
        info_layout = QHBoxLayout(info_btn)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_label = QLabel("i")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            QLabel {
                color: #00d4aa;
                font-size: 14px;
                font-weight: bold;
                background-color: transparent;
                border: none;
            }
        """)
        info_layout.addWidget(info_label)
        
        # Make it clickable
        def show_info():
            from gui.tabs.backtest_tab import FeatureInfoDialog
            dialog = FeatureInfoDialog(feature_name, self)
            dialog.exec()
        
        info_btn.mousePressEvent = lambda e: show_info()
        return info_btn
    
    def _filter_features(self, text: str):
        """Filter features based on search text."""
        search_lower = text.lower()
        
        # Hide/show group boxes and features based on search
        for i in range(self.features_layout.count()):
            item = self.features_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QGroupBox):
                group_box = item.widget()
                group_layout = group_box.layout()
                
                if not group_layout:
                    continue
                
                visible_items = 0
                for j in range(group_layout.count()):
                    layout_item = group_layout.itemAt(j)
                    if layout_item and layout_item.layout():
                        feature_row = layout_item.layout()
                        # Find the checkbox and label
                        checkbox = None
                        label = None
                        for k in range(feature_row.count()):
                            widget = feature_row.itemAt(k).widget()
                            if isinstance(widget, QCheckBox):
                                checkbox = widget
                            elif isinstance(widget, QLabel):
                                label = widget
                        
                        if checkbox and label:
                            feature_name = None
                            for name, cb in self.feature_checkboxes.items():
                                if cb == checkbox:
                                    feature_name = name
                                    break
                            
                            if feature_name:
                                display_name = get_feature_display_name(feature_name)
                                matches = (search_lower in feature_name.lower() or 
                                          search_lower in display_name.lower())
                                
                                checkbox.setVisible(matches)
                                label.setVisible(matches)
                                if matches:
                                    visible_items += 1
                
                # Show/hide group box based on visible items
                group_box.setVisible(visible_items > 0)
    
    def _select_all(self):
        """Select all visible features."""
        for checkbox in self.feature_checkboxes.values():
            if checkbox.isVisible():
                checkbox.setChecked(True)
        self._update_count_label()
    
    def _deselect_all(self):
        """Deselect all visible features."""
        for checkbox in self.feature_checkboxes.values():
            if checkbox.isVisible():
                checkbox.setChecked(False)
        self._update_count_label()
    
    def _reset_to_default(self):
        """Reset to default (all enabled)."""
        for checkbox in self.feature_checkboxes.values():
            checkbox.setChecked(True)
        self._update_count_label()
    
    def _update_count_label(self):
        """Update the count label."""
        enabled_count = sum(1 for cb in self.feature_checkboxes.values() if cb.isChecked())
        total_count = len(self.feature_checkboxes)
        self.count_label.setText(f"{enabled_count} of {total_count} features selected")
    
    def _save_and_close(self):
        """Save feature selection and close dialog."""
        # Get enabled features
        enabled_features = {}
        for feature_name, checkbox in self.feature_checkboxes.items():
            enabled_features[feature_name] = 1 if checkbox.isChecked() else 0
        
        # Validate at least one feature is enabled
        if sum(enabled_features.values()) == 0:
            QMessageBox.warning(self, "No Features Selected", 
                              "Please select at least one feature for training.")
            return
        
        # Backup existing config
        if self.train_config_path.exists():
            backup_path = self.train_config_path.parent / f"{self.train_config_path.stem}.backup.yaml"
            import shutil
            shutil.copy2(self.train_config_path, backup_path)
        
        # Save to train_features.yaml
        # Use the same structure as features.yaml to preserve order and comments
        try:
            # Read features.yaml to get the structure
            features_yaml_lines = []
            if self.features_config_path.exists():
                with open(self.features_config_path, 'r', encoding='utf-8') as f:
                    features_yaml_lines = f.readlines()
            
            # Write new config following features.yaml structure
            with open(self.train_config_path, 'w', encoding='utf-8') as f:
                f.write("features:\n")
                
                # Track which features we've written
                written_features = set()
                
                # Write features following features.yaml structure
                for line in features_yaml_lines:
                    line_stripped = line.strip()
                    # Preserve comments
                    if line_stripped.startswith("#"):
                        f.write(line)
                    elif ":" in line_stripped and not line_stripped.startswith("#"):
                        # It's a feature line
                        parts = line_stripped.split(":")
                        if len(parts) >= 2:
                            feature_name = parts[0].strip()
                            if feature_name in enabled_features:
                                # Preserve indentation (usually 2 spaces)
                                indent = len(line) - len(line.lstrip())
                                if indent == 0:
                                    indent = 2  # Default to 2 spaces if no indent
                                f.write(" " * indent + f"{feature_name}: {enabled_features[feature_name]}\n")
                                written_features.add(feature_name)
                
                # Add any features that weren't in features.yaml (shouldn't happen, but just in case)
                for feature_name in sorted(self.all_features.keys()):
                    if feature_name not in written_features and feature_name in enabled_features:
                        f.write(f"  {feature_name}: {enabled_features[feature_name]}\n")
        
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                              f"Failed to save feature configuration: {str(e)}")
            return
        
        # Emit signal
        self.features_saved.emit()
        
        # Close dialog
        self.accept()
    
    def _save_preset(self):
        """Save current feature selection as a preset."""
        # Get preset name
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name.strip():
            return
        
        # Get enabled features
        enabled_features = {}
        for feature_name, checkbox in self.feature_checkboxes.items():
            enabled_features[feature_name] = 1 if checkbox.isChecked() else 0
        
        # Save preset
        preset_path = self.presets_dir / f"{name.strip().replace(' ', '_')}.json"
        import json
        preset_data = {
            "name": name.strip(),
            "features": enabled_features,
            "created_date": datetime.now().isoformat()
        }
        
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=2)
            QMessageBox.information(self, "Preset Saved", 
                                  f"Preset '{name.strip()}' saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", 
                              f"Failed to save preset: {str(e)}")
    
    def _load_preset(self):
        """Load a feature preset."""
        # List available presets
        preset_files = list(self.presets_dir.glob("*.json"))
        if not preset_files:
            QMessageBox.information(self, "No Presets", 
                                  "No feature presets found.")
            return
        
        # Show preset selection dialog
        preset_names = [f.stem.replace('_', ' ') for f in preset_files]
        name, ok = QInputDialog.getItem(self, "Load Preset", "Select preset:", preset_names, 0, False)
        if not ok:
            return
        
        # Load preset
        preset_path = self.presets_dir / f"{name.replace(' ', '_')}.json"
        import json
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            features = preset_data.get("features", {})
            
            # Update checkboxes
            for feature_name, checkbox in self.feature_checkboxes.items():
                enabled = features.get(feature_name, 0) == 1
                checkbox.setChecked(enabled)
            
            self._update_count_label()
            QMessageBox.information(self, "Preset Loaded", 
                                  f"Preset '{name}' loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", 
                              f"Failed to load preset: {str(e)}")
    
    def _center_dialog(self):
        """Center the dialog on the screen."""
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        dialog_geometry = self.frameGeometry()
        center_point = screen.center()
        dialog_geometry.moveCenter(center_point)
        self.move(dialog_geometry.topLeft())
    
    def get_enabled_count(self) -> int:
        """Get the number of enabled features."""
        return sum(1 for cb in self.feature_checkboxes.values() if cb.isChecked())

