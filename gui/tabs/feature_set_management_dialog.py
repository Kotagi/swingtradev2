"""
Feature Set Management Dialog

Dialog for creating, viewing, and deleting feature sets.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QLineEdit, QTextEdit,
    QGroupBox, QDialogButtonBox, QComboBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from pathlib import Path
import sys
import yaml
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from feature_set_manager import (
        list_feature_sets,
        feature_set_exists,
        create_feature_set,
        get_feature_set_info,
        validate_feature_set,
        get_all_feature_sets,
        update_feature_set_metadata,
        DEFAULT_FEATURE_SET
    )
    HAS_FEATURE_SET_MANAGER = True
except ImportError:
    HAS_FEATURE_SET_MANAGER = False
    DEFAULT_FEATURE_SET = "v3_New_Dawn"


class FeatureSetManagementDialog(QDialog):
    """Dialog for managing feature sets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Set Management")
        self.setMinimumSize(1000, 900)
        self.resize(1000, 900)
        self.init_ui()
        self.load_feature_sets()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Feature Set Management")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Feature sets list
        list_group = QGroupBox("Available Feature Sets")
        list_layout = QVBoxLayout()
        
        self.feature_set_list = QListWidget()
        self.feature_set_list.itemSelectionChanged.connect(self.on_selection_changed)
        list_layout.addWidget(self.feature_set_list)
        
        # Info display
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(250)
        self.info_text.setPlaceholderText("Select a feature set to view details...")
        list_layout.addWidget(self.info_text)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Edit feature set group
        edit_group = QGroupBox("Edit Selected Feature Set")
        edit_layout = QVBoxLayout()
        
        # Name (read-only display)
        name_display_row = QHBoxLayout()
        name_display_row.addWidget(QLabel("Name:"))
        self.edit_name_label = QLabel("(Select a feature set)")
        self.edit_name_label.setStyleSheet("color: #00d4aa; font-weight: bold;")
        name_display_row.addWidget(self.edit_name_label)
        name_display_row.addStretch()
        edit_layout.addLayout(name_display_row)
        
        # Note about renaming
        note_label = QLabel("Note: To rename a feature set, create a copy with the new name instead.")
        note_label.setStyleSheet("color: #b0b0b0; font-style: italic; font-size: 11px;")
        note_label.setWordWrap(True)
        edit_layout.addWidget(note_label)
        
        # View Features button
        view_features_row = QHBoxLayout()
        self.view_features_btn = QPushButton("View Features")
        self.view_features_btn.setEnabled(False)
        self.view_features_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        self.view_features_btn.clicked.connect(self.view_features)
        view_features_row.addWidget(self.view_features_btn)
        view_features_row.addStretch()
        edit_layout.addLayout(view_features_row)
        
        # Description
        desc_edit_row = QHBoxLayout()
        desc_edit_row.addWidget(QLabel("Description:"))
        self.edit_desc_edit = QLineEdit()
        self.edit_desc_edit.setPlaceholderText("Enter description")
        self.edit_desc_edit.setEnabled(False)
        desc_edit_row.addWidget(self.edit_desc_edit)
        self.update_desc_btn = QPushButton("Update Description")
        self.update_desc_btn.setEnabled(False)
        self.update_desc_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        self.update_desc_btn.clicked.connect(self.update_description)
        desc_edit_row.addWidget(self.update_desc_btn)
        edit_layout.addLayout(desc_edit_row)
        
        edit_group.setLayout(edit_layout)
        layout.addWidget(edit_group)
        
        # Create new feature set group
        create_group = QGroupBox("Create New Feature Set")
        create_layout = QVBoxLayout()
        
        # Name input
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g., v2, experimental")
        name_row.addWidget(self.name_edit)
        create_layout.addLayout(name_row)
        
        # Create from scratch checkbox
        self.from_scratch_check = QCheckBox("Create from scratch (no features enabled)")
        self.from_scratch_check.setChecked(False)
        self.from_scratch_check.stateChanged.connect(self.on_from_scratch_changed)
        create_layout.addWidget(self.from_scratch_check)
        
        # Copy from
        copy_row = QHBoxLayout()
        copy_row.addWidget(QLabel("Copy from:"))
        self.copy_from_combo = QComboBox()
        self.copy_from_combo.setMinimumWidth(150)
        copy_row.addWidget(self.copy_from_combo)
        copy_row.addStretch()
        create_layout.addLayout(copy_row)
        
        # Description
        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("Description:"))
        self.desc_edit = QLineEdit()
        self.desc_edit.setPlaceholderText("Optional description")
        desc_row.addWidget(self.desc_edit)
        create_layout.addLayout(desc_row)
        
        # Create button
        self.create_btn = QPushButton("Create Feature Set")
        self.create_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #000000;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #00c896;
            }
        """)
        self.create_btn.clicked.connect(self.create_feature_set)
        create_layout.addWidget(self.create_btn)
        
        create_group.setLayout(create_layout)
        layout.addWidget(create_group)
        
        # Delete button
        delete_row = QHBoxLayout()
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: #ffffff;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_feature_set)
        delete_row.addWidget(self.delete_btn)
        delete_row.addStretch()
        layout.addLayout(delete_row)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Initialize checkbox state
        self.on_from_scratch_changed(0)
    
    def load_feature_sets(self):
        """Load feature sets into the list."""
        self.feature_set_list.clear()
        
        if not HAS_FEATURE_SET_MANAGER:
            item = QListWidgetItem(f"{DEFAULT_FEATURE_SET} (default)")
            item.setData(Qt.ItemDataRole.UserRole, DEFAULT_FEATURE_SET)
            self.feature_set_list.addItem(item)
            return
        
        try:
            all_sets = get_all_feature_sets()
            for info in all_sets:
                if 'error' in info:
                    continue
                
                fs = info['name']
                is_default = " (default)" if fs == DEFAULT_FEATURE_SET else ""
                validation = info.get('validation', {})
                is_valid = validation.get('is_valid', False)
                status = "[OK]" if is_valid else "[WARN]"
                
                display_text = f"{status} {fs}{is_default}"
                item = QListWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.UserRole, fs)
                self.feature_set_list.addItem(item)
            
            # Also populate copy_from combo
            self.copy_from_combo.clear()
            for info in all_sets:
                if 'error' not in info:
                    fs = info['name']
                    display_name = f"{fs} (default)" if fs == DEFAULT_FEATURE_SET else fs
                    self.copy_from_combo.addItem(display_name, fs)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error Loading Feature Sets",
                f"Could not load feature sets: {e}"
            )
    
    def on_selection_changed(self):
        """Handle feature set selection change."""
        current_item = self.feature_set_list.currentItem()
        if not current_item:
            self.info_text.clear()
            self.delete_btn.setEnabled(False)
            self.edit_name_label.setText("(Select a feature set)")
            self.view_features_btn.setEnabled(False)
            self.edit_desc_edit.setEnabled(False)
            self.edit_desc_edit.clear()
            self.update_desc_btn.setEnabled(False)
            return
        
        feature_set = current_item.data(Qt.ItemDataRole.UserRole)
        if not feature_set:
            return
        
        # Enable/disable delete button (can't delete default)
        can_edit = feature_set != DEFAULT_FEATURE_SET
        self.delete_btn.setEnabled(can_edit)
        
        # Update edit section
        self.edit_name_label.setText(feature_set)
        self.view_features_btn.setEnabled(True)  # Can view features for any set
        self.edit_desc_edit.setEnabled(True)  # Can always edit description
        self.update_desc_btn.setEnabled(True)
        
        # Load current description
        if HAS_FEATURE_SET_MANAGER:
            try:
                info = get_feature_set_info(feature_set)
                if 'metadata' in info and 'description' in info['metadata']:
                    self.edit_desc_edit.setText(info['metadata']['description'])
                else:
                    self.edit_desc_edit.clear()
            except Exception:
                self.edit_desc_edit.clear()
        
        if not HAS_FEATURE_SET_MANAGER:
            return
        
        try:
            info = get_feature_set_info(feature_set)
            validation = validate_feature_set(feature_set)
            
            # Build info text
            info_lines = []
            info_lines.append(f"Feature Set: {feature_set}")
            if feature_set == DEFAULT_FEATURE_SET:
                info_lines.append("(Default)")
            info_lines.append("")
            
            info_lines.append("Paths:")
            info_lines.append(f"  Config: {info['config_path']} {'[OK]' if info['config_exists'] else '[MISSING]'}")
            info_lines.append(f"  Data: {info['data_path']} {'[OK]' if info['data_exists'] else '[MISSING]'}")
            info_lines.append(f"  Train Config: {'[OK]' if info['train_config_exists'] else '[MISSING]'}")
            info_lines.append("")
            
            if 'enabled_features' in info:
                info_lines.append(f"Features: {info['enabled_features']} enabled / {info['total_features']} total")
            
            if 'data_files' in info:
                info_lines.append(f"Data Files: {info['data_files']} tickers")
            
            if 'metadata' in info and 'description' in info['metadata']:
                info_lines.append(f"Description: {info['metadata']['description']}")
            
            if validation.get('warnings'):
                info_lines.append("")
                info_lines.append("Warnings:")
                for warning in validation['warnings']:
                    info_lines.append(f"  [WARN] {warning}")
            
            if validation.get('errors'):
                info_lines.append("")
                info_lines.append("Errors:")
                for error in validation['errors']:
                    info_lines.append(f"  [ERROR] {error}")
            
            self.info_text.setText("\n".join(info_lines))
        except Exception as e:
            self.info_text.setText(f"Error loading info: {e}")
    
    def on_from_scratch_changed(self, state):
        """Handle 'from scratch' checkbox change."""
        # Disable/enable copy_from combo based on checkbox
        self.copy_from_combo.setEnabled(not self.from_scratch_check.isChecked())
    
    def create_feature_set(self):
        """Create a new feature set."""
        if not HAS_FEATURE_SET_MANAGER:
            QMessageBox.warning(
                self,
                "Not Available",
                "Feature set manager is not available."
            )
            return
        
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a feature set name.")
            return
        
        from_scratch = self.from_scratch_check.isChecked()
        copy_from = None
        
        if not from_scratch:
            # Only use copy_from if not creating from scratch
            if self.copy_from_combo.currentData():
                copy_from = self.copy_from_combo.currentData()
        
        description = self.desc_edit.text().strip() or None
        
        try:
            create_feature_set(
                feature_set=name,
                copy_from=copy_from,
                description=description,
                from_scratch=from_scratch
            )
            QMessageBox.information(
                self,
                "Success",
                f"Feature set '{name}' created successfully!"
            )
            
            # Clear inputs
            self.name_edit.clear()
            self.desc_edit.clear()
            self.from_scratch_check.setChecked(False)
            self.on_from_scratch_changed(0)  # Reset combo state
            
            # Reload lists
            self.load_feature_sets()
            
            # Select the new feature set
            for i in range(self.feature_set_list.count()):
                item = self.feature_set_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == name:
                    self.feature_set_list.setCurrentItem(item)
                    break
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to create feature set: {e}"
            )
    
    def view_features(self):
        """Open a dialog to view all features in the selected feature set."""
        current_item = self.feature_set_list.currentItem()
        if not current_item:
            return
        
        feature_set = current_item.data(Qt.ItemDataRole.UserRole)
        if not feature_set:
            return
        
        if not HAS_FEATURE_SET_MANAGER:
            QMessageBox.warning(
                self,
                "Not Available",
                "Feature set manager is not available."
            )
            return
        
        try:
            info = get_feature_set_info(feature_set)
            config_path = info.get('config_path')
            
            if not config_path or not Path(config_path).exists():
                QMessageBox.warning(
                    self,
                    "Config Not Found",
                    f"Config file not found for feature set '{feature_set}'."
                )
                return
            
            # Load config file
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            features = config.get('features', {})
            
            if not features:
                QMessageBox.information(
                    self,
                    "No Features",
                    f"Feature set '{feature_set}' has no features defined."
                )
                return
            
            # Extract descriptions from YAML comments and docstrings
            descriptions = self._extract_feature_descriptions(config_path, feature_set, features)
            
            # Create and show feature list dialog
            dialog = FeatureListDialog(self, feature_set, features, descriptions)
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load features: {e}"
            )
    
    def _extract_feature_descriptions(self, config_path: str, feature_set: str, features: dict) -> dict:
        """Extract feature descriptions from YAML comments and function docstrings."""
        descriptions = {}
        
        # First, try to extract from YAML comments
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
            
            # Parse comments - look for pattern: # N. Description\n  feature_name:
            for feature_name in features.keys():
                # Pattern: # (optional number). Description\n  feature_name:
                pattern = rf'#\s*(?:\d+\.\s*)?([^\n]+)\n\s*{re.escape(feature_name)}:'
                match = re.search(pattern, yaml_content)
                if match:
                    desc = match.group(1).strip()
                    # Remove leading number if present (e.g., "1. Price" -> "Price")
                    desc = re.sub(r'^\d+\.\s*', '', desc)
                    descriptions[feature_name] = desc
        except Exception:
            pass
        
        # Fall back to docstrings for features without YAML comments
        if HAS_FEATURE_SET_MANAGER:
            try:
                import importlib
                # Convert feature set name to valid Python module name (spaces/dashes to underscores)
                module_name = feature_set.replace(" ", "_").replace("-", "_")
                registry_module = importlib.import_module(f"features.sets.{module_name}.registry")
                
                # Get FEATURE_REGISTRY
                if hasattr(registry_module, 'FEATURE_REGISTRY'):
                    for feature_name, feature_func in registry_module.FEATURE_REGISTRY.items():
                        if feature_name not in descriptions and feature_name in features:
                            # Get docstring
                            if hasattr(feature_func, '__doc__') and feature_func.__doc__:
                                doc = feature_func.__doc__.strip()
                                # Extract first line or first sentence
                                first_line = doc.split('\n')[0].strip()
                                # Remove common prefixes
                                first_line = re.sub(r'^(Compute|Return|Calculate|Get)\s+', '', first_line, flags=re.IGNORECASE)
                                if first_line and len(first_line) < 100:  # Reasonable length
                                    descriptions[feature_name] = first_line
            except Exception:
                pass
        
        return descriptions
    
    def update_description(self):
        """Update the description of the selected feature set."""
        current_item = self.feature_set_list.currentItem()
        if not current_item:
            return
        
        feature_set = current_item.data(Qt.ItemDataRole.UserRole)
        if not feature_set:
            return
        
        if not HAS_FEATURE_SET_MANAGER:
            QMessageBox.warning(
                self,
                "Not Available",
                "Feature set manager is not available."
            )
            return
        
        description = self.edit_desc_edit.text().strip() or None
        
        try:
            update_feature_set_metadata(
                feature_set=feature_set,
                description=description
            )
            QMessageBox.information(
                self,
                "Success",
                f"Description updated for feature set '{feature_set}'!"
            )
            
            # Reload info to show updated description
            self.on_selection_changed()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to update description: {e}"
            )
    
    def delete_feature_set(self):
        """Delete the selected feature set."""
        current_item = self.feature_set_list.currentItem()
        if not current_item:
            return
        
        feature_set = current_item.data(Qt.ItemDataRole.UserRole)
        if not feature_set or feature_set == DEFAULT_FEATURE_SET:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                f"Cannot delete default feature set '{DEFAULT_FEATURE_SET}'."
            )
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete feature set '{feature_set}'?\n\n"
            "This will delete:\n"
            "- Config files\n"
            "- Feature implementation code\n\n"
            "Data directory will be kept unless you delete it manually.\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Use command line tool to delete
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "src" / "manage_feature_sets.py"), 
                     "delete", feature_set, "--force"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Feature set '{feature_set}' deleted successfully!"
                    )
                    self.load_feature_sets()
                else:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Failed to delete feature set:\n{result.stderr}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete feature set: {e}"
                )


class FeatureListDialog(QDialog):
    """Dialog for viewing features in a feature set."""
    
    def __init__(self, parent=None, feature_set: str = "", features: dict = None, descriptions: dict = None):
        super().__init__(parent)
        self.feature_set = feature_set
        self.features = features or {}
        self.descriptions = descriptions or {}
        self.setWindowTitle(f"Features: {feature_set}")
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Title
        title = QLabel(f"Features in '{self.feature_set}'")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Summary
        enabled_count = sum(1 for v in self.features.values() if v == 1)
        total_count = len(self.features)
        summary = QLabel(f"Total: {total_count} features | Enabled: {enabled_count} | Disabled: {total_count - enabled_count}")
        summary.setStyleSheet("color: #b0b0b0; font-size: 12px; margin-bottom: 10px;")
        layout.addWidget(summary)
        
        # Features table
        self.features_table = QTableWidget()
        self.features_table.setColumnCount(3)
        self.features_table.setHorizontalHeaderLabels(["Feature Name", "Status", "Value"])
        self.features_table.horizontalHeader().setStretchLastSection(True)
        self.features_table.setAlternatingRowColors(True)
        self.features_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.features_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Populate table
        self.features_table.setRowCount(len(self.features))
        
        for row, (feature_name, value) in enumerate(sorted(self.features.items())):
            # Feature name
            name_item = QTableWidgetItem(feature_name)
            self.features_table.setItem(row, 0, name_item)
            
            # Status
            is_enabled = value == 1
            status_text = "Enabled" if is_enabled else "Disabled"
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(Qt.GlobalColor.green if is_enabled else Qt.GlobalColor.red)
            if is_enabled:
                status_item.setText("✓ Enabled")
            else:
                status_item.setText("✗ Disabled")
            self.features_table.setItem(row, 1, status_item)
            
            # Value
            value_item = QTableWidgetItem(str(value))
            self.features_table.setItem(row, 2, value_item)
        
        # Resize columns to content
        self.features_table.resizeColumnsToContents()
        self.features_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        
        layout.addWidget(self.features_table)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
