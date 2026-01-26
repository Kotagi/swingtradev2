"""
Feature Set Selector Widget

A widget for selecting and managing feature sets in the GUI.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    # Try importing from src first (current location)
    try:
        from src.feature_set_manager import (
            list_feature_sets,
            feature_set_exists,
            DEFAULT_FEATURE_SET
        )
    except ImportError:
        # Fallback to direct import (if src is in path)
        from feature_set_manager import (
            list_feature_sets,
            feature_set_exists,
            DEFAULT_FEATURE_SET
        )
    HAS_FEATURE_SET_MANAGER = True
except ImportError as e:
    HAS_FEATURE_SET_MANAGER = False
    DEFAULT_FEATURE_SET = "v3_New_Dawn"
    print(f"Warning: Could not import feature_set_manager: {e}")


class FeatureSetSelector(QWidget):
    """Widget for selecting feature sets."""
    
    # Signal emitted when feature set changes
    feature_set_changed = pyqtSignal(str)
    
    def __init__(self, parent=None, show_manage_button=True):
        super().__init__(parent)
        self.show_manage_button = show_manage_button
        self.current_feature_set = DEFAULT_FEATURE_SET
        self.init_ui()
        self.load_feature_sets()
    
    def init_ui(self):
        """Initialize the UI components."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Label
        label = QLabel("Feature Set:")
        layout.addWidget(label)
        
        # Combo box
        self.combo = QComboBox()
        self.combo.setMinimumWidth(150)
        self.combo.currentTextChanged.connect(self.on_selection_changed)
        layout.addWidget(self.combo)
        
        # Manage button (optional)
        if self.show_manage_button:
            self.manage_btn = QPushButton("Manage")
            self.manage_btn.setMinimumWidth(90)
            self.manage_btn.setMinimumHeight(25)
            self.manage_btn.clicked.connect(self.show_manage_dialog)
            layout.addWidget(self.manage_btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_feature_sets(self):
        """Load available feature sets into the combo box."""
        self.combo.blockSignals(True)
        self.combo.clear()
        
        if not HAS_FEATURE_SET_MANAGER:
            self.combo.addItem(DEFAULT_FEATURE_SET)
            self.combo.blockSignals(False)
            return
        
        try:
            feature_sets = list_feature_sets()
            for fs in feature_sets:
                display_name = f"{fs} (default)" if fs == DEFAULT_FEATURE_SET else fs
                self.combo.addItem(display_name, fs)
            
            # Set current selection
            index = self.combo.findData(self.current_feature_set)
            if index >= 0:
                self.combo.setCurrentIndex(index)
            elif self.combo.count() > 0:
                self.combo.setCurrentIndex(0)
                self.current_feature_set = self.combo.currentData()
        except Exception as e:
            QMessageBox.warning(
                self,
                "Error Loading Feature Sets",
                f"Could not load feature sets: {e}\n\nUsing default: {DEFAULT_FEATURE_SET}"
            )
            self.combo.addItem(DEFAULT_FEATURE_SET, DEFAULT_FEATURE_SET)
            self.current_feature_set = DEFAULT_FEATURE_SET
        
        self.combo.blockSignals(False)
    
    def on_selection_changed(self, text):
        """Handle feature set selection change."""
        if self.combo.currentData():
            new_set = self.combo.currentData()
            if new_set != self.current_feature_set:
                self.current_feature_set = new_set
                self.feature_set_changed.emit(self.current_feature_set)
    
    def get_current_feature_set(self) -> str:
        """Get the currently selected feature set."""
        if self.combo.currentData():
            return self.combo.currentData()
        return self.current_feature_set
    
    def set_feature_set(self, feature_set: str):
        """Set the selected feature set."""
        if not HAS_FEATURE_SET_MANAGER:
            return
        
        if not feature_set_exists(feature_set):
            QMessageBox.warning(
                self,
                "Feature Set Not Found",
                f"Feature set '{feature_set}' does not exist.\nUsing default: {DEFAULT_FEATURE_SET}"
            )
            feature_set = DEFAULT_FEATURE_SET
        
        # Try findData first (if data is set)
        index = self.combo.findData(feature_set)
        if index < 0:
            # Fallback to findText if data isn't set
            index = self.combo.findText(feature_set)
        
        if index >= 0:
            self.combo.setCurrentIndex(index)
            self.current_feature_set = feature_set
            # Emit signal to notify other components
            self.feature_set_changed.emit(feature_set)
    
    def refresh(self):
        """Refresh the feature set list."""
        current = self.get_current_feature_set()
        self.load_feature_sets()
        self.set_feature_set(current)
    
    def show_manage_dialog(self):
        """Show feature set management dialog."""
        try:
            from gui.tabs.feature_set_management_dialog import FeatureSetManagementDialog
            dialog = FeatureSetManagementDialog(self)
            dialog.exec()
            # Refresh the selector after dialog closes (in case feature sets were added/deleted)
            self.refresh()
        except ImportError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Error",
                f"Could not load management dialog: {e}\n\n"
                "You can use the command line:\n"
                "python src/manage_feature_sets.py list\n"
                "python src/manage_feature_sets.py create <name>"
            )
