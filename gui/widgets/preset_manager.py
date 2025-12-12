"""
Preset Manager Widget

Reusable widget for save/load preset functionality.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QComboBox, QLabel, QInputDialog, QMessageBox
)
from gui.config_manager import ConfigManager


class PresetManagerWidget(QWidget):
    """Widget for managing configuration presets."""
    
    def __init__(self, tab_name: str, parent=None):
        super().__init__(parent)
        self.tab_name = tab_name
        self.config_manager = ConfigManager()
        self.load_config_callback = None
        self.save_config_callback = None
        
        self.init_ui()
        self.refresh_presets()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Label
        label = QLabel("Preset:")
        layout.addWidget(label)
        
        # Preset dropdown
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.preset_combo.lineEdit().setPlaceholderText("Select or enter preset name...")
        self.preset_combo.setMinimumWidth(200)
        layout.addWidget(self.preset_combo)
        
        # Load button
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_preset)
        layout.addWidget(self.load_btn)
        
        # Save button
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_preset)
        layout.addWidget(self.save_btn)
        
        # Delete button
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self.delete_preset)
        layout.addWidget(self.delete_btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def refresh_presets(self):
        """Refresh the preset list."""
        self.preset_combo.clear()
        presets = self.config_manager.list_presets(self.tab_name)
        self.preset_combo.addItems(presets)
    
    def set_callbacks(self, load_callback, save_callback):
        """Set callbacks for loading and saving config."""
        self.load_config_callback = load_callback
        self.save_config_callback = save_callback
    
    def load_preset(self):
        """Load the selected preset."""
        preset_name = self.preset_combo.currentText().strip()
        if not preset_name:
            QMessageBox.warning(self, "No Preset", "Please select or enter a preset name.")
            return
        
        config = self.config_manager.load_preset(preset_name, self.tab_name)
        if config is None:
            QMessageBox.warning(self, "Preset Not Found", f"Preset '{preset_name}' not found.")
            return
        
        if self.load_config_callback:
            self.load_config_callback(config)
            QMessageBox.information(self, "Preset Loaded", f"Preset '{preset_name}' loaded successfully.")
        else:
            QMessageBox.warning(self, "No Load Handler", "No load callback configured.")
    
    def save_preset(self):
        """Save current configuration as a preset."""
        preset_name = self.preset_combo.currentText().strip()
        if not preset_name:
            preset_name, ok = QInputDialog.getText(
                self, "Save Preset", "Enter preset name:"
            )
            if not ok or not preset_name.strip():
                return
            preset_name = preset_name.strip()
        
        if self.save_config_callback:
            config = self.save_config_callback()
            if config is None:
                QMessageBox.warning(self, "Save Failed", "Could not retrieve current configuration.")
                return
            
            if self.config_manager.save_preset(preset_name, self.tab_name, config):
                QMessageBox.information(self, "Preset Saved", f"Preset '{preset_name}' saved successfully.")
                self.refresh_presets()
                # Select the newly saved preset
                index = self.preset_combo.findText(preset_name)
                if index >= 0:
                    self.preset_combo.setCurrentIndex(index)
            else:
                QMessageBox.critical(self, "Save Failed", "Failed to save preset.")
        else:
            QMessageBox.warning(self, "No Save Handler", "No save callback configured.")
    
    def delete_preset(self):
        """Delete the selected preset."""
        preset_name = self.preset_combo.currentText().strip()
        if not preset_name:
            QMessageBox.warning(self, "No Preset", "Please select a preset to delete.")
            return
        
        reply = QMessageBox.question(
            self, "Delete Preset",
            f"Are you sure you want to delete preset '{preset_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.config_manager.delete_preset(preset_name, self.tab_name):
                QMessageBox.information(self, "Preset Deleted", f"Preset '{preset_name}' deleted successfully.")
                self.refresh_presets()
            else:
                QMessageBox.warning(self, "Delete Failed", f"Could not delete preset '{preset_name}'.")

