"""
Preset Management Dialog

Allows users to view, rename, and delete filter presets.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QInputDialog, QLineEdit
)
from PyQt6.QtCore import Qt
from pathlib import Path
from datetime import datetime

from gui.services import StopLossAnalysisService


class PresetManagementDialog(QDialog):
    """Dialog for managing filter presets."""
    
    def __init__(self, service: StopLossAnalysisService, parent=None):
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("Manage Filter Presets")
        self.setMinimumSize(700, 500)
        self.init_ui()
        self.refresh_presets()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("Filter Presets")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4aa;")
        layout.addWidget(title)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Name", "Source Backtest", "Stop-Loss Rate", "Filters", "Created", "Actions"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        
        # Buttons
        btn_row = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_presets)
        btn_row.addWidget(refresh_btn)
        
        btn_row.addStretch()
        
        rename_btn = QPushButton("Rename")
        rename_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #ff9800;
                color: #000000;
            }
        """)
        rename_btn.clicked.connect(self.rename_selected)
        btn_row.addWidget(rename_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #f44336;
                color: #ffffff;
            }
        """)
        delete_btn.clicked.connect(self.delete_selected)
        btn_row.addWidget(delete_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        
        layout.addLayout(btn_row)
        self.setLayout(layout)
    
    def refresh_presets(self):
        """Refresh the preset list."""
        try:
            presets = self.service.list_presets()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load presets: {str(e)}")
            return
        
        self.table.setRowCount(len(presets))
        
        for row_idx, preset in enumerate(presets):
            # Name
            self.table.setItem(row_idx, 0, QTableWidgetItem(preset.get('name', 'Unknown')))
            
            # Source backtest
            source = preset.get('source_backtest', 'N/A')
            self.table.setItem(row_idx, 1, QTableWidgetItem(source))
            
            # Stop-loss rate
            sl_before = preset.get('stop_loss_rate_before', 0.0) * 100
            sl_after = preset.get('stop_loss_rate_after', 0.0) * 100
            sl_text = f"{sl_before:.1f}% → {sl_after:.1f}%"
            self.table.setItem(row_idx, 2, QTableWidgetItem(sl_text))
            
            # Filters count
            filters_count = preset.get('filters_count', 0)
            self.table.setItem(row_idx, 3, QTableWidgetItem(str(filters_count)))
            
            # Created date
            created = preset.get('created_date', '')
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    created = dt.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pass
            self.table.setItem(row_idx, 4, QTableWidgetItem(created))
            
            # Store filename for actions
            filename_item = QTableWidgetItem(preset.get('filename', ''))
            filename_item.setData(Qt.ItemDataRole.UserRole, preset.get('filename', ''))
            self.table.setItem(row_idx, 5, filename_item)
    
    def get_selected_filename(self) -> str:
        """Get filename of selected preset."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        
        row = selected_rows[0].row()
        filename_item = self.table.item(row, 5)
        if filename_item:
            return filename_item.data(Qt.ItemDataRole.UserRole)
        return None
    
    def rename_selected(self):
        """Rename selected preset."""
        filename = self.get_selected_filename()
        if not filename:
            QMessageBox.information(self, "No Selection", "Please select a preset to rename.")
            return
        
        # Load preset to get current name
        try:
            preset_data = self.service.load_preset(filename)
            current_name = preset_data.get('name', filename)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load preset: {str(e)}")
            return
        
        # Get new name
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Preset",
            "Enter new name:",
            text=current_name
        )
        
        if not ok or not new_name.strip():
            return
        
        # Rename
        try:
            success = self.service.rename_preset(filename, new_name.strip())
            if success:
                QMessageBox.information(self, "Success", "Preset renamed successfully.")
                self.refresh_presets()
            else:
                QMessageBox.warning(self, "Error", "Failed to rename preset.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to rename preset: {str(e)}")
    
    def delete_selected(self):
        """Delete selected preset."""
        filename = self.get_selected_filename()
        if not filename:
            QMessageBox.information(self, "No Selection", "Please select a preset to delete.")
            return
        
        # Load preset to get name for confirmation
        try:
            preset_data = self.service.load_preset(filename)
            preset_name = preset_data.get('name', filename)
        except Exception:
            preset_name = filename
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete preset '{preset_name}'?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                success = self.service.delete_preset(filename)
                if success:
                    QMessageBox.information(self, "Success", "Preset deleted successfully.")
                    self.refresh_presets()
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete preset.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete preset: {str(e)}")

