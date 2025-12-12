"""
Drag and Drop LineEdit

LineEdit widget that accepts file drops.
"""

from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent


class DragDropLineEdit(QLineEdit):
    """LineEdit that accepts file drag-and-drop."""
    
    def __init__(self, parent=None, accepted_extensions=None):
        super().__init__(parent)
        self.accepted_extensions = accepted_extensions or []
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_valid_file(file_path):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                if self.is_valid_file(file_path):
                    self.setText(file_path)
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def is_valid_file(self, file_path: str) -> bool:
        """Check if file is valid (has accepted extension if specified)."""
        if not self.accepted_extensions:
            return True
        
        from pathlib import Path
        path = Path(file_path)
        if path.is_file():
            ext = path.suffix.lower()
            return ext in [e.lower() for e in self.accepted_extensions]
        return False

