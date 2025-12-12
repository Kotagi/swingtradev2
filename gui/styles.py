"""
Styling and theme configuration for the GUI.
"""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtCore import Qt


class DarkTheme:
    """Dark theme color scheme for the trading application."""
    
    # Background colors
    BG_PRIMARY = QColor(30, 30, 30)      # #1e1e1e
    BG_SECONDARY = QColor(45, 45, 45)    # #2d2d2d
    BG_TERTIARY = QColor(60, 60, 60)     # #3c3c3c
    
    # Text colors
    TEXT_PRIMARY = QColor(224, 224, 224)  # #e0e0e0
    TEXT_SECONDARY = QColor(176, 176, 176)  # #b0b0b0
    TEXT_DISABLED = QColor(128, 128, 128)  # #808080
    
    # Accent colors
    ACCENT_PRIMARY = QColor(0, 212, 170)  # #00d4aa
    ACCENT_SECONDARY = QColor(0, 200, 150)  # #00c896
    
    # Status colors
    SUCCESS = QColor(76, 175, 80)         # #4caf50
    WARNING = QColor(255, 152, 0)        # #ff9800
    ERROR = QColor(244, 67, 54)          # #f44336
    INFO = QColor(33, 150, 243)          # #2196f3
    
    # Border colors
    BORDER = QColor(60, 60, 60)          # #3c3c3c
    BORDER_LIGHT = QColor(80, 80, 80)    # #505050
    
    @staticmethod
    def apply_theme(app):
        """Apply the dark theme to a QApplication."""
        app.setStyle("Fusion")
        
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.ColorRole.Window, DarkTheme.BG_PRIMARY)
        palette.setColor(QPalette.ColorRole.WindowText, DarkTheme.TEXT_PRIMARY)
        
        # Base colors (for input widgets)
        palette.setColor(QPalette.ColorRole.Base, DarkTheme.BG_SECONDARY)
        palette.setColor(QPalette.ColorRole.AlternateBase, DarkTheme.BG_TERTIARY)
        
        # Text colors
        palette.setColor(QPalette.ColorRole.Text, DarkTheme.TEXT_PRIMARY)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.white)
        
        # Button colors
        palette.setColor(QPalette.ColorRole.Button, DarkTheme.BG_SECONDARY)
        palette.setColor(QPalette.ColorRole.ButtonText, DarkTheme.TEXT_PRIMARY)
        
        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, DarkTheme.ACCENT_PRIMARY)
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, DarkTheme.TEXT_DISABLED)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, DarkTheme.TEXT_DISABLED)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, DarkTheme.TEXT_DISABLED)
        
        # Link colors
        palette.setColor(QPalette.ColorRole.Link, DarkTheme.ACCENT_PRIMARY)
        palette.setColor(QPalette.ColorRole.LinkVisited, DarkTheme.ACCENT_SECONDARY)
        
        app.setPalette(palette)
        
        # Set stylesheet for additional styling
        app.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
                background-color: #1e1e1e;
            }
            
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                border-bottom: 2px solid #00d4aa;
            }
            
            QTabBar::tab:hover {
                background-color: #3c3c3c;
            }
            
            QPushButton {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #3c3c3c;
                border-color: #00d4aa;
            }
            
            QPushButton:pressed {
                background-color: #1e1e1e;
            }
            
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
                border-color: #3c3c3c;
            }
            
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 6px;
            }
            
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #00d4aa;
            }
            
            QLabel {
                color: #e0e0e0;
            }
            
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #2d2d2d;
                color: #e0e0e0;
                gridline-color: #3c3c3c;
                border: 1px solid #3c3c3c;
            }
            
            QTableWidget::item {
                padding: 4px;
            }
            
            QTableWidget::item:selected {
                background-color: #00d4aa;
                color: #000000;
            }
            
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 6px;
                border: none;
                border-right: 1px solid #3c3c3c;
                border-bottom: 1px solid #3c3c3c;
            }
            
            QProgressBar {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
            }
            
            QProgressBar::chunk {
                background-color: #00d4aa;
            }
            
            QGroupBox {
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e0e0e0;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            
            QCheckBox::indicator:checked {
                background-color: #00d4aa;
                border-color: #00d4aa;
            }
            
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
        """)

