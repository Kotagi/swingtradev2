"""
Main entry point for the Swing Trading ML Application GUI.
"""

import sys
from PyQt6.QtWidgets import QApplication
from pathlib import Path

from gui.main_window import MainWindow
from gui.styles import DarkTheme


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    
    # Apply dark theme
    DarkTheme.apply_theme(app)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

