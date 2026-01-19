"""
Main entry point for the Swing Trading ML Application GUI.
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication
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

