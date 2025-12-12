"""
Main window for the Swing Trading ML Application GUI.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QStatusBar, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from gui.help_panel import HelpDialog

from gui.tabs.dashboard_tab import DashboardTab
from gui.tabs.identify_tab import IdentifyTab
from gui.tabs.data_tab import DataTab
from gui.tabs.features_tab import FeaturesTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.backtest_tab import BacktestTab
from gui.tabs.analysis_tab import AnalysisTab


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swing Trading ML Application")
        self.setMinimumSize(1200, 800)
        
        # Set default window size (80% of screen size, but at least minimum)
        app = QApplication.instance()
        if app:
            screen = app.primaryScreen()
            if screen:
                screen_geometry = screen.availableGeometry()
                default_width = max(1200, int(screen_geometry.width() * 0.8))
                default_height = max(800, int(screen_geometry.height() * 0.8))
                self.resize(default_width, default_height)
            else:
                self.resize(1400, 900)
        else:
            self.resize(1400, 900)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(layout)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Add tabs
        # Tab 1: Dashboard
        dashboard_tab = DashboardTab()
        self.tabs.addTab(dashboard_tab, "Dashboard")
        
        # Tab 2: Trade Identification
        identify_tab = IdentifyTab()
        self.tabs.addTab(identify_tab, "Trade Identification")
        
        # Tab 3: Data Management
        data_tab = DataTab()
        self.tabs.addTab(data_tab, "Data Management")
        
        # Tab 4: Feature Engineering
        features_tab = FeaturesTab()
        self.tabs.addTab(features_tab, "Feature Engineering")
        
        # Tab 5: Model Training
        training_tab = TrainingTab()
        self.tabs.addTab(training_tab, "Model Training")
        
        # Tab 6: Backtesting
        backtest_tab = BacktestTab()
        self.tabs.addTab(backtest_tab, "Backtesting")
        
        # Tab 7: Analysis
        analysis_tab = AnalysisTab()
        self.tabs.addTab(analysis_tab, "Analysis")
        
        layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("color: #b0b0b0;")
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # F1 - Help
        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.activated.connect(self.show_help)
        
        # Ctrl+Q - Quit
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        quit_shortcut.activated.connect(self.close)
        
        # Ctrl+Tab - Next tab
        next_tab_shortcut = QShortcut(QKeySequence("Ctrl+Tab"), self)
        next_tab_shortcut.activated.connect(self.next_tab)
        
        # Ctrl+Shift+Tab - Previous tab
        prev_tab_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Tab"), self)
        prev_tab_shortcut.activated.connect(self.previous_tab)
        
        # Ctrl+1-7 - Switch to specific tab
        for i in range(1, 8):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{i}"), self)
            shortcut.activated.connect(lambda checked=False, idx=i-1: self.switch_to_tab(idx))
        
        # F5 - Refresh dashboard
        refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        refresh_shortcut.activated.connect(self.refresh_dashboard)
    
    def show_help(self):
        """Show help dialog."""
        dialog = HelpDialog(self)
        dialog.exec()
    
    def next_tab(self):
        """Switch to next tab."""
        current = self.tabs.currentIndex()
        next_idx = (current + 1) % self.tabs.count()
        self.tabs.setCurrentIndex(next_idx)
    
    def previous_tab(self):
        """Switch to previous tab."""
        current = self.tabs.currentIndex()
        prev_idx = (current - 1) % self.tabs.count()
        self.tabs.setCurrentIndex(prev_idx)
    
    def switch_to_tab(self, index: int):
        """Switch to specific tab by index."""
        if 0 <= index < self.tabs.count():
            self.tabs.setCurrentIndex(index)
    
    def refresh_dashboard(self):
        """Refresh dashboard if it's the current tab."""
        if self.tabs.currentIndex() == 0:  # Dashboard is first tab
            dashboard = self.tabs.currentWidget()
            if hasattr(dashboard, 'refresh_stats'):
                dashboard.refresh_stats()

