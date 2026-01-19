"""
Main window for the Swing Trading ML Application GUI.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QStatusBar, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from gui.help_panel import HelpDialog
from gui.widgets.feature_set_selector import FeatureSetSelector

from gui.tabs.dashboard_tab import DashboardTab
from gui.tabs.identify_tab import IdentifyTab
from gui.tabs.data_tab import DataTab
from gui.tabs.features_tab import FeaturesTab
from gui.tabs.training_tab import TrainingTab
from gui.tabs.backtest_tab import BacktestTab
from gui.tabs.analysis_tab import AnalysisTab
from gui.tabs.model_comparison_tab import ModelComparisonTab


class MainWindow(QMainWindow):
    """Main application window with tabbed interface."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swing Trading ML Application")
        self.setMinimumSize(1200, 800)
        
        # Set window to maximized by default
        self.setWindowState(Qt.WindowState.WindowMaximized)
        
        # Feature set selector (shared across all tabs)
        self.feature_set_selector = None
        
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
        
        # Store tab references for feature set updates
        self.features_tab = features_tab
        self.training_tab = training_tab
        self.backtest_tab = backtest_tab
        self.identify_tab = identify_tab
        
        # Create shared feature set selector (used by tabs that need it)
        self.feature_set_selector = FeatureSetSelector(show_manage_button=True)
        self.feature_set_selector.feature_set_changed.connect(self.on_feature_set_changed)
        
        # Pass feature set selector to Feature Engineering tab
        features_tab.set_feature_set_selector(self.feature_set_selector)
        
        # Notify tabs of initial feature set
        initial_feature_set = self.get_current_feature_set()
        self.on_feature_set_changed(initial_feature_set)
        
        # Tab 7: Analysis
        analysis_tab = AnalysisTab()
        self.tabs.addTab(analysis_tab, "Analysis")
        
        # Tab 8: Model Comparison
        model_comparison_tab = ModelComparisonTab()
        self.tabs.addTab(model_comparison_tab, "Model Comparison")
        
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
    
    def on_feature_set_changed(self, feature_set: str):
        """Handle feature set change - notify all tabs."""
        # Update status bar
        self.statusBar().showMessage(f"Feature set changed to: {feature_set}", 3000)
        
        # Notify specific tabs that support feature sets
        if hasattr(self, 'features_tab') and hasattr(self.features_tab, 'on_feature_set_changed'):
            self.features_tab.on_feature_set_changed(feature_set)
        if hasattr(self, 'training_tab') and hasattr(self.training_tab, 'on_feature_set_changed'):
            self.training_tab.on_feature_set_changed(feature_set)
        if hasattr(self, 'backtest_tab') and hasattr(self.backtest_tab, 'on_feature_set_changed'):
            self.backtest_tab.on_feature_set_changed(feature_set)
        if hasattr(self, 'identify_tab') and hasattr(self.identify_tab, 'on_feature_set_changed'):
            self.identify_tab.on_feature_set_changed(feature_set)
        
        # Also notify all tabs generically
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if hasattr(tab, 'on_feature_set_changed'):
                tab.on_feature_set_changed(feature_set)
    
    def get_current_feature_set(self) -> str:
        """Get the currently selected feature set."""
        if self.feature_set_selector:
            try:
                return self.feature_set_selector.get_current_feature_set()
            except Exception:
                return "v1"
        return "v1"

