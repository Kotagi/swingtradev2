"""
Help Panel

In-app help and documentation system.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel,
    QTabWidget, QWidget, QScrollArea
)
from PyQt6.QtCore import Qt


class HelpDialog(QDialog):
    """Help dialog with documentation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help & Documentation")
        self.setMinimumSize(800, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel("Swing Trading ML Application - Help")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Tab widget for different help sections
        tabs = QTabWidget()
        
        # Overview tab
        overview_tab = self.create_overview_tab()
        tabs.addTab(overview_tab, "Overview")
        
        # Dashboard tab
        dashboard_tab = self.create_dashboard_help()
        tabs.addTab(dashboard_tab, "Dashboard")
        
        # Trade Identification tab
        identify_tab = self.create_identify_help()
        tabs.addTab(identify_tab, "Trade Identification")
        
        # Data Management tab
        data_tab = self.create_data_help()
        tabs.addTab(data_tab, "Data Management")
        
        # Feature Engineering tab
        features_tab = self.create_features_help()
        tabs.addTab(features_tab, "Feature Engineering")
        
        # Model Training tab
        training_tab = self.create_training_help()
        tabs.addTab(training_tab, "Model Training")
        
        # Backtesting tab
        backtest_tab = self.create_backtest_help()
        tabs.addTab(backtest_tab, "Backtesting")
        
        # Keyboard Shortcuts tab
        shortcuts_tab = self.create_shortcuts_help()
        tabs.addTab(shortcuts_tab, "Keyboard Shortcuts")
        
        layout.addWidget(tabs)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def create_scrollable_text(self, content: str) -> QWidget:
        """Create a scrollable text widget."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(content)
        text_edit.setStyleSheet("background-color: #1e1e1e; color: #b0b0b0; padding: 10px;")
        
        layout.addWidget(text_edit)
        widget.setLayout(layout)
        return widget
    
    def create_overview_tab(self) -> QWidget:
        """Create overview help tab."""
        content = """
        <h2>Welcome to the Swing Trading ML Application</h2>
        <p>This application provides a complete GUI for managing your swing trading machine learning pipeline.</p>
        
        <h3>Workflow</h3>
        <ol>
            <li><b>Data Management</b>: Download and clean stock data</li>
            <li><b>Feature Engineering</b>: Build technical indicators and labels</li>
            <li><b>Model Training</b>: Train ML models with hyperparameter tuning</li>
            <li><b>Backtesting</b>: Test strategies on historical data</li>
            <li><b>Trade Identification</b>: Find current trading opportunities</li>
            <li><b>Analysis</b>: Review results and performance metrics</li>
        </ol>
        
        <h3>Key Features</h3>
        <ul>
            <li>Full feature parity with CLI commands</li>
            <li>Real-time progress tracking</li>
            <li>Configuration presets (save/load settings)</li>
            <li>Visualizations and charts</li>
            <li>Recent opportunities dashboard</li>
        </ul>
        """
        return self.create_scrollable_text(content)
    
    def create_dashboard_help(self) -> QWidget:
        """Create dashboard help tab."""
        content = """
        <h2>Dashboard</h2>
        <p>The dashboard provides an overview of your system status and recent opportunities.</p>
        
        <h3>System Status</h3>
        <ul>
            <li><b>Raw Data</b>: Number of downloaded CSV files</li>
            <li><b>Clean Data</b>: Number of cleaned Parquet files</li>
            <li><b>Features</b>: Number of feature files ready for training</li>
            <li><b>Models</b>: Number of trained model files</li>
        </ul>
        
        <h3>Recent Opportunities</h3>
        <p>Displays the top 10 most recent trading opportunities identified in the Trade Identification tab.</p>
        <p>Click "Refresh" to update the display with the latest results.</p>
        """
        return self.create_scrollable_text(content)
    
    def create_identify_help(self) -> QWidget:
        """Create trade identification help tab."""
        content = """
        <h2>Trade Identification</h2>
        <p>Find current trading opportunities using your trained model.</p>
        
        <h3>Steps</h3>
        <ol>
            <li>Load a trained model (select from dropdown or browse)</li>
            <li>Set minimum probability threshold (0-100%)</li>
            <li>Set number of top results to return</li>
            <li>Optionally enable recommended filters</li>
            <li>Configure stop-loss settings if desired</li>
            <li>Click "Identify Opportunities"</li>
        </ol>
        
        <h3>Parameters</h3>
        <ul>
            <li><b>Minimum Probability</b>: Minimum model prediction probability (0-100%)</li>
            <li><b>Top N Results</b>: Maximum number of opportunities to return</li>
            <li><b>Use Recommended Filters</b>: Apply filters from stop-loss analysis</li>
            <li><b>Stop-Loss Mode</b>: Configure adaptive or swing ATR stop-loss</li>
        </ul>
        
        <h3>Results</h3>
        <p>Results are displayed in a sortable table and automatically saved for dashboard display.</p>
        <p>You can export results to CSV for further analysis.</p>
        """
        return self.create_scrollable_text(content)
    
    def create_data_help(self) -> QWidget:
        """Create data management help tab."""
        content = """
        <h2>Data Management</h2>
        <p>Download and clean stock data for your pipeline.</p>
        
        <h3>Download Data</h3>
        <ul>
            <li><b>Tickers File</b>: CSV file with ticker symbols (one per line)</li>
            <li><b>Start Date</b>: Beginning date for historical data (YYYY-MM-DD)</li>
            <li><b>Full Download</b>: Re-download all tickers, ignoring existing files</li>
            <li><b>Resume</b>: Continue from where download left off</li>
            <li><b>Max Retries</b>: Maximum retry attempts for failed downloads</li>
        </ul>
        
        <h3>Clean Data</h3>
        <ul>
            <li><b>Workers</b>: Number of parallel workers for cleaning</li>
            <li><b>Resume</b>: Skip already cleaned files</li>
            <li><b>Full Clean</b>: Re-clean all files, ignoring existing cleaned files</li>
        </ul>
        
        <h3>Clear Downloads</h3>
        <p>Deletes all files in the raw data directory. Use with caution!</p>
        """
        return self.create_scrollable_text(content)
    
    def create_features_help(self) -> QWidget:
        """Create feature engineering help tab."""
        content = """
        <h2>Feature Engineering</h2>
        <p>Build technical indicators and labels from cleaned data.</p>
        
        <h3>Parameters</h3>
        <ul>
            <li><b>Trade Horizon</b>: Number of days to hold a position (1-365)</li>
            <li><b>Return Threshold</b>: Minimum return percentage to consider a win (0-100%)</li>
            <li><b>Feature Set</b>: Feature set version (v1, v2, or custom)</li>
            <li><b>Force Full Recompute</b>: Rebuild all features, ignoring cached files</li>
        </ul>
        
        <h3>Paths</h3>
        <p>Leave input/output directories empty to use defaults:</p>
        <ul>
            <li>Input: data/clean</li>
            <li>Output: data/features_labeled</li>
        </ul>
        """
        return self.create_scrollable_text(content)
    
    def create_training_help(self) -> QWidget:
        """Create model training help tab."""
        content = """
        <h2>Model Training</h2>
        <p>Train machine learning models with various configurations.</p>
        
        <h3>Options</h3>
        <ul>
            <li><b>Hyperparameter Tuning</b>: Enable RandomizedSearchCV for parameter optimization</li>
            <li><b>Tuning Iterations</b>: Number of parameter combinations to try (5-100)</li>
            <li><b>Cross-Validation</b>: Use k-fold cross-validation</li>
            <li><b>CV Folds</b>: Number of cross-validation folds (2-10)</li>
            <li><b>Fast Mode</b>: Faster training with less optimal parameters</li>
            <li><b>Generate Plots</b>: Create training curves and feature importance charts</li>
            <li><b>SHAP Diagnostics</b>: Compute SHAP values for model interpretation</li>
            <li><b>Disable Early Stopping</b>: Train for full number of rounds</li>
        </ul>
        
        <h3>Advanced Parameters</h3>
        <ul>
            <li><b>Class Imbalance Multiplier</b>: Adjust sample weights for imbalanced classes (0.5-5.0)</li>
            <li><b>Feature Set</b>: Feature set version to use</li>
            <li><b>Model Output</b>: Custom path for saved model</li>
        </ul>
        
        <h3>Presets</h3>
        <p>Save your training configuration as a preset for easy reuse. Load presets to quickly restore settings.</p>
        """
        return self.create_scrollable_text(content)
    
    def create_backtest_help(self) -> QWidget:
        """Create backtesting help tab."""
        content = """
        <h2>Backtesting</h2>
        <p>Test your trading strategies on historical data.</p>
        
        <h3>Strategy Configuration</h3>
        <ul>
            <li><b>Strategy</b>: Trading strategy (model, oracle, rsi)</li>
            <li><b>Model File</b>: Path to trained model (for model strategy)</li>
            <li><b>Model Threshold</b>: Probability threshold for entry signals (0-100%)</li>
        </ul>
        
        <h3>Backtest Parameters</h3>
        <ul>
            <li><b>Trade Horizon</b>: Number of days to hold positions (1-365)</li>
            <li><b>Return Threshold</b>: Minimum return to consider a win (0-100%)</li>
            <li><b>Position Size</b>: Dollar amount per trade ($100-$100,000)</li>
        </ul>
        
        <h3>Stop-Loss Configuration</h3>
        <ul>
            <li><b>None</b>: No stop-loss</li>
            <li><b>Constant</b>: Fixed percentage stop-loss</li>
            <li><b>Adaptive ATR</b>: ATR-based stop-loss with min/max bounds</li>
            <li><b>Swing ATR</b>: Swing low + ATR buffer stop-loss</li>
        </ul>
        
        <h3>Output</h3>
        <p>Specify a CSV file path to save backtest results. Results include all trades with entry/exit dates, prices, returns, and P&L.</p>
        """
        return self.create_scrollable_text(content)
    
    def create_shortcuts_help(self) -> QWidget:
        """Create keyboard shortcuts help tab."""
        content = """
        <h2>Keyboard Shortcuts</h2>
        
        <h3>General</h3>
        <ul>
            <li><b>F1</b>: Show help dialog</li>
            <li><b>Ctrl+Q</b>: Quit application</li>
            <li><b>Ctrl+1-7</b>: Switch to tab 1-7</li>
        </ul>
        
        <h3>Navigation</h3>
        <ul>
            <li><b>Ctrl+Tab</b>: Next tab</li>
            <li><b>Ctrl+Shift+Tab</b>: Previous tab</li>
        </ul>
        
        <h3>Operations</h3>
        <ul>
            <li><b>Ctrl+S</b>: Save preset (when in tab with presets)</li>
            <li><b>Ctrl+O</b>: Load preset (when in tab with presets)</li>
            <li><b>F5</b>: Refresh dashboard</li>
        </ul>
        """
        return self.create_scrollable_text(content)

