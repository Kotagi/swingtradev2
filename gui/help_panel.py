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
        
        # Analysis tab
        analysis_tab = self.create_analysis_help()
        tabs.addTab(analysis_tab, "Analysis")
        
        # Stop-Loss Analysis tab
        stoploss_tab = self.create_stoploss_help()
        tabs.addTab(stoploss_tab, "Stop-Loss Analysis")
        
        # Filter Editor tab
        filter_editor_tab = self.create_filter_editor_help()
        tabs.addTab(filter_editor_tab, "Filter Editor")
        
        # Model Comparison tab
        model_comp_tab = self.create_model_comp_help()
        tabs.addTab(model_comp_tab, "Model Comparison")
        
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
        
        <h3>Complete Workflow</h3>
        <ol>
            <li><b>Data Management</b>: Download and clean stock data from yfinance</li>
            <li><b>Feature Engineering</b>: Build 57+ technical indicators</li>
            <li><b>Model Training</b>: Train XGBoost models with hyperparameter tuning (labels calculated on-the-fly)</li>
            <li><b>Backtesting</b>: Test strategies with adaptive stop-losses and entry filters</li>
            <li><b>Stop-Loss Analysis</b>: Analyze stop-loss patterns and generate filter recommendations</li>
            <li><b>Trade Identification</b>: Find current trading opportunities</li>
            <li><b>Analysis</b>: Review results, compare models/backtests, analyze performance</li>
        </ol>
        
        <h3>Key Features</h3>
        <ul>
            <li><b>57+ Technical Indicators</b>: Price, returns, volatility, momentum, volume, trend features</li>
            <li><b>Real-time Progress Tracking</b>: Accurate progress bars for all operations</li>
            <li><b>Configuration Presets</b>: Save/load settings for training and backtesting</li>
            <li><b>Advanced Visualizations</b>: Equity curves, returns distribution, drawdown charts, rolling metrics</li>
            <li><b>Stop-Loss Analysis</b>: Pattern identification, filter recommendations, impact preview</li>
            <li><b>Model Registry</b>: Automatic model registration and comparison</li>
            <li><b>Backtest Comparison</b>: Side-by-side comparison of backtest results</li>
            <li><b>Filter Presets</b>: Save and apply entry filters from stop-loss analysis</li>
            <li><b>Adaptive Stop-Losses</b>: ATR-based stops that adapt to volatility</li>
            <li><b>SHAP Explainability</b>: Model interpretability with feature importance rankings and visualizations</li>
        </ul>
        
        <h3>Documentation</h3>
        <p>For detailed documentation, see the <code>/info</code> folder in the project directory:</p>
        <ul>
            <li><b>PIPELINE_STEPS.md</b>: Complete guide to all pipeline steps</li>
            <li><b>FEATURE_GUIDE.md</b>: Documentation for all 57 technical indicators</li>
            <li><b>QUICKSTART.md</b>: Quick start guide</li>
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
        <p>Build 57+ technical indicators from cleaned data. Labels are now calculated during training, not during feature engineering.</p>
        
        <h3>Important Note</h3>
        <p><b>Labels are calculated during training</b> based on the horizon and return threshold you specify in the Model Training tab. 
        Feature engineering now only builds technical indicators, making features independent of your trading parameters.</p>
        
        <h3>Parameters</h3>
        <ul>
            <li><b>Feature Set</b>: Feature set version (v1, v2, or custom)</li>
            <li><b>Force Full Recompute</b>: Rebuild all features, ignoring cached files (unchecked by default - uses incremental updates)</li>
        </ul>
        
        <h3>Paths</h3>
        <p>Leave input/output directories empty to use defaults:</p>
        <ul>
            <li>Input: data/clean</li>
            <li>Output: data/features_labeled</li>
        </ul>
        
        <h3>Features Computed</h3>
        <p>The pipeline computes 57 technical indicators across 11 categories:</p>
        <ul>
            <li>Price Features (3): Raw price, log price, price vs MA200</li>
            <li>Return Features (6): Daily, weekly, monthly, quarterly, YTD returns</li>
            <li>52-Week Features (3): Distance to highs/lows, position in range</li>
            <li>Moving Averages (8): SMA/EMA ratios, slopes, crossovers</li>
            <li>Volatility (8): ATR, Bollinger Bands, TTM Squeeze, volatility ratios</li>
            <li>Volume (5): Log volume, relative volume, Chaikin Money Flow, OBV momentum</li>
            <li>Momentum (9): RSI, MACD, PPO, ROC, Stochastic, CCI, Williams %R</li>
            <li>Market Context (1): Beta vs SPY</li>
            <li>Candlestick (3): Body, wick percentages</li>
            <li>Price Action (4): Higher highs/lows, Donchian channels</li>
            <li>Trend (8): Trend residual, ADX, Aroon indicators</li>
        </ul>
        
        <h3>Delete Feature Data</h3>
        <p>Use the "Delete Feature Data" button to remove all feature files and start fresh.</p>
        """
        return self.create_scrollable_text(content)
    
    def create_training_help(self) -> QWidget:
        """Create model training help tab."""
        content = """
        <h2>Model Training</h2>
        <p>Train XGBoost machine learning models with hyperparameter tuning. Labels are calculated on-the-fly based on your horizon and return threshold.</p>
        
        <h3>Label Calculation</h3>
        <p><b>Labels are calculated during training</b> based on:</p>
        <ul>
            <li><b>Horizon (Trading Days)</b>: How many trading days to look ahead (1-365)</li>
            <li><b>Return Threshold (%)</b>: Minimum return to consider a win (0-100%)</li>
        </ul>
        <p>This means you can train models with different parameters without rebuilding features!</p>
        
        <h3>Training Options</h3>
        <ul>
            <li><b>Hyperparameter Tuning</b>: Enable RandomizedSearchCV for parameter optimization (recommended)</li>
            <li><b>Tuning Iterations</b>: Number of parameter combinations to try (5-9999, default: 20)</li>
            <li><b>Early Stopping Rounds</b>: Number of rounds without improvement before stopping (1-1000, default: 50)</li>
            <li><b>Cross-Validation</b>: Use k-fold cross-validation for more robust evaluation</li>
            <li><b>CV Folds</b>: Number of cross-validation folds (2-10, auto: 3 or 2 if fast mode)</li>
            <li><b>Fast Mode</b>: Faster training with reduced search space (~3-5x faster)</li>
            <li><b>Generate Plots</b>: Create training curves and feature importance charts</li>
            <li><b>Compute SHAP Explanations</b>: Generate SHAP values for model interpretability (recommended, adds ~30-60 seconds)</li>
            <li><b>SHAP Diagnostics (Legacy)</b>: Legacy SHAP diagnostics - use 'Compute SHAP Explanations' instead</li>
            <li><b>Disable Early Stopping</b>: Train for full number of rounds (not recommended)</li>
        </ul>
        
        <h3>Advanced Parameters</h3>
        <ul>
            <li><b>Horizon (Trading Days)</b>: Trade window in trading days (1-365, default: 30)</li>
            <li><b>Return Threshold (%)</b>: Return threshold for label calculation (0-100%, default: 5%)</li>
            <li><b>Class Imbalance Multiplier</b>: Adjust sample weights for imbalanced classes (0.5-5.0, default: 1.0)</li>
            <li><b>Feature Set</b>: Feature set version to use</li>
            <li><b>Model Output</b>: Custom path for saved model (auto-named if empty)</li>
        </ul>
        
        <h3>Model Registry</h3>
        <p>Models are automatically registered in the model registry with metrics, parameters, and training info. 
        View and compare models in the "Model Comparison" tab.</p>
        
        <h3>SHAP Explainability</h3>
        <p>Enable "Compute SHAP Explanations" to generate model interpretability artifacts:</p>
        <ul>
            <li>Feature importance rankings based on SHAP values</li>
            <li>Summary visualizations showing how features impact predictions</li>
            <li>View SHAP explanations in the Model Comparison tab</li>
            <li>Compare SHAP explanations between two models</li>
            <li>Recompute SHAP for models trained without it</li>
        </ul>
        
        <h3>Presets</h3>
        <p>Save your training configuration as a preset for easy reuse. Load presets to quickly restore settings.</p>
        """
        return self.create_scrollable_text(content)
    
    def create_backtest_help(self) -> QWidget:
        """Create backtesting help tab."""
        content = """
        <h2>Backtesting</h2>
        <p>Test your trading strategies on historical data with comprehensive metrics and visualizations.</p>
        
        <h3>Strategy Configuration</h3>
        <ul>
            <li><b>Strategy</b>: Trading strategy (model, oracle, rsi)</li>
            <li><b>Model File</b>: Path to trained model (for model strategy). Use dropdown button to select from available models.</li>
            <li><b>Model Threshold</b>: Probability threshold for entry signals (0-100%)</li>
        </ul>
        
        <h3>Backtest Parameters</h3>
        <ul>
            <li><b>Horizon (Trading Days)</b>: Number of trading days to hold positions (1-365)</li>
            <li><b>Return Threshold (%)</b>: Minimum return to consider a win (0-100%)</li>
            <li><b>Position Size</b>: Dollar amount per trade ($100-$100,000)</li>
        </ul>
        
        <h3>Stop-Loss Configuration</h3>
        <ul>
            <li><b>None</b>: No stop-loss</li>
            <li><b>Constant</b>: Fixed percentage stop-loss (e.g., -7.5%)</li>
            <li><b>Adaptive ATR</b>: ATR-based stop-loss that adapts to volatility
                <ul>
                    <li><b>ATR K</b>: ATR multiplier (default: 1.8)</li>
                    <li><b>Min Stop %</b>: Minimum stop distance (default: 4%)</li>
                    <li><b>Max Stop %</b>: Maximum stop distance (default: 10%)</li>
                </ul>
            </li>
        </ul>
        
        <h3>Entry Filters</h3>
        <p>Click "Apply Features" to configure entry filters:</p>
        <ul>
            <li>Select features and set thresholds (e.g., RSI > 50, volume > 1.5x average)</li>
            <li>Filters are applied post-backtest to filter trades before saving results</li>
            <li>Use "Load Preset" to apply saved filter presets from stop-loss analysis</li>
            <li>Click info (ℹ) buttons next to features for descriptions and interpretation</li>
        </ul>
        
        <h3>Output</h3>
        <p>Specify a CSV file path to save backtest results. If only a name is provided, it defaults to <code>data/backtest_results/</code> and automatically appends <code>.csv</code>.</p>
        <p>Results include all trades with entry/exit dates, prices, returns, P&L, exit reasons, and holding periods.</p>
        <p>After backtesting, visualizations are displayed: equity curve (cumulative P&L) and returns distribution.</p>
        
        <h3>Auto-Naming</h3>
        <p>If the output field is empty, a filename is auto-generated based on your parameters: <code>backtest_{strategy}_{horizon}d_{threshold}pct_{stop_loss}_{timestamp}.csv</code></p>
        """
        return self.create_scrollable_text(content)
    
    def create_analysis_help(self) -> QWidget:
        """Create analysis help tab."""
        content = """
        <h2>Analysis</h2>
        <p>Comprehensive analysis of backtest results with multiple views and comparison tools.</p>
        
        <h3>Backtest Comparison Tab</h3>
        <p>Compare multiple backtest results side-by-side:</p>
        <ul>
            <li>Automatically discovers all CSV files in <code>data/backtest_results/</code></li>
            <li>Displays key metrics: trades, win rate, P&L, Sharpe ratio, drawdown, annual return</li>
            <li>Shows which filters were applied (double-click to view details)</li>
            <li>Select multiple backtests for side-by-side comparison</li>
            <li>Search and filter by filename or minimum win rate</li>
            <li>Delete selected backtest files</li>
        </ul>
        
        <h3>Performance Metrics Tab</h3>
        <p>Detailed performance analysis of a single backtest:</p>
        <ul>
            <li>Load a backtest CSV file</li>
            <li>View comprehensive metrics table</li>
            <li>Visualizations:
                <ul>
                    <li><b>Equity Curve</b>: Cumulative P&L over time</li>
                    <li><b>Returns Distribution</b>: Histogram of trade returns</li>
                    <li><b>Performance Metrics</b>: Bar chart of key metrics</li>
                </ul>
            </li>
            <li>Export reports to CSV, HTML, or Excel</li>
        </ul>
        
        <h3>Trade Log Tab</h3>
        <p>Detailed trade-by-trade analysis:</p>
        <ul>
            <li>Load a backtest CSV file</li>
            <li>Filter trades by:
                <ul>
                    <li>Date range</li>
                    <li>Return percentage range</li>
                    <li>Ticker symbol (search)</li>
                </ul>
            </li>
            <li>Sortable table with all trade details</li>
            <li>Visualizations:
                <ul>
                    <li><b>Drawdown Chart</b>: Drawdown over time</li>
                    <li><b>Rolling Metrics</b>: Rolling Sharpe ratio and win rate</li>
                </ul>
            </li>
            <li>Export filtered trades to CSV, HTML, or Excel</li>
        </ul>
        
        <h3>Stop-Loss Analysis Tab</h3>
        <p>Analyze stop-loss patterns and generate filter recommendations (see Stop-Loss Analysis help tab for details).</p>
        
        <h3>Filter Editor Tab</h3>
        <p>Interactive filter editor with SHAP-guided feature importance and real-time impact calculation (see Filter Editor help tab for details).</p>
        """
        return self.create_scrollable_text(content)
    
    def create_filter_editor_help(self) -> QWidget:
        """Create filter editor help tab."""
        content = """
        <h2>Filter Editor</h2>
        <p>Interactive tool for testing and optimizing entry filters with real-time impact calculation and SHAP feature importance guidance.</p>
        
        <h3>Overview</h3>
        <p>The Filter Editor allows you to:</p>
        <ul>
            <li>Test different filter combinations on existing backtest results</li>
            <li>See real-time impact on winning trades, stop-losses, and performance metrics</li>
            <li>Use SHAP feature importance to guide filter selection</li>
            <li>Save and compare multiple filter presets</li>
        </ul>
        
        <h3>Getting Started</h3>
        <ol>
            <li><b>Load a Backtest CSV</b>: Click "Browse..." to load a backtest results file</li>
            <li><b>Wait for Feature Extraction</b>: Features are automatically extracted from the feature data files</li>
            <li><b>SHAP Importance</b>: If the backtest has model metadata, SHAP importance loads automatically</li>
            <li><b>Enable Filters</b>: Check the box next to features you want to filter on</li>
            <li><b>Adjust Values</b>: Use +/- buttons or type values directly</li>
            <li><b>View Impact</b>: Impact preview updates automatically as you change filters</li>
        </ol>
        
        <h3>Feature Controls</h3>
        <p>Each feature has the following controls:</p>
        <ul>
            <li><b>Enable Checkbox</b>: Turn the filter on/off</li>
            <li><b>Feature Name</b>: Display name with SHAP importance indicators:
                <ul>
                    <li>★ (green) = Top 10 most important features</li>
                    <li>● (gold) = Top 11-20 important features</li>
                    <li>○ (gray) = Lower importance features</li>
                </ul>
            </li>
            <li><b>Info Button (ℹ)</b>: View feature description and interpretation guide</li>
            <li><b>Operator</b>: Choose >, >=, <, or <=</li>
            <li><b>Value</b>: Set the threshold value (use +/- buttons or type directly)</li>
            <li><b>Increment</b>: Adjust the step size for +/- buttons</li>
        </ul>
        
        <h3>Grouping and Search</h3>
        <ul>
            <li><b>Search</b>: Filter features by name</li>
            <li><b>Group By</b>:
                <ul>
                    <li><b>Category</b>: Group by feature type (Price, Momentum, Volume, etc.)</li>
                    <li><b>None</b>: Show all features (sorted by SHAP importance if available)</li>
                    <li><b>SHAP Importance</b>: Group into tiers (Top 10, Top 11-20, Top 21-50, Lower, No SHAP Data)</li>
                </ul>
            </li>
            <li><b>Show Only Enabled</b>: Hide disabled filters</li>
        </ul>
        
        <h3>Impact Preview</h3>
        <p>The impact preview shows:</p>
        <ul>
            <li><b>Trade Volume</b>: Total trades, winners excluded, stop-losses excluded</li>
            <li><b>Performance Metrics</b>: P&L, win rate, Sharpe ratio, drawdown, profit factor, annual return</li>
            <li><b>Before/After Comparison</b>: Shows changes with color coding (green = improvement, red = decline)</li>
            <li><b>Warnings</b>: Alerts for excluding too many winners/trades, P&L reduction, etc.</li>
            <li><b>Errors</b>: Critical issues like all trades filtered out</li>
            <li><b>Improvements</b>: Positive changes highlighted in green</li>
            <li><b>Per-Feature Impact</b>: Individual impact of each enabled filter</li>
        </ul>
        
        <h3>Preset Management</h3>
        <ul>
            <li><b>Save Preset</b>: Save current filter configuration with a name and description</li>
            <li><b>Load Preset</b>: Load a previously saved filter preset</li>
            <li><b>Compare Presets</b>: Compare 2-4 presets side-by-side on the same backtest</li>
            <li><b>Clear All</b>: Disable all filters</li>
            <li><b>Reset All</b>: Reset all filters to default neutral values</li>
        </ul>
        
        <h3>SHAP Integration</h3>
        <p>If your backtest has model metadata, SHAP importance is automatically loaded:</p>
        <ul>
            <li>Features are color-coded by importance (green = high, gray = low)</li>
            <li>Rank and importance percentage shown in tooltips</li>
            <li>Group by "SHAP Importance" to focus on top features</li>
            <li>Features automatically sorted by importance when grouping is "None"</li>
        </ul>
        <p><b>Note</b>: SHAP data requires the backtest CSV to have a corresponding <code>_metadata.json</code> file with model information.</p>
        
        <h3>Tips</h3>
        <ul>
            <li>Start with SHAP top 10 features - they're most likely to have meaningful impact</li>
            <li>Use small increments when fine-tuning values</li>
            <li>Watch the impact preview warnings - they help avoid over-filtering</li>
            <li>Compare multiple presets to find the best filter combination</li>
            <li>Save presets with descriptive names for easy identification</li>
            <li>Test filters on multiple backtests to ensure robustness</li>
        </ul>
        """
        return self.create_scrollable_text(content)
    
    def create_stoploss_help(self) -> QWidget:
        """Create stop-loss analysis help tab."""
        content = """
        <h2>Stop-Loss Analysis</h2>
        <p>Analyze patterns in stop-loss trades to identify risk factors and generate filter recommendations.</p>
        
        <h3>Overview</h3>
        <p>This tool helps you understand why trades hit stop-losses and provides actionable filters to reduce stop-loss rate.</p>
        
        <h3>Steps</h3>
        <ol>
            <li>Load a backtest CSV file (one with stop-loss trades)</li>
            <li>Analysis runs automatically when a file is loaded</li>
            <li>Review summary cards: Total Trades, Stop-Losses, Winners, Target Reached</li>
            <li>Examine feature comparisons to see which features differ between stop-losses and winners</li>
            <li>Review recommendations grouped by strength (Strong, Moderate, Weak)</li>
            <li>Select individual recommendations using checkboxes</li>
            <li>Preview the impact of selected filters</li>
            <li>Save selected filters as a preset for use in backtesting</li>
        </ol>
        
        <h3>Feature Comparison</h3>
        <p>The table shows features that differ significantly between stop-loss trades and winners:</p>
        <ul>
            <li><b>Effect Size (Cohen's d)</b>: Magnitude of difference (larger = more significant)</li>
            <li><b>Stop-Loss Mean</b>: Average value for stop-loss trades</li>
            <li><b>Winner Mean</b>: Average value for winning trades</li>
            <li>Sort by effect size to find the most differentiating features</li>
        </ul>
        
        <h3>Recommendations</h3>
        <p>Recommendations are generated based on effect size thresholds:</p>
        <ul>
            <li><b>Strong</b>: Effect size > 0.5 (highly significant differences)</li>
            <li><b>Moderate</b>: Effect size 0.3-0.5 (moderate differences)</li>
            <li><b>Weak</b>: Effect size 0.2-0.3 (weak but potentially useful differences)</li>
        </ul>
        <p>Each recommendation includes:
            <ul>
                <li>Feature name and operator (>, <, >=, <=)</li>
                <li>Recommended threshold value</li>
                <li>Effect size and expected impact</li>
                <li>Info button (ℹ) for feature description</li>
            </ul>
        </p>
        
        <h3>Impact Preview</h3>
        <p>Shows estimated impact of applying selected filters:</p>
        <ul>
            <li>New stop-loss rate (if filters were applied)</li>
            <li>Number of stop-losses and winners that would be excluded</li>
            <li>Warnings if filters would exclude too many trades (>50% total or >20% winners)</li>
        </ul>
        
        <h3>Immediate Stop-Loss Analysis</h3>
        <p>Special analysis for trades that hit stop-loss within 1 day:</p>
        <ul>
            <li>These trades often have distinct patterns</li>
            <li>Select individual immediate recommendations using checkboxes</li>
            <li>Include selected recommendations in main recommendations</li>
            <li>Remove selected recommendations from main recommendations</li>
        </ul>
        
        <h3>Charts</h3>
        <p>Visual analysis of stop-loss patterns:</p>
        <ul>
            <li><b>Timing</b>: Day of week and month distributions</li>
            <li><b>Holding Period</b>: Histogram of days to stop-loss</li>
            <li><b>Returns</b>: Distribution of stop-loss return percentages</li>
        </ul>
        
        <h3>Export</h3>
        <p>Export analysis results:</p>
        <ul>
            <li><b>HTML Report</b>: Comprehensive report with all analysis (can be printed to PDF in browser)</li>
            <li><b>CSV Table</b>: Feature comparison table as CSV</li>
            <li><b>Export Preset</b>: Save selected filters as a preset for use in backtesting</li>
        </ul>
        
        <h3>Using Presets in Backtesting</h3>
        <p>After saving a preset, you can load it in the Backtesting tab:</p>
        <ol>
            <li>Go to Backtesting tab</li>
            <li>Click "Apply Features"</li>
            <li>Click "Load Preset" and select your saved preset</li>
            <li>Filters are automatically applied to your backtest</li>
        </ol>
        """
        return self.create_scrollable_text(content)
    
    def create_model_comp_help(self) -> QWidget:
        """Create model comparison help tab."""
        content = """
        <h2>Model Comparison</h2>
        <p>View and compare trained models side-by-side, including SHAP explainability.</p>
        
        <h3>Model List</h3>
        <p>The main table displays all registered models with key information:</p>
        <ul>
            <li><b>Select</b>: Checkbox to select models for comparison</li>
            <li><b>Model Name</b>: Auto-generated or custom model name (hover to see full file path)</li>
            <li><b>Date</b>: Training date</li>
            <li><b>Metrics</b>: ROC AUC, Accuracy, Precision, Recall, F1 Score, Average Precision</li>
            <li><b>Feature Set</b>: Feature set version used</li>
            <li><b>Features</b>: Number of enabled features (e.g., "45/57")</li>
            <li><b>Horizon</b>: Trading days used for label calculation</li>
            <li><b>Tuned</b>: Whether hyperparameter tuning was used</li>
            <li><b>CV</b>: Whether cross-validation was used</li>
            <li><b>Iterations</b>: Number of tuning iterations</li>
            <li><b>CV Folds</b>: Number of cross-validation folds</li>
            <li><b>Class Imbalance</b>: Class imbalance ratio</li>
        </ul>
        
        <h3>Search and Filter</h3>
        <ul>
            <li>Search by model name</li>
            <li>Filter by minimum ROC AUC</li>
            <li>Click "Refresh" to reload models from registry</li>
        </ul>
        
        <h3>Compare Selected Models</h3>
        <p>Select multiple models using checkboxes, then click "Compare Selected" to view side-by-side comparison:</p>
        <ul>
            <li>All metrics displayed in comparison table</li>
            <li>Easy to identify best-performing models</li>
            <li>Compare different hyperparameter configurations</li>
        </ul>
        
        <h3>SHAP Explainability</h3>
        <p>View and compare SHAP (SHapley Additive exPlanations) values for model interpretability:</p>
        <ul>
            <li><b>View SHAP</b>: View SHAP explanations for a single selected model
                <ul>
                    <li>Feature importance ranking (top features by SHAP value)</li>
                    <li>Summary statistics (data split, sample size, computation details)</li>
                    <li>SHAP summary plot visualization (beeswarm plot showing feature impacts)</li>
                    <li>Info button (ℹ) next to plot explains how to interpret SHAP plots</li>
                </ul>
            </li>
            <li><b>Compare SHAP</b>: Compare SHAP explanations between exactly 2 selected models
                <ul>
                    <li>Side-by-side feature importance rankings</li>
                    <li>Comparison summary highlighting differences</li>
                    <li>Identify which features gained/lost importance between models</li>
                </ul>
            </li>
            <li><b>Recompute SHAP</b>: Generate SHAP explanations for models trained without SHAP
                <ul>
                    <li>Select a single model without SHAP artifacts</li>
                    <li>Recomputes SHAP using validation data</li>
                    <li>Updates model registry with new SHAP artifacts</li>
                    <li>Progress shown during computation (may take several minutes)</li>
                </ul>
            </li>
        </ul>
        <p><b>Note:</b> SHAP requires the <code>shap</code> library. Install with <code>pip install shap</code>.</p>
        
        <h3>Model Management</h3>
        <ul>
            <li><b>Rename Selected</b>: Rename a model in the registry (hover over name to see full file path)</li>
            <li><b>Delete Selected</b>: Delete selected models from registry and remove model files from disk</li>
        </ul>
        
        <h3>Help Button</h3>
        <p>Click the "?" button for detailed explanations of all metrics and how to interpret them.</p>
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
            <li><b>Ctrl+1-8</b>: Switch to tab 1-8 (Dashboard, Identify, Data, Features, Training, Backtest, Analysis, Model Comparison)</li>
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

