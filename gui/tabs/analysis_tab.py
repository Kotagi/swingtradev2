"""
Analysis Tab - Phase 4 Enhanced

Comprehensive analysis tools with trade log viewer, performance metrics, and visualizations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QComboBox, QSpinBox, QDateEdit, QTabWidget
)
from PyQt6.QtCore import Qt, QDate
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

from gui.services import DataService
from gui.widgets import (
    EquityCurveWidget, ReturnsDistributionWidget, PerformanceMetricsWidget,
    DrawdownChartWidget, RollingMetricsWidget
)
from gui.utils.report_exporter import ReportExporter

PROJECT_ROOT = Path(__file__).parent.parent.parent


class AnalysisTab(QWidget):
    """Enhanced tab for analysis and reports."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_service = DataService()
        self.current_trades_df = None
        self.current_metrics = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout for the tab
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Content widget
        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Analysis & Reports")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Tab widget for different analysis views
        self.tab_widget = QTabWidget()
        
        # Tab 1: Backtest Comparison
        from gui.tabs.backtest_comparison_tab import BacktestComparisonTab
        comparison_tab = BacktestComparisonTab()
        self.tab_widget.addTab(comparison_tab, "Backtest Comparison")
        
        # Tab 2: Performance Metrics
        metrics_tab = self.create_metrics_tab()
        self.tab_widget.addTab(metrics_tab, "Performance Metrics")
        
        # Tab 3: Trade Log Viewer
        trade_log_tab = self.create_trade_log_tab()
        self.tab_widget.addTab(trade_log_tab, "Trade Log")
        
        # Tab 4: Stop-Loss Analysis
        from gui.tabs.stop_loss_analysis_tab import StopLossAnalysisTab
        stop_loss_tab = StopLossAnalysisTab()
        self.tab_widget.addTab(stop_loss_tab, "Stop-Loss Analysis")
        
        layout.addWidget(self.tab_widget)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def create_trade_log_tab(self) -> QWidget:
        """Create trade log viewer tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Load file section
        load_group = QGroupBox("Load Backtest Results")
        load_layout = QVBoxLayout()
        
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Backtest CSV File:"))
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a backtest results CSV file...")
        file_row.addWidget(self.file_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_backtest_file)
        file_row.addWidget(browse_btn)
        
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self.load_backtest_results)
        file_row.addWidget(load_btn)
        
        load_layout.addLayout(file_row)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Filters section
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        
        # Date range filter
        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("Date Range:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addYears(-1))
        date_row.addWidget(self.start_date_edit)
        date_row.addWidget(QLabel("to"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        date_row.addWidget(self.end_date_edit)
        date_row.addStretch()
        filter_layout.addLayout(date_row)
        
        # Return filter
        return_row = QHBoxLayout()
        return_row.addWidget(QLabel("Min Return (%):"))
        self.min_return_spin = QSpinBox()
        self.min_return_spin.setRange(-100, 100)
        self.min_return_spin.setValue(-100)
        self.min_return_spin.setSuffix("%")
        return_row.addWidget(self.min_return_spin)
        
        return_row.addWidget(QLabel("Max Return (%):"))
        self.max_return_spin = QSpinBox()
        self.max_return_spin.setRange(-100, 100)
        self.max_return_spin.setValue(100)
        self.max_return_spin.setSuffix("%")
        return_row.addWidget(self.max_return_spin)
        
        return_row.addStretch()
        filter_layout.addLayout(return_row)
        
        # Ticker filter
        ticker_row = QHBoxLayout()
        ticker_row.addWidget(QLabel("Ticker:"))
        self.ticker_filter = QLineEdit()
        self.ticker_filter.setPlaceholderText("Filter by ticker (leave empty for all)")
        ticker_row.addWidget(self.ticker_filter)
        ticker_row.addStretch()
        filter_layout.addLayout(ticker_row)
        
        # Apply filters button
        apply_btn = QPushButton("Apply Filters")
        apply_btn.clicked.connect(self.apply_filters)
        filter_layout.addWidget(apply_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Visualizations
        viz_group = QGroupBox("Visualizations")
        viz_layout = QVBoxLayout()
        
        # Drawdown chart
        drawdown_label = QLabel("Drawdown Chart")
        drawdown_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        viz_layout.addWidget(drawdown_label)
        self.drawdown_chart = DrawdownChartWidget(self, width=10, height=4)
        self.drawdown_chart.setMinimumHeight(200)
        viz_layout.addWidget(self.drawdown_chart)
        
        # Rolling metrics
        rolling_label = QLabel("Rolling Metrics")
        rolling_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        viz_layout.addWidget(rolling_label)
        self.rolling_chart = RollingMetricsWidget(self, width=10, height=8)
        self.rolling_chart.setMinimumHeight(400)
        viz_layout.addWidget(self.rolling_chart)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Trade log table
        table_label = QLabel("Trade Log")
        table_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(table_label)
        
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            "Entry Date", "Exit Date", "Ticker", "Entry Price", "Exit Price",
            "Return", "P&L", "Holding Days"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.setAlternatingRowColors(True)
        self.trades_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.trades_table.setSortingEnabled(True)
        self.trades_table.setMinimumHeight(500)  # Show more trades at once
        layout.addWidget(self.trades_table)
        
        # Export button
        export_row = QHBoxLayout()
        export_row.addStretch()
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_trades)
        export_row.addWidget(export_btn)
        layout.addLayout(export_row)
        
        widget.setLayout(layout)
        return widget
    
    def create_metrics_tab(self) -> QWidget:
        """Create performance metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Load file section (reuse from trade log)
        load_group = QGroupBox("Load Backtest Results")
        load_layout = QVBoxLayout()
        
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Backtest CSV File:"))
        self.metrics_file_edit = QLineEdit()
        self.metrics_file_edit.setPlaceholderText("Select a backtest results CSV file...")
        file_row.addWidget(self.metrics_file_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_metrics_file)
        file_row.addWidget(browse_btn)
        
        load_btn = QPushButton("Load & Calculate Metrics")
        load_btn.clicked.connect(self.load_and_calculate_metrics)
        file_row.addWidget(load_btn)
        
        load_layout.addLayout(file_row)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Metrics display
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()
        
        # Metrics chart
        self.metrics_chart = PerformanceMetricsWidget(self, width=10, height=6)
        self.metrics_chart.setMinimumHeight(300)
        metrics_layout.addWidget(self.metrics_chart)
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setMaximumHeight(400)
        metrics_layout.addWidget(self.metrics_table)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Additional visualizations
        viz_group = QGroupBox("Additional Visualizations")
        viz_layout = QVBoxLayout()
        
        # Equity curve
        equity_label = QLabel("Equity Curve")
        equity_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        viz_layout.addWidget(equity_label)
        self.metrics_equity_chart = EquityCurveWidget(self, width=10, height=4)
        self.metrics_equity_chart.setMinimumHeight(200)
        viz_layout.addWidget(self.metrics_equity_chart)
        
        # Returns distribution
        returns_label = QLabel("Returns Distribution")
        returns_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-top: 10px;")
        viz_layout.addWidget(returns_label)
        self.metrics_returns_chart = ReturnsDistributionWidget(self, width=10, height=4)
        self.metrics_returns_chart.setMinimumHeight(200)
        viz_layout.addWidget(self.metrics_returns_chart)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        widget.setLayout(layout)
        return widget
    
    def create_comparison_tab(self) -> QWidget:
        """Create model comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Info
        info_group = QGroupBox("Model Comparison")
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            "Compare multiple models by loading their backtest results.\n\n"
            "1. Load backtest results for each model\n"
            "2. Models will be compared side-by-side\n"
            "3. View performance metrics comparison\n\n"
            "This feature will be enhanced in a future update."
        )
        info_text.setStyleSheet("color: #b0b0b0; font-size: 14px;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        # Model 1
        model1_row = QHBoxLayout()
        model1_row.addWidget(QLabel("Model 1 CSV:"))
        self.model1_file = QLineEdit()
        self.model1_file.setPlaceholderText("Select backtest results for model 1...")
        model1_row.addWidget(self.model1_file)
        browse1_btn = QPushButton("Browse...")
        browse1_btn.clicked.connect(lambda: self.browse_comparison_file(1))
        model1_row.addWidget(browse1_btn)
        info_layout.addLayout(model1_row)
        
        # Model 2
        model2_row = QHBoxLayout()
        model2_row.addWidget(QLabel("Model 2 CSV:"))
        self.model2_file = QLineEdit()
        self.model2_file.setPlaceholderText("Select backtest results for model 2...")
        model2_row.addWidget(self.model2_file)
        browse2_btn = QPushButton("Browse...")
        browse2_btn.clicked.connect(lambda: self.browse_comparison_file(2))
        model2_row.addWidget(browse2_btn)
        info_layout.addLayout(model2_row)
        
        # Compare button
        compare_btn = QPushButton("Compare Models")
        compare_btn.clicked.connect(self.compare_models)
        info_layout.addWidget(compare_btn)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels(["Metric", "Model 1", "Model 2"])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.setAlternatingRowColors(True)
        layout.addWidget(self.comparison_table)
        
        widget.setLayout(layout)
        return widget
    
    def browse_backtest_file(self):
        """Browse for backtest CSV file."""
        # Default to data/backtest_results directory
        backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backtest Results CSV",
            str(backtest_dir),
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.file_edit.setText(file_path)
    
    def browse_metrics_file(self):
        """Browse for metrics CSV file."""
        # Default to data/backtest_results directory
        backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backtest Results CSV",
            str(backtest_dir),
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.metrics_file_edit.setText(file_path)
    
    def browse_comparison_file(self, model_num: int):
        """Browse for comparison CSV file."""
        # Default to data/backtest_results directory
        backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Backtest Results CSV for Model {model_num}",
            str(backtest_dir),
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            if model_num == 1:
                self.model1_file.setText(file_path)
            else:
                self.model2_file.setText(file_path)
    
    def load_backtest_results(self):
        """Load backtest results from CSV file."""
        file_path = self.file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "No File", "Please select a CSV file.")
            return
        
        try:
            # Read CSV - entry_date is already a column (string format like '2024-01-24')
            # Read WITHOUT index_col since entry_date is a regular column
            df = pd.read_csv(file_path)
            
            # Convert date columns from strings to datetime
            for date_col in ['entry_date', 'exit_date']:
                if date_col in df.columns:
                    # Convert string dates to datetime
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Populate table FIRST (before normalize, which might modify dates)
            self.populate_trades_table(df)
            
            # Normalize for charts (after populating table)
            self.current_trades_df = self._normalize_trades_for_charts(df)
            
            # Update visualizations
            if not self.current_trades_df.empty:
                self.drawdown_chart.plot_drawdown(self.current_trades_df, initial_capital=10000.0)
                self.rolling_chart.plot_rolling_metrics(self.current_trades_df, window=20)
            
            QMessageBox.information(self, "Success", f"Loaded {len(self.current_trades_df)} trades.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")
    
    def populate_trades_table(self, df: pd.DataFrame):
        """Populate trades table with data."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Disable sorting during population to ensure EditRole values are set correctly
        self.trades_table.setSortingEnabled(False)
        self.trades_table.setRowCount(len(df))
        
        # CRITICAL: Ensure date columns are datetime BEFORE accessing values
        for date_col in ['entry_date', 'exit_date']:
            if date_col in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        for row_idx in range(len(df)):
            # Entry Date - get value and format
            entry_date_str = "N/A"
            entry_date_sort = 0
            
            if 'entry_date' in df.columns:
                entry_date_val = df.iloc[row_idx]['entry_date']
                
                # At this point, entry_date_val should be a Timestamp (we converted the column above)
                if pd.notna(entry_date_val):
                    if isinstance(entry_date_val, pd.Timestamp):
                        entry_date_str = entry_date_val.strftime('%m-%d-%Y')
                        entry_date_sort = entry_date_val.timestamp()
                    else:
                        # Fallback: convert if somehow not a Timestamp
                        entry_date_ts = pd.to_datetime(entry_date_val, errors='coerce')
                        if isinstance(entry_date_ts, pd.Timestamp) and pd.notna(entry_date_ts):
                            entry_date_str = entry_date_ts.strftime('%m-%d-%Y')
                            entry_date_sort = entry_date_ts.timestamp()
            
            # Create item - ONLY set text, don't set EditRole with numeric value
            # Setting EditRole with a number can cause Qt to display the number instead of text
            entry_item = QTableWidgetItem(entry_date_str)
            # Don't set EditRole with numeric value - it causes display issues
            # String dates will sort correctly as strings (YYYY-MM-DD format sorts chronologically)
            self.trades_table.setItem(row_idx, 0, entry_item)
            
            # Exit Date - get value and format
            exit_date_str = "N/A"
            exit_date_sort = 0
            
            if 'exit_date' in df.columns:
                exit_date_val = df.iloc[row_idx]['exit_date']
                
                # At this point, exit_date_val should be a Timestamp (we converted the column above)
                if pd.notna(exit_date_val):
                    if isinstance(exit_date_val, pd.Timestamp):
                        exit_date_str = exit_date_val.strftime('%m-%d-%Y')
                        exit_date_sort = exit_date_val.timestamp()
                    else:
                        # Fallback: convert if somehow not a Timestamp
                        exit_date_ts = pd.to_datetime(exit_date_val, errors='coerce')
                        if isinstance(exit_date_ts, pd.Timestamp) and pd.notna(exit_date_ts):
                            exit_date_str = exit_date_ts.strftime('%m-%d-%Y')
                            exit_date_sort = exit_date_ts.timestamp()
            
            # Create item - ONLY set text, don't set EditRole with numeric value
            # This ensures Qt displays the string, not a number
            exit_item = QTableWidgetItem(exit_date_str)
            # For sorting, we'll use the string comparison or set a custom sort key
            # But don't set EditRole with numeric value as it might cause display issues
            if exit_date_sort > 0:
                exit_item.setData(Qt.ItemDataRole.UserRole, exit_date_sort)
            self.trades_table.setItem(row_idx, 1, exit_item)
            
            # Ticker - string, no special handling needed
            if 'ticker' in df.columns:
                ticker = df.iloc[row_idx]['ticker']
            else:
                ticker = 'N/A'
            self.trades_table.setItem(row_idx, 2, QTableWidgetItem(str(ticker)))
            
            # Entry Price - store numeric value for sorting
            if 'entry_price' in df.columns:
                entry_price = df.iloc[row_idx]['entry_price']
            else:
                entry_price = 0
            if pd.notna(entry_price):
                entry_price_item = QTableWidgetItem(f"${entry_price:.2f}")
                entry_price_item.setData(Qt.ItemDataRole.EditRole, float(entry_price))  # Use EditRole for sorting
            else:
                entry_price_item = QTableWidgetItem("N/A")
                entry_price_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.trades_table.setItem(row_idx, 3, entry_price_item)
            
            # Exit Price - store numeric value for sorting
            if 'exit_price' in df.columns:
                exit_price = df.iloc[row_idx]['exit_price']
            else:
                exit_price = 0
            if pd.notna(exit_price):
                exit_price_item = QTableWidgetItem(f"${exit_price:.2f}")
                exit_price_item.setData(Qt.ItemDataRole.EditRole, float(exit_price))  # Use EditRole for sorting
            else:
                exit_price_item = QTableWidgetItem("N/A")
                exit_price_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.trades_table.setItem(row_idx, 4, exit_price_item)
            
            # Return - store numeric value for sorting
            if 'return' in df.columns:
                ret = df.iloc[row_idx]['return']
            else:
                ret = 0
            if pd.notna(ret):
                ret_item = QTableWidgetItem(f"{ret:.2%}")
                ret_item.setData(Qt.ItemDataRole.EditRole, float(ret))  # Use EditRole for sorting
                ret_item.setData(Qt.ItemDataRole.UserRole, ret)  # Keep UserRole for compatibility
                # Color code: green for positive, red for negative
                if ret > 0:
                    ret_item.setForeground(Qt.GlobalColor.green)
                else:
                    ret_item.setForeground(Qt.GlobalColor.red)
            else:
                ret_item = QTableWidgetItem("N/A")
                ret_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.trades_table.setItem(row_idx, 5, ret_item)
            
            # P&L - store numeric value for sorting
            if 'pnl' in df.columns:
                pnl = df.iloc[row_idx]['pnl']
            else:
                pnl = 0
            if pd.notna(pnl):
                pnl_item = QTableWidgetItem(f"${pnl:.2f}")
                pnl_item.setData(Qt.ItemDataRole.EditRole, float(pnl))  # Use EditRole for sorting
                pnl_item.setData(Qt.ItemDataRole.UserRole, pnl)  # Keep UserRole for compatibility
                if pnl > 0:
                    pnl_item.setForeground(Qt.GlobalColor.green)
                else:
                    pnl_item.setForeground(Qt.GlobalColor.red)
            else:
                pnl_item = QTableWidgetItem("N/A")
                pnl_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.trades_table.setItem(row_idx, 6, pnl_item)
            
            # Holding Days - store numeric value for sorting
            if 'holding_days' in df.columns:
                holding = df.iloc[row_idx]['holding_days']
            else:
                holding = 0
            if pd.notna(holding):
                # Convert to numeric, handling any string or mixed types
                holding_num = pd.to_numeric(holding, errors='coerce')
                if pd.isna(holding_num):
                    holding_num = 0
                else:
                    # Convert to int (holding days should be whole numbers)
                    holding_num = int(round(float(holding_num)))
                holding_item = QTableWidgetItem(f"{holding_num:.1f}")
                # Store as int in EditRole for proper numeric sorting
                # Use int to ensure consistent numeric sorting
                holding_item.setData(Qt.ItemDataRole.EditRole, int(holding_num))
            else:
                holding_item = QTableWidgetItem("N/A")
                # Store -1 for N/A values so they sort to bottom consistently
                holding_item.setData(Qt.ItemDataRole.EditRole, -1)
            self.trades_table.setItem(row_idx, 7, holding_item)
        
        # Re-enable sorting and sort by entry date descending by default
        self.trades_table.setSortingEnabled(True)
        self.trades_table.sortItems(0, Qt.SortOrder.DescendingOrder)
    
    def apply_filters(self):
        """Apply filters to trades table."""
        if self.current_trades_df is None or self.current_trades_df.empty:
            QMessageBox.warning(self, "No Data", "Please load backtest results first.")
            return
        
        df = self.current_trades_df.copy()
        
        # Date range filter
        start_date = self.start_date_edit.date().toPython()
        end_date = self.end_date_edit.date().toPython()
        
        if 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df = df[(df['entry_date'] >= pd.Timestamp(start_date)) & 
                   (df['entry_date'] <= pd.Timestamp(end_date))]
        
        # Return filter
        min_return = self.min_return_spin.value() / 100.0
        max_return = self.max_return_spin.value() / 100.0
        
        if 'return' in df.columns:
            df = df[(df['return'] >= min_return) & (df['return'] <= max_return)]
        
        # Ticker filter
        ticker_filter = self.ticker_filter.text().strip().upper()
        if ticker_filter and 'ticker' in df.columns:
            df = df[df['ticker'].str.upper().str.contains(ticker_filter, na=False)]
        
        # Update table
        self.populate_trades_table(df)
        
        # Update visualizations
        if not df.empty:
            self.drawdown_chart.plot_drawdown(df, initial_capital=10000.0)
            self.rolling_chart.plot_rolling_metrics(df, window=20)
    
    def export_trades(self):
        """Export filtered trades to CSV, HTML, Excel, or PDF."""
        if self.current_trades_df is None or self.current_trades_df.empty:
            QMessageBox.warning(self, "No Data", "No trades to export.")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "CSV Files (*.csv);;HTML Files (*.html);;Excel Files (*.xlsx);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            try:
                if selected_filter.startswith("CSV") or not selected_filter:
                    self.current_trades_df.to_csv(file_path, index=False)
                    QMessageBox.information(self, "Success", f"Trades exported to:\n{file_path}")
                elif selected_filter.startswith("HTML"):
                    # Calculate metrics if not already calculated
                    if self.current_metrics is None:
                        from src.enhanced_backtest import calculate_metrics
                        metrics = calculate_metrics(self.current_trades_df, 1000.0)
                    else:
                        metrics = self.current_metrics
                    
                    if ReportExporter.export_to_html(self.current_trades_df, metrics, file_path):
                        QMessageBox.information(self, "Success", f"HTML report exported to:\n{file_path}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to export HTML report.")
                elif selected_filter.startswith("Excel"):
                    # Calculate metrics if not already calculated
                    if self.current_metrics is None:
                        from src.enhanced_backtest import calculate_metrics
                        metrics = calculate_metrics(self.current_trades_df, 1000.0)
                    else:
                        metrics = self.current_metrics
                    
                    if ReportExporter.export_to_excel(self.current_trades_df, metrics, file_path):
                        QMessageBox.information(self, "Success", f"Excel report exported to:\n{file_path}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to export Excel report. Install openpyxl: pip install openpyxl")
                elif selected_filter.startswith("PDF"):
                    # Calculate metrics if not already calculated
                    if self.current_metrics is None:
                        from src.enhanced_backtest import calculate_metrics
                        metrics = calculate_metrics(self.current_trades_df, 1000.0)
                    else:
                        metrics = self.current_metrics
                    
                    if ReportExporter.export_to_pdf(self.current_trades_df, metrics, file_path):
                        QMessageBox.information(self, "Success", f"PDF report exported to:\n{file_path}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to export PDF report. Install reportlab: pip install reportlab")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    def load_and_calculate_metrics(self):
        """Load backtest results and calculate performance metrics."""
        file_path = self.metrics_file_edit.text().strip()
        if not file_path:
            QMessageBox.warning(self, "No File", "Please select a CSV file.")
            return
        
        try:
            trades_df = self._normalize_trades_for_charts(pd.read_csv(file_path))
            
            if trades_df.empty:
                QMessageBox.warning(self, "Empty File", "The CSV file contains no trades.")
                return
            
            # Calculate metrics (using the same logic as enhanced_backtest.py)
            from src.enhanced_backtest import calculate_metrics
            position_size = 1000.0  # Default, could be extracted from file or user input
            metrics = calculate_metrics(trades_df, position_size)
            self.current_metrics = metrics
            self.current_trades_df = trades_df  # Store for export
            
            # Display metrics
            self.display_metrics(metrics)
            
            # Update visualizations
            self.metrics_equity_chart.plot_cumulative_pnl(trades_df)
            self.metrics_returns_chart.plot_returns_distribution(trades_df)
            self.metrics_chart.plot_metrics(metrics)
            
            QMessageBox.information(self, "Success", "Metrics calculated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate metrics: {str(e)}")
    
    def display_metrics(self, metrics: dict):
        """Display performance metrics in table."""
        # Format metrics for display
        win_rate_val = metrics.get('win_rate', 0)
        annual_return_val = metrics.get('annual_return', 0)
        total_pnl_val = metrics.get('total_pnl', 0)
        
        display_metrics = {
            "Total Trades": f"{metrics.get('n_trades', 0):,}",
            "Win Rate": f"{win_rate_val * 100:.2f}%" if isinstance(win_rate_val, (int, float)) else "N/A",
            "Average Return": f"{metrics.get('avg_return', 0):.2%}",
            "Annual Return": f"{annual_return_val * 100:.2f}%" if isinstance(annual_return_val, (int, float)) else "N/A",
            "Total P&L": f"${total_pnl_val:,.2f}" if isinstance(total_pnl_val, (int, float)) else "N/A",
            "Average P&L": f"${metrics.get('avg_pnl', 0):,.2f}",
            "Maximum Drawdown": f"${metrics.get('max_drawdown', 0):,.2f}",
            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Profit Factor": f"{metrics.get('profit_factor', 0):.2f}",
            "Average Holding Days": f"{metrics.get('avg_holding_days', 0):.1f}",
            "Max Concurrent Positions": f"{metrics.get('max_concurrent_positions', 0)}",
            "Max Capital Invested": f"${metrics.get('max_capital_invested', 0):,.2f}",
            "Date Range": metrics.get('date_range', 'N/A')
        }
        
        self.metrics_table.setRowCount(len(display_metrics))
        
        for row_idx, (key, value) in enumerate(display_metrics.items()):
            key_item = QTableWidgetItem(key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.metrics_table.setItem(row_idx, 0, key_item)
            
            value_item = QTableWidgetItem(str(value))
            value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.metrics_table.setItem(row_idx, 1, value_item)
    
    def compare_models(self):
        """Compare two models side-by-side."""
        file1 = self.model1_file.text().strip()
        file2 = self.model2_file.text().strip()
        
        if not file1 or not file2:
            QMessageBox.warning(self, "Missing Files", "Please select CSV files for both models.")
            return
        
        try:
            # Load both files
            trades1 = pd.read_csv(file1)
            trades2 = pd.read_csv(file2)
            
            # Calculate metrics for both
            from src.enhanced_backtest import calculate_metrics
            position_size = 1000.0
            
            metrics1 = calculate_metrics(trades1, position_size)
            metrics2 = calculate_metrics(trades2, position_size)
            
            # Display comparison
            comparison_metrics = [
                ("Total Trades", metrics1.get('n_trades', 0), metrics2.get('n_trades', 0)),
                ("Win Rate", f"{metrics1.get('win_rate', 0) * 100:.2f}%", f"{metrics2.get('win_rate', 0) * 100:.2f}%"),
                ("Average Return", f"{metrics1.get('avg_return', 0):.2%}", f"{metrics2.get('avg_return', 0):.2%}"),
                ("Annual Return", f"{metrics1.get('annual_return', 0) * 100:.2f}%", f"{metrics2.get('annual_return', 0) * 100:.2f}%"),
                ("Total P&L", f"${metrics1.get('total_pnl', 0):,.2f}", f"${metrics2.get('total_pnl', 0):,.2f}"),
                ("Sharpe Ratio", f"{metrics1.get('sharpe_ratio', 0):.2f}", f"{metrics2.get('sharpe_ratio', 0):.2f}"),
                ("Profit Factor", f"{metrics1.get('profit_factor', 0):.2f}", f"{metrics2.get('profit_factor', 0):.2f}"),
                ("Max Drawdown", f"${metrics1.get('max_drawdown', 0):,.2f}", f"${metrics2.get('max_drawdown', 0):,.2f}"),
            ]
            
            self.comparison_table.setRowCount(len(comparison_metrics))
            
            for row_idx, (metric, val1, val2) in enumerate(comparison_metrics):
                metric_item = QTableWidgetItem(metric)
                metric_item.setFlags(metric_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.comparison_table.setItem(row_idx, 0, metric_item)
                
                val1_item = QTableWidgetItem(str(val1))
                val1_item.setFlags(val1_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.comparison_table.setItem(row_idx, 1, val1_item)
                
                val2_item = QTableWidgetItem(str(val2))
                val2_item.setFlags(val2_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.comparison_table.setItem(row_idx, 2, val2_item)
            
            QMessageBox.information(self, "Success", "Models compared successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compare models: {str(e)}")
    
    def _normalize_trades_for_charts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare trades DataFrame for charting: ensure return column and dates."""
        if df is None or df.empty:
            return pd.DataFrame()
        trades = df.copy()
        # Ensure return column
        if 'return' not in trades.columns:
            if 'ret' in trades.columns:
                trades['return'] = trades['ret']
            elif 'pnl' in trades.columns:
                # approximate returns from pnl with a heuristic position size
                pos_size = trades['pnl'].abs().median()
                if pos_size <= 0:
                    pos_size = max(1e-9, trades['pnl'].abs().mean())
                if pos_size <= 0:
                    pos_size = 1.0
                trades['return'] = trades['pnl'] / pos_size
            else:
                return pd.DataFrame()
        # Ensure numeric returns
        trades['return'] = pd.to_numeric(trades['return'], errors='coerce').fillna(0.0)
        # Ensure pnl is numeric if it exists
        if 'pnl' in trades.columns:
            trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce').fillna(0.0)
        # Ensure entry_date is a Timestamp (preserve if already converted)
        if 'entry_date' in trades.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades['entry_date']):
                trades['entry_date'] = pd.to_datetime(trades['entry_date'], errors='coerce')
        # Ensure exit_date for timing; fall back to entry_date or index
        if 'exit_date' in trades.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades['exit_date']):
                trades['exit_date'] = pd.to_datetime(trades['exit_date'], errors='coerce')
        elif 'entry_date' in trades.columns:
            trades['exit_date'] = trades['entry_date'].copy()
        else:
            # create a synthetic date index if none provided
            trades['exit_date'] = pd.to_datetime(pd.Series(range(len(trades))), errors='coerce')
        return trades
