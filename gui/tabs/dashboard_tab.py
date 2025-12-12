"""
Dashboard Tab

Shows overview metrics and recent opportunities.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout, QPushButton,
    QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView
)
import pandas as pd
from PyQt6.QtCore import Qt
from pathlib import Path
import pandas as pd
from datetime import datetime

from gui.services import DataService, TradeIdentificationService

# Project root for path resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DashboardTab(QWidget):
    """Dashboard tab showing overview metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_service = DataService()
        self.init_ui()
        self.refresh_stats()
    
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
        title = QLabel("Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Stats grid
        stats_group = QGroupBox("System Status")
        stats_layout = QGridLayout()
        stats_layout.setSpacing(15)
        
        # Data status
        self.raw_data_label = QLabel("Raw Data: Checking...")
        self.raw_data_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self.raw_data_label, 0, 0)
        
        self.clean_data_label = QLabel("Clean Data: Checking...")
        self.clean_data_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self.clean_data_label, 0, 1)
        
        self.features_label = QLabel("Features: Checking...")
        self.features_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self.features_label, 1, 0)
        
        self.models_label = QLabel("Models: Checking...")
        self.models_label.setStyleSheet("font-size: 14px;")
        stats_layout.addWidget(self.models_label, 1, 1)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Performance metrics group (if recent backtest available)
        perf_group = QGroupBox("Recent Performance Metrics")
        perf_layout = QVBoxLayout()
        
        self.perf_info_label = QLabel("No recent backtest results found. Run a backtest to see metrics here.")
        self.perf_info_label.setStyleSheet("color: #b0b0b0; font-style: italic; font-size: 12px;")
        self.perf_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.perf_info_label.setWordWrap(True)
        perf_layout.addWidget(self.perf_info_label)
        
        # Metrics table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setMaximumHeight(200)
        self.metrics_table.setVisible(False)
        perf_layout.addWidget(self.metrics_table)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Recent opportunities
        opps_group = QGroupBox("Recent Opportunities")
        opps_layout = QVBoxLayout()
        
        # Table for opportunities
        self.opps_table = QTableWidget()
        self.opps_table.setColumnCount(4)
        self.opps_table.setHorizontalHeaderLabels(["Ticker", "Probability", "Price", "Stop Loss %"])
        self.opps_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.opps_table.setAlternatingRowColors(True)
        self.opps_table.setMaximumHeight(200)
        self.opps_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        opps_layout.addWidget(self.opps_table)
        
        # Info label
        self.opps_info_label = QLabel("No recent opportunities found. Use the 'Trade Identification' tab to find opportunities.")
        self.opps_info_label.setStyleSheet("color: #b0b0b0; font-style: italic; font-size: 12px;")
        self.opps_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.opps_info_label.setWordWrap(True)
        opps_layout.addWidget(self.opps_info_label)
        
        opps_group.setLayout(opps_layout)
        layout.addWidget(opps_group)
        
        # Refresh button
        refresh_layout = QHBoxLayout()
        refresh_btn_actual = QPushButton("Refresh")
        refresh_btn_actual.clicked.connect(self.refresh_stats)
        refresh_layout.addWidget(refresh_btn_actual)
        refresh_layout.addStretch()
        self.last_updated_label = QLabel("Last updated: Never")
        self.last_updated_label.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        refresh_layout.addWidget(self.last_updated_label)
        layout.addLayout(refresh_layout)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def refresh_stats(self):
        """Refresh dashboard statistics."""
        from datetime import datetime
        self.last_updated_label.setText(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
        
        # Check raw data
        raw_dir = Path(self.data_service.get_raw_dir())
        if raw_dir.exists():
            raw_files = list(raw_dir.glob("*.csv"))
            self.raw_data_label.setText(f"Raw Data: {len(raw_files)} files")
            self.raw_data_label.setStyleSheet("font-size: 14px; color: #4caf50;")
        else:
            self.raw_data_label.setText("Raw Data: No data")
            self.raw_data_label.setStyleSheet("font-size: 14px; color: #ff9800;")
        
        # Check clean data
        clean_dir = Path(self.data_service.get_clean_dir())
        if clean_dir.exists():
            clean_files = list(clean_dir.glob("*.parquet"))
            self.clean_data_label.setText(f"Clean Data: {len(clean_files)} files")
            self.clean_data_label.setStyleSheet("font-size: 14px; color: #4caf50;")
        else:
            self.clean_data_label.setText("Clean Data: No data")
            self.clean_data_label.setStyleSheet("font-size: 14px; color: #ff9800;")
        
        # Check features
        features_dir = Path(self.data_service.get_data_dir())
        if features_dir.exists():
            feature_files = list(features_dir.glob("*.parquet"))
            self.features_label.setText(f"Features: {len(feature_files)} files")
            self.features_label.setStyleSheet("font-size: 14px; color: #4caf50;")
        else:
            self.features_label.setText("Features: No data")
            self.features_label.setStyleSheet("font-size: 14px; color: #ff9800;")
        
        # Check models
        models_dir = Path(self.data_service.get_models_dir())
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            self.models_label.setText(f"Models: {len(model_files)} files")
            self.models_label.setStyleSheet("font-size: 14px; color: #4caf50;")
        else:
            self.models_label.setText("Models: No models")
            self.models_label.setStyleSheet("font-size: 14px; color: #ff9800;")
        
        # Load recent opportunities
        self._load_recent_opportunities()
        
        # Load recent performance metrics
        self._load_recent_metrics()
    
    def _load_recent_opportunities(self):
        """Load and display recent opportunities from saved files."""
        try:
            opps_dir = PROJECT_ROOT / "data" / "opportunities"
            latest_file = opps_dir / "latest_opportunities.csv"
            
            if latest_file.exists():
                df = pd.read_csv(latest_file)
                
                if not df.empty:
                    # Show top 10 opportunities
                    display_df = df.head(10).copy()
                    
                    self.opps_table.setRowCount(len(display_df))
                    self.opps_info_label.setVisible(False)
                    self.opps_table.setVisible(True)
                    
                    for row_idx, (_, row) in enumerate(display_df.iterrows()):
                        # Ticker
                        ticker_item = QTableWidgetItem(str(row.get('ticker', 'N/A')))
                        self.opps_table.setItem(row_idx, 0, ticker_item)
                        
                        # Probability
                        prob = row.get('probability', 0)
                        prob_item = QTableWidgetItem(f"{prob:.1%}")
                        self.opps_table.setItem(row_idx, 1, prob_item)
                        
                        # Price
                        price = row.get('current_price', None)
                        if pd.notna(price) and price is not None:
                            price_item = QTableWidgetItem(f"${price:.2f}")
                        else:
                            price_item = QTableWidgetItem("N/A")
                        self.opps_table.setItem(row_idx, 2, price_item)
                        
                        # Stop Loss %
                        stop_loss = row.get('stop_loss_pct', None)
                        if pd.notna(stop_loss) and stop_loss is not None:
                            stop_item = QTableWidgetItem(f"{stop_loss:.2%}")
                        else:
                            stop_item = QTableWidgetItem("N/A")
                        self.opps_table.setItem(row_idx, 3, stop_item)
                    
                    # Sort by probability descending
                    self.opps_table.sortItems(1, Qt.SortOrder.DescendingOrder)
                else:
                    self.opps_table.setVisible(False)
                    self.opps_info_label.setText("No opportunities found in recent results.")
                    self.opps_info_label.setVisible(True)
            else:
                self.opps_table.setVisible(False)
                self.opps_info_label.setText("No recent opportunities found. Use the 'Trade Identification' tab to find opportunities.")
                self.opps_info_label.setVisible(True)
        except Exception as e:
            self.opps_table.setVisible(False)
            self.opps_info_label.setText(f"Error loading opportunities: {str(e)}")
            self.opps_info_label.setVisible(True)
    
    def _load_recent_metrics(self):
        """Load and display recent performance metrics from backtest results."""
        try:
            # Look for most recent backtest CSV file
            # Check common locations
            search_dirs = [
                PROJECT_ROOT / "reports",
                PROJECT_ROOT,
                PROJECT_ROOT / "data"
            ]
            
            latest_file = None
            latest_time = 0
            
            for search_dir in search_dirs:
                if search_dir.exists():
                    csv_files = list(search_dir.glob("backtest_*.csv"))
                    for csv_file in csv_files:
                        mtime = csv_file.stat().st_mtime
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_file = csv_file
            
            if latest_file and latest_file.exists():
                # Load and calculate metrics
                trades_df = pd.read_csv(latest_file)
                
                if not trades_df.empty:
                    # Calculate metrics
                    from src.enhanced_backtest import calculate_metrics
                    position_size = 1000.0  # Default
                    metrics = calculate_metrics(trades_df, position_size)
                    
                    # Display key metrics
                    key_metrics = {
                        "Total Trades": f"{metrics.get('n_trades', 0):,}",
                        "Win Rate": f"{metrics.get('win_rate', 0):.2%}",
                        "Total P&L": f"${metrics.get('total_pnl', 0):,.2f}",
                        "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
                        "Max Drawdown": f"${metrics.get('max_drawdown', 0):,.2f}"
                    }
                    
                    self.metrics_table.setRowCount(len(key_metrics))
                    self.perf_info_label.setVisible(False)
                    self.metrics_table.setVisible(True)
                    
                    for row_idx, (key, value) in enumerate(key_metrics.items()):
                        key_item = QTableWidgetItem(key)
                        key_item.setFlags(key_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.metrics_table.setItem(row_idx, 0, key_item)
                        
                        value_item = QTableWidgetItem(str(value))
                        value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        self.metrics_table.setItem(row_idx, 1, value_item)
                else:
                    self.metrics_table.setVisible(False)
                    self.perf_info_label.setVisible(True)
            else:
                self.metrics_table.setVisible(False)
                self.perf_info_label.setVisible(True)
        except Exception as e:
            self.metrics_table.setVisible(False)
            self.perf_info_label.setText(f"Error loading metrics: {str(e)}")
            self.perf_info_label.setVisible(True)

