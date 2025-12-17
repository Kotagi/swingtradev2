"""
Backtest Comparison Tab

Allows users to view and compare multiple backtest results side-by-side.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QMessageBox, QScrollArea, QCheckBox, QLineEdit, QComboBox,
    QDialog, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette
from pathlib import Path
from datetime import datetime
import pandas as pd
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.tabs.metrics_help_dialog import MetricsHelpDialog

# Import calculate_metrics from enhanced_backtest
try:
    from src.enhanced_backtest import calculate_metrics
except ImportError:
    # Fallback if import fails
    calculate_metrics = None


class BacktestComparisonTab(QWidget):
    """Tab for comparing backtest results."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.backtest_results_dir = PROJECT_ROOT / "data" / "backtest_results"
        self.backtest_data = []  # List of dicts with backtest info and metrics
        self.selected_backtests = []  # List of backtest IDs for comparison
        self.init_ui()
        self.refresh_backtest_list()
    
    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
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
        
        # Title row with help button
        title_row = QHBoxLayout()
        title = QLabel("Backtest Comparison")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        title_row.addWidget(title)
        title_row.addStretch()
        
        # Help button
        help_btn_container = QWidget()
        help_btn_container.setFixedSize(35, 35)
        help_btn_container.setToolTip("Click for help on understanding backtest metrics")
        help_btn_container.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border: 2px solid #00d4aa;
                border-radius: 17px;
            }
            QWidget:hover {
                background-color: #3d3d3d;
                border-color: #00ffcc;
            }
        """)
        help_btn_layout = QHBoxLayout(help_btn_container)
        help_btn_layout.setContentsMargins(0, 0, 0, 0)
        help_label = QLabel("?")
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        help_label.setStyleSheet("""
            QLabel {
                color: #00d4aa;
                font-size: 24px;
                font-weight: bold;
                background-color: transparent;
                border: none;
            }
        """)
        help_btn_layout.addWidget(help_label)
        # Make it clickable
        help_btn_container.mousePressEvent = lambda e: self.show_metrics_help()
        title_row.addWidget(help_btn_container)
        layout.addLayout(title_row)
        
        # Filter/Search section
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search by filename...")
        self.search_edit.textChanged.connect(self.refresh_backtest_list)
        filter_layout.addWidget(self.search_edit)
        
        filter_layout.addWidget(QLabel("Min Win Rate:"))
        self.min_winrate_combo = QComboBox()
        self.min_winrate_combo.addItems(["Any", "30%", "40%", "50%", "60%", "70%"])
        self.min_winrate_combo.currentTextChanged.connect(self.refresh_backtest_list)
        filter_layout.addWidget(self.min_winrate_combo)
        
        filter_layout.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_backtest_list)
        filter_layout.addWidget(refresh_btn)
        
        # Get refresh button height to match delete button
        refresh_height = refresh_btn.sizeHint().height()
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.setFixedHeight(refresh_height)
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b71c1c;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        delete_btn.clicked.connect(self.delete_selected_backtests)
        filter_layout.addWidget(delete_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Backtest list table
        list_label = QLabel("All Backtests")
        list_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)
        
        self.backtests_table = QTableWidget()
        self.backtests_table.setColumnCount(15)  # 15 columns including Filters Used
        self.backtests_table.setHorizontalHeaderLabels([
            "Select", "Filename", "Date", "Total Trades", "Win Rate", 
            "Avg Return", "Total P&L", "Avg P&L", "Max Drawdown", 
            "Sharpe Ratio", "Profit Factor", "Avg Hold Days", "Annual Return", "Date Range", "Filters Used"
        ])
        # Set Select column to fixed width
        self.backtests_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.backtests_table.setColumnWidth(0, 50)
        # All other columns stretch to fill available space
        for col in range(1, 15):  # Include column 14 (Filters Used)
            self.backtests_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        self.backtests_table.setAlternatingRowColors(True)
        self.backtests_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.backtests_table.setMinimumHeight(400)
        layout.addWidget(self.backtests_table)
        
        # Comparison section
        comparison_group = QGroupBox("Comparison")
        comparison_layout = QVBoxLayout()
        
        # Selected backtests info
        self.selected_label = QLabel("No backtests selected for comparison")
        self.selected_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        comparison_layout.addWidget(self.selected_label)
        
        # Comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(0)
        self.comparison_table.setRowCount(0)
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setMinimumHeight(300)
        # Set minimum width for vertical header to prevent text cutoff
        self.comparison_table.verticalHeader().setMinimumWidth(150)
        comparison_layout.addWidget(self.comparison_table)
        
        # Compare button
        compare_btn = QPushButton("Compare Selected")
        compare_btn.clicked.connect(self.compare_selected_backtests)
        comparison_layout.addWidget(compare_btn)
        
        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def refresh_backtest_list(self):
        """Refresh the list of backtest files."""
        # Clear existing data
        self.backtest_data = []
        
        # Get search filter
        name_filter = self.search_edit.text().strip().lower()
        
        # Get min win rate filter
        min_winrate_text = self.min_winrate_combo.currentText()
        min_winrate = None
        if min_winrate_text != "Any":
            min_winrate = float(min_winrate_text.replace("%", "")) / 100.0
        
        # Discover backtest CSV files
        if not self.backtest_results_dir.exists():
            self.backtest_results_dir.mkdir(parents=True, exist_ok=True)
            self.backtests_table.setRowCount(0)
            return
        
        csv_files = sorted(self.backtest_results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not csv_files:
            self.backtests_table.setRowCount(0)
            return
        
        # Process each CSV file
        for csv_file in csv_files:
            try:
                # Load the CSV (don't use first column as index, in case entry_date is first column)
                df = pd.read_csv(csv_file, index_col=False)
                
                # Check if DataFrame is empty
                if df.empty:
                    continue
                
                # Normalize column names to lowercase for case-insensitive matching
                df.columns = df.columns.str.lower().str.strip()
                
                # Check if required columns exist
                required_cols = ['return', 'pnl']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    continue
                
                # Calculate holding_days if not present (needed for calculate_metrics)
                if 'holding_days' not in df.columns:
                    if 'entry_date' in df.columns and 'exit_date' in df.columns:
                        try:
                            entry_dates = pd.to_datetime(df['entry_date'], errors='coerce')
                            exit_dates = pd.to_datetime(df['exit_date'], errors='coerce')
                            df['holding_days'] = (exit_dates - entry_dates).dt.days
                            # Fill any NaN values with 0
                            df['holding_days'] = df['holding_days'].fillna(0)
                        except Exception:
                            # Set default holding days
                            df['holding_days'] = 0
                    else:
                        # No date columns, set default
                        df['holding_days'] = 0
                
                # Calculate metrics
                if calculate_metrics is None:
                    QMessageBox.warning(self, "Import Error", "Could not import calculate_metrics function.")
                    return
                
                # Determine position size (default 1000, could be extracted from CSV if available)
                position_size = 1000.0
                try:
                    metrics = calculate_metrics(df, position_size)
                    # Verify metrics were calculated
                    if not metrics or 'n_trades' not in metrics:
                        continue
                except Exception:
                    continue
                
                # Get file info
                file_date = datetime.fromtimestamp(csv_file.stat().st_mtime)
                filename = csv_file.name
                
                # Load metadata if available
                metadata = None
                metadata_path = csv_file.parent / f"{csv_file.stem}_metadata.json"
                if metadata_path.exists():
                    try:
                        import json
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception:
                        pass
                
                # Apply filters
                if name_filter and name_filter not in filename.lower():
                    continue
                
                if min_winrate is not None and metrics.get("win_rate", 0.0) < min_winrate:
                    continue
                
                # Store backtest data
                backtest_id = str(csv_file)
                self.backtest_data.append({
                    "id": backtest_id,
                    "filename": filename,
                    "file_path": str(csv_file),
                    "date": file_date,
                    "metrics": metrics,
                    "metadata": metadata
                })
            except Exception:
                # Skip files that can't be parsed
                continue
        
        # Populate table
        self.backtests_table.setRowCount(len(self.backtest_data))
        self.backtests_table.setSortingEnabled(False)
        
        for row_idx, backtest in enumerate(self.backtest_data):
            # Select checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(backtest.get("id") in self.selected_backtests)
            checkbox.stateChanged.connect(
                lambda state, bid=backtest.get("id"): self.toggle_backtest_selection(bid, state == Qt.CheckState.Checked.value)
            )
            # Center the checkbox
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.backtests_table.setCellWidget(row_idx, 0, checkbox_widget)
            
            # Filename
            filename_item = QTableWidgetItem(backtest.get("filename", "Unknown"))
            self.backtests_table.setItem(row_idx, 1, filename_item)
            
            # Date
            file_date = backtest.get("date")
            if file_date:
                date_str = file_date.strftime('%Y-%m-%d %H:%M')
            else:
                date_str = "Unknown"
            date_item = QTableWidgetItem(date_str)
            self.backtests_table.setItem(row_idx, 2, date_item)
            
            # Metrics
            metrics = backtest.get("metrics", {})
            
            # Total Trades
            n_trades = metrics.get("n_trades", 0)
            trades_item = QTableWidgetItem(str(n_trades))
            trades_item.setData(Qt.ItemDataRole.EditRole, n_trades)
            self.backtests_table.setItem(row_idx, 3, trades_item)
            
            # Win Rate
            win_rate = metrics.get("win_rate", 0.0)
            if win_rate is not None and not pd.isna(win_rate):
                winrate_item = QTableWidgetItem(f"{win_rate:.2%}")
                winrate_item.setData(Qt.ItemDataRole.EditRole, float(win_rate))
            else:
                winrate_item = QTableWidgetItem("N/A")
                winrate_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 4, winrate_item)
            
            # Avg Return
            avg_return = metrics.get("avg_return", 0.0)
            if avg_return is not None and not pd.isna(avg_return):
                avgret_item = QTableWidgetItem(f"{avg_return:.2%}")
                avgret_item.setData(Qt.ItemDataRole.EditRole, float(avg_return))
            else:
                avgret_item = QTableWidgetItem("N/A")
                avgret_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 5, avgret_item)
            
            # Total P&L
            total_pnl = metrics.get("total_pnl", 0.0)
            if total_pnl is not None and not pd.isna(total_pnl):
                pnl_item = QTableWidgetItem(f"${total_pnl:,.2f}")
                pnl_item.setData(Qt.ItemDataRole.EditRole, float(total_pnl))
            else:
                pnl_item = QTableWidgetItem("N/A")
                pnl_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 6, pnl_item)
            
            # Avg P&L
            avg_pnl = metrics.get("avg_pnl", 0.0)
            if avg_pnl is not None and not pd.isna(avg_pnl):
                avgpnl_item = QTableWidgetItem(f"${avg_pnl:,.2f}")
                avgpnl_item.setData(Qt.ItemDataRole.EditRole, float(avg_pnl))
            else:
                avgpnl_item = QTableWidgetItem("N/A")
                avgpnl_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 7, avgpnl_item)
            
            # Max Drawdown
            max_dd = metrics.get("max_drawdown", 0.0)
            if max_dd is not None and not pd.isna(max_dd):
                dd_item = QTableWidgetItem(f"${max_dd:,.2f}")
                dd_item.setData(Qt.ItemDataRole.EditRole, float(max_dd))
            else:
                dd_item = QTableWidgetItem("N/A")
                dd_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 8, dd_item)
            
            # Sharpe Ratio
            sharpe = metrics.get("sharpe_ratio", 0.0)
            if sharpe is not None and not pd.isna(sharpe):
                sharpe_item = QTableWidgetItem(f"{sharpe:.2f}")
                sharpe_item.setData(Qt.ItemDataRole.EditRole, float(sharpe))
            else:
                sharpe_item = QTableWidgetItem("N/A")
                sharpe_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 9, sharpe_item)
            
            # Profit Factor
            pf = metrics.get("profit_factor", 0.0)
            if pf is not None and not pd.isna(pf) and pf > 0:
                pf_item = QTableWidgetItem(f"{pf:.2f}")
                pf_item.setData(Qt.ItemDataRole.EditRole, float(pf))
            else:
                pf_item = QTableWidgetItem("N/A")
                pf_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 10, pf_item)
            
            # Avg Holding Days
            avg_holding = metrics.get("avg_holding_days", 0.0)
            if avg_holding is not None and not pd.isna(avg_holding):
                holding_item = QTableWidgetItem(f"{avg_holding:.1f}")
                holding_item.setData(Qt.ItemDataRole.EditRole, float(avg_holding))
            else:
                holding_item = QTableWidgetItem("N/A")
                holding_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 11, holding_item)
            
            # Annual Return
            annual_ret = metrics.get("annual_return", 0.0)
            if annual_ret is not None and not pd.isna(annual_ret):
                annual_item = QTableWidgetItem(f"{annual_ret:.2%}")
                annual_item.setData(Qt.ItemDataRole.EditRole, float(annual_ret))
            else:
                annual_item = QTableWidgetItem("N/A")
                annual_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.backtests_table.setItem(row_idx, 12, annual_item)
            
            # Date Range
            date_range = metrics.get("date_range", "N/A")
            range_item = QTableWidgetItem(str(date_range))
            self.backtests_table.setItem(row_idx, 13, range_item)
            
            # Filters Used
            metadata = backtest.get("metadata")
            if metadata and metadata.get("filters_applied"):
                filters_count = len(metadata.get("filters_applied", []))
                preset_name = metadata.get("filter_preset_name")
                if preset_name:
                    filter_text = f"{preset_name} ({filters_count})"
                else:
                    filter_text = f"{filters_count} filters"
                filter_item = QTableWidgetItem(filter_text)
                filter_item.setToolTip(f"Click to view filter details\nPreset: {preset_name or 'None'}\nFilters: {filters_count}")
                # Store metadata for click handler
                filter_item.setData(Qt.ItemDataRole.UserRole, metadata)
            else:
                filter_item = QTableWidgetItem("None")
            self.backtests_table.setItem(row_idx, 14, filter_item)
        
        # Make filter column clickable (disconnect first to avoid duplicate connections)
        try:
            self.backtests_table.cellDoubleClicked.disconnect(self.on_filter_cell_clicked)
        except TypeError:
            pass  # Signal wasn't connected yet
        self.backtests_table.cellDoubleClicked.connect(self.on_filter_cell_clicked)
        
        self.backtests_table.setSortingEnabled(True)
        # Sort by date descending by default
        if len(self.backtest_data) > 0:
            self.backtests_table.sortItems(2, Qt.SortOrder.DescendingOrder)
    
    def toggle_backtest_selection(self, backtest_id: str, selected: bool):
        """Toggle backtest selection."""
        if selected and backtest_id not in self.selected_backtests:
            self.selected_backtests.append(backtest_id)
        elif not selected and backtest_id in self.selected_backtests:
            self.selected_backtests.remove(backtest_id)
        
        # Update selected label
        count = len(self.selected_backtests)
        if count > 0:
            self.selected_label.setText(f"{count} backtest(s) selected for comparison")
            self.selected_label.setStyleSheet("color: #00d4aa; font-weight: bold;")
        else:
            self.selected_label.setText("No backtests selected for comparison")
            self.selected_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
    
    def compare_selected_backtests(self):
        """Compare selected backtests side-by-side."""
        if len(self.selected_backtests) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least 2 backtests to compare.")
            return
        
        # Get selected backtest data
        selected_data = [bt for bt in self.backtest_data if bt.get("id") in self.selected_backtests]
        
        if len(selected_data) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least 2 backtests to compare.")
            return
        
        # Define metrics to compare
        metric_names = [
            "Total Trades", "Win Rate", "Avg Return", "Total P&L", "Avg P&L",
            "Max Drawdown", "Sharpe Ratio", "Profit Factor", "Avg Hold Days",
            "Annual Return", "Date Range"
        ]
        
        metric_keys = [
            "n_trades", "win_rate", "avg_return", "total_pnl", "avg_pnl",
            "max_drawdown", "sharpe_ratio", "profit_factor", "avg_holding_days",
            "annual_return", "date_range"
        ]
        
        # Set up comparison table
        self.comparison_table.setColumnCount(len(selected_data))
        self.comparison_table.setRowCount(len(metric_names))
        
        # Set headers (use filenames)
        headers = [bt.get("filename", "Unknown") for bt in selected_data]
        self.comparison_table.setHorizontalHeaderLabels(headers)
        
        # Set row labels
        self.comparison_table.setVerticalHeaderLabels(metric_names)
        
        # Populate table
        for col_idx, backtest in enumerate(selected_data):
            metrics = backtest.get("metrics", {})
            
            for row_idx, (metric_name, metric_key) in enumerate(zip(metric_names, metric_keys)):
                value = metrics.get(metric_key, "N/A")
                
                # Format the value based on type
                if isinstance(value, float):
                    if "rate" in metric_key or "return" in metric_key:
                        display_value = f"{value:.2%}"
                    elif "pnl" in metric_key or "drawdown" in metric_key:
                        display_value = f"${value:,.2f}"
                    elif "ratio" in metric_key or "factor" in metric_key:
                        display_value = f"{value:.2f}"
                    elif "days" in metric_key:
                        display_value = f"{value:.1f}"
                    else:
                        display_value = f"{value:.2f}"
                elif isinstance(value, int):
                    display_value = str(value)
                else:
                    display_value = str(value)
                
                item = QTableWidgetItem(display_value)
                if isinstance(value, (int, float)):
                    item.setData(Qt.ItemDataRole.EditRole, value)
                self.comparison_table.setItem(row_idx, col_idx, item)
    
    def delete_selected_backtests(self):
        """Delete selected backtest CSV files."""
        if not self.selected_backtests:
            QMessageBox.warning(self, "No Selection", "Please select backtests to delete by checking the boxes in the 'Select' column.")
            return
        
        # Get filenames for confirmation dialog
        selected_filenames = []
        for backtest in self.backtest_data:
            if backtest.get("id") in self.selected_backtests:
                selected_filenames.append(backtest.get("filename", "Unknown"))
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(selected_filenames)} backtest file(s)?\n\n"
            f"Files to delete:\n" + "\n".join(f"  • {name}" for name in selected_filenames[:10]) +
            (f"\n  ... and {len(selected_filenames) - 10} more" if len(selected_filenames) > 10 else ""),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Delete the files
        deleted_count = 0
        failed_count = 0
        failed_files = []
        
        for backtest in self.backtest_data:
            if backtest.get("id") in self.selected_backtests:
                file_path = Path(backtest.get("file_path", ""))
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        failed_count += 1
                        failed_files.append(backtest.get("filename", "Unknown"))
        
        # Clear selection
        self.selected_backtests = []
        
        # Refresh the list
        self.refresh_backtest_list()
        
        # Show result message
        if failed_count == 0:
            QMessageBox.information(
                self,
                "Deletion Complete",
                f"Successfully deleted {deleted_count} backtest file(s)."
            )
        else:
            QMessageBox.warning(
                self,
                "Deletion Partial",
                f"Deleted {deleted_count} file(s), but {failed_count} file(s) could not be deleted:\n\n"
                + "\n".join(f"  • {name}" for name in failed_files)
            )
    
    def on_filter_cell_clicked(self, row: int, col: int):
        """Handle double-click on filter cell to show filter details."""
        if col != 14:  # Filters Used column
            return
        
        item = self.backtests_table.item(row, col)
        if not item:
            return
        
        metadata = item.data(Qt.ItemDataRole.UserRole)
        if not metadata or not metadata.get("filters_applied"):
            QMessageBox.information(self, "No Filters", "No filters were applied to this backtest.")
            return
        
        # Create filter details dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Details")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        # Preset info
        preset_name = metadata.get("filter_preset_name")
        if preset_name:
            preset_label = QLabel(f"<b>Preset:</b> {preset_name}")
            preset_label.setStyleSheet("color: #00d4aa; font-size: 14px; margin-bottom: 10px;")
            layout.addWidget(preset_label)
        
        # Filters list
        filters_label = QLabel(f"<b>Filters Applied ({len(metadata.get('filters_applied', []))}):</b>")
        filters_label.setStyleSheet("color: #ffffff; font-size: 12px; margin-top: 10px;")
        layout.addWidget(filters_label)
        
        # Scrollable text area with filters
        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setStyleSheet("background-color: #2d2d2d; color: #ffffff; border: 1px solid #555;")
        
        filters_text = ""
        for i, filter_dict in enumerate(metadata.get("filters_applied", []), 1):
            feature = filter_dict.get("feature", "Unknown")
            operator = filter_dict.get("operator", "")
            value = filter_dict.get("value", "")
            filters_text += f"{i}. {feature} {operator} {value}\n"
        
        text_area.setPlainText(filters_text)
        layout.addWidget(text_area)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: dialog.done(QDialog.DialogCode.Accepted))
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()  # exec() returns when dialog is closed
    
    def show_metrics_help(self):
        """Show help dialog for backtest metrics."""
        dialog = MetricsHelpDialog(self)
        dialog.exec()

