"""
Preset Comparison Dialog

Dialog for comparing multiple filter presets side-by-side.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QMessageBox, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt
from typing import List, Dict, Any, Tuple
import pandas as pd

from gui.services import FilterEditorService


class PresetComparisonDialog(QDialog):
    """Dialog for comparing filter presets."""
    
    def __init__(self, service: FilterEditorService, trades_df: pd.DataFrame,
                 features_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.service = service
        self.trades_df = trades_df
        self.features_df = features_df
        self.presets = []
        self.comparison_results = {}
        self.setWindowTitle("Compare Filter Presets")
        self.setMinimumSize(1000, 700)
        self.init_ui()
        self.load_presets()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Compare Filter Presets")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4aa;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Select 2-4 presets to compare. Each preset will be tested on the loaded backtest."
        )
        instructions.setStyleSheet("color: #b0b0b0;")
        layout.addWidget(instructions)
        
        # Preset selection
        selection_group = QGroupBox("Select Presets to Compare")
        selection_layout = QVBoxLayout()
        
        # Scroll area for preset checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll_widget = QWidget()
        self.preset_layout = QVBoxLayout()
        scroll_widget.setLayout(self.preset_layout)
        scroll.setWidget(scroll_widget)
        selection_layout.addWidget(scroll)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Comparison table
        comparison_group = QGroupBox("Comparison Results")
        comparison_layout = QVBoxLayout()
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(0)
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        comparison_layout.addWidget(self.comparison_table)
        
        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        compare_btn = QPushButton("Compare Selected")
        compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #000000;
                font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #00c896;
            }
        """)
        compare_btn.clicked.connect(self.run_comparison)
        button_layout.addWidget(compare_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_presets(self):
        """Load all available presets."""
        self.presets = self.service.list_filter_presets()
        
        # Clear existing checkboxes
        while self.preset_layout.count():
            child = self.preset_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.presets:
            no_presets_label = QLabel("No filter presets found. Create some in the Filter Editor tab.")
            no_presets_label.setStyleSheet("color: #808080; font-style: italic;")
            self.preset_layout.addWidget(no_presets_label)
            return
        
        # Create checkboxes for each preset
        for preset in self.presets:
            checkbox = QCheckBox(preset.get("name", preset.get("filename", "Unknown")))
            checkbox.setData(preset)  # Store preset data
            self.preset_layout.addWidget(checkbox)
        
        self.preset_layout.addStretch()
    
    def get_selected_presets(self) -> List[Dict[str, Any]]:
        """Get list of selected presets."""
        selected = []
        for i in range(self.preset_layout.count()):
            item = self.preset_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QCheckBox) and widget.isChecked():
                    selected.append(widget.data())
        return selected
    
    def run_comparison(self):
        """Run comparison on selected presets."""
        selected = self.get_selected_presets()
        
        if len(selected) < 2:
            QMessageBox.warning(self, "Invalid Selection", "Please select at least 2 presets to compare.")
            return
        
        if len(selected) > 4:
            QMessageBox.warning(self, "Too Many Presets", "Please select at most 4 presets to compare.")
            return
        
        # Calculate impact for each preset
        self.comparison_results = {}
        
        for preset in selected:
            preset_name = preset.get("name", preset.get("filename", "Unknown"))
            filters = preset.get("filters", [])
            
            if not filters:
                continue
            
            # Convert filters to tuples
            filter_tuples = [
                (f["feature"], f["operator"], f["value"])
                for f in filters
            ]
            
            # Calculate impact
            try:
                impact_data = self.service.calculate_filter_impact(
                    self.trades_df,
                    self.features_df,
                    filter_tuples
                )
                self.comparison_results[preset_name] = impact_data
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to calculate impact for '{preset_name}': {str(e)}")
                continue
        
        # Update comparison table
        self.update_comparison_table()
    
    def update_comparison_table(self):
        """Update the comparison table with results."""
        if not self.comparison_results:
            self.comparison_table.setRowCount(0)
            self.comparison_table.setColumnCount(0)
            return
        
        # Define metrics to compare
        metrics = [
            ("Metric", "str"),
            ("Filters", "int"),
            ("Total Trades", "int"),
            ("Winners Excluded", "int"),
            ("Stop-Losses Excluded", "int"),
            ("Win Rate", "pct"),
            ("Total P&L", "currency"),
            ("Avg P&L/Trade", "currency"),
            ("Sharpe Ratio", "float"),
            ("Max Drawdown", "currency"),
            ("Profit Factor", "float"),
            ("Annual Return", "pct")
        ]
        
        preset_names = list(self.comparison_results.keys())
        
        # Set up table
        self.comparison_table.setColumnCount(len(preset_names) + 1)  # +1 for metric column
        self.comparison_table.setRowCount(len(metrics))
        
        # Set headers
        headers = ["Metric"] + preset_names
        self.comparison_table.setHorizontalHeaderLabels(headers)
        
        # Populate data
        for row_idx, (metric_name, metric_type) in enumerate(metrics):
            # Metric name
            metric_item = QTableWidgetItem(metric_name)
            metric_item.setFlags(metric_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.comparison_table.setItem(row_idx, 0, metric_item)
            
            # Values for each preset
            for col_idx, preset_name in enumerate(preset_names, start=1):
                impact_data = self.comparison_results[preset_name]
                combined = impact_data.get("combined", {})
                after_metrics = impact_data.get("after_metrics", {})
                
                # Get value based on metric
                if metric_name == "Filters":
                    value = len(impact_data.get("per_feature", {}))
                elif metric_name == "Total Trades":
                    value = after_metrics.get("n_trades", 0)
                elif metric_name == "Winners Excluded":
                    value = combined.get("winners_excluded", 0)
                elif metric_name == "Stop-Losses Excluded":
                    value = combined.get("stop_losses_excluded", 0)
                elif metric_name == "Win Rate":
                    value = after_metrics.get("win_rate", 0) * 100
                elif metric_name == "Total P&L":
                    value = after_metrics.get("total_pnl", 0)
                elif metric_name == "Avg P&L/Trade":
                    value = after_metrics.get("avg_pnl", 0)
                elif metric_name == "Sharpe Ratio":
                    value = after_metrics.get("sharpe_ratio", 0)
                elif metric_name == "Max Drawdown":
                    value = after_metrics.get("max_drawdown", 0)
                elif metric_name == "Profit Factor":
                    value = after_metrics.get("profit_factor", 0)
                elif metric_name == "Annual Return":
                    value = after_metrics.get("annual_return", 0) * 100
                else:
                    value = 0
                
                # Format value
                if metric_type == "pct":
                    display_text = f"{value:.2f}%"
                elif metric_type == "currency":
                    display_text = f"${value:,.2f}"
                elif metric_type == "float":
                    display_text = f"{value:.3f}"
                elif metric_type == "int":
                    display_text = f"{int(value)}"
                else:
                    display_text = str(value)
                
                item = QTableWidgetItem(display_text)
                item.setData(Qt.ItemDataRole.EditRole, value)  # For sorting
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.comparison_table.setItem(row_idx, col_idx, item)
        
        # Resize columns
        self.comparison_table.resizeColumnsToContents()
        self.comparison_table.setSortingEnabled(True)
        
        # Highlight best values (optional - can be enhanced)
        self.highlight_best_values()
    
    def highlight_best_values(self):
        """Highlight best values in the comparison table."""
        # This is a simple implementation - can be enhanced
        # For now, just ensure the table is readable
        pass

