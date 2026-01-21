"""
Parquet Inspector Tab

File browser and inspector for Parquet files in the data directory.
Allows browsing, viewing metadata, and exporting to CSV for Excel viewing.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget,
    QTreeWidgetItem, QSplitter, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QFileSystemWatcher
from pathlib import Path
import pandas as pd
import numpy as np
import os
import subprocess
import platform
import tempfile
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_EXPORT_FILE = PROJECT_ROOT / "data" / "inspect_parquet" / "temp_export.csv"


class ParquetInspectorTab(QWidget):
    """Tab for inspecting Parquet files in the data directory."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self.current_df = None
        self.file_watcher = QFileSystemWatcher()
        self.init_ui()
        self.setup_file_watcher()
    
    def init_ui(self):
        """Initialize the UI components."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(3)

        # Title with minimal margin
        title = QLabel("Data Inspector")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4aa; margin: 0px;")
        main_layout.addWidget(title)

        # Top row: browser + basic info
        top_split = QSplitter(Qt.Orientation.Horizontal)

        # File Browser panel
        browser_group = QGroupBox("File Browser")
        browser_group.setFlat(True)
        browser_layout = QVBoxLayout()
        browser_layout.setContentsMargins(4, 4, 4, 4)
        browser_layout.setSpacing(3)

        self.path_label = QLabel(f"Path: {DATA_DIR}")
        self.path_label.setStyleSheet("font-weight: bold; color: #666; margin: 0px;")
        browser_layout.addWidget(self.path_label)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabel("Files and Folders")
        self.file_tree.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.file_tree.itemExpanded.connect(self.on_item_expanded)
        self.file_tree.itemSelectionChanged.connect(self.on_item_selected)
        self.file_tree.setAlternatingRowColors(True)
        browser_layout.addWidget(self.file_tree)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_file_browser)
        browser_layout.addWidget(refresh_btn)

        browser_group.setLayout(browser_layout)
        top_split.addWidget(browser_group)

        # File Information panel
        info_group = QGroupBox("File Information")
        info_group.setFlat(True)
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(4, 4, 4, 4)
        info_layout.setSpacing(3)

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("font-weight: bold; color: #333; margin: 0px;")
        self.file_path_label.setWordWrap(True)
        info_layout.addWidget(self.file_path_label)

        self.basic_info_text = QTextEdit()
        self.basic_info_text.setReadOnly(True)
        # Slightly taller to avoid scrolling for short summaries
        self.basic_info_text.setMaximumHeight(100)
        info_layout.addWidget(QLabel("Basic Information:"))
        info_layout.addWidget(self.basic_info_text)

        self.trading_day_label = QLabel("")
        self.trading_day_label.setStyleSheet("font-weight: bold; color: #00d4aa; margin: 2px 0 0 0;")
        info_layout.addWidget(self.trading_day_label)

        self.inspect_btn = QPushButton("Inspect (Export to CSV & Open Excel)")
        self.inspect_btn.setEnabled(False)
        self.inspect_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.inspect_btn.clicked.connect(self.export_and_open)
        info_layout.addWidget(self.inspect_btn)

        info_group.setLayout(info_layout)
        top_split.addWidget(info_group)

        top_split.setSizes([320, 360])
        main_layout.addWidget(top_split)

        # Column info full width
        column_group = QGroupBox("Column Information")
        column_group.setFlat(True)
        column_layout = QVBoxLayout()
        column_layout.setContentsMargins(4, 4, 4, 4)
        column_layout.setSpacing(3)

        self.column_table = QTableWidget()
        self.column_table.setColumnCount(4)
        self.column_table.setHorizontalHeaderLabels(["Column", "Type", "Populated", "Null"])
        self.column_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.column_table.setAlternatingRowColors(True)
        column_layout.addWidget(self.column_table)

        column_group.setLayout(column_layout)
        main_layout.addWidget(column_group)

        # Preview full width
        preview_group = QGroupBox("Quick Preview (First 10 Rows)")
        preview_group.setFlat(True)
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(4, 4, 4, 4)
        preview_layout.setSpacing(3)

        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)

        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        self.setLayout(main_layout)

        # Load initial file browser
        self.refresh_file_browser()
    
    def setup_file_watcher(self):
        """Setup file system watcher for auto-refresh."""
        try:
            # Watch the data directory
            if DATA_DIR.exists():
                self.file_watcher.addPath(str(DATA_DIR))
                self.file_watcher.directoryChanged.connect(self.on_directory_changed)
        except Exception as e:
            print(f"Warning: Could not setup file watcher: {e}")
    
    def on_directory_changed(self, path):
        """Handle directory change event."""
        # Refresh browser after a short delay to avoid rapid refreshes
        QTimer.singleShot(1000, self.refresh_file_browser)
    
    def refresh_file_browser(self):
        """Refresh the file browser tree while preserving expansion/selection."""
        expanded_paths, selected_path = self._capture_tree_state()
        self.file_tree.clear()
        
        if not DATA_DIR.exists():
            item = QTreeWidgetItem(self.file_tree, ["Data directory not found"])
            return
        
        # Add root item
        root_item = QTreeWidgetItem(self.file_tree, [DATA_DIR.name])
        root_item.setData(0, Qt.ItemDataRole.UserRole, str(DATA_DIR))
        
        # Populate root directory
        self.populate_tree_item(root_item, DATA_DIR)
        
        # Expand root
        root_item.setExpanded(True)
        self.file_tree.expandItem(root_item)

        # Restore expansion/selection state
        self._restore_tree_state(expanded_paths, selected_path)
    
    def populate_tree_item(self, parent_item, directory_path):
        """Populate a tree item with folders and parquet files."""
        try:
            path = Path(directory_path)
            if not path.exists() or not path.is_dir():
                return
            
            # Get folders and parquet files
            items = []
            for item in sorted(path.iterdir()):
                if item.is_dir():
                    items.append((item, True))  # True = is directory
                elif item.suffix.lower() == '.parquet':
                    items.append((item, False))  # False = is file
            
            # Add to tree
            for item_path, is_dir in items:
                item_name = item_path.name
                tree_item = QTreeWidgetItem(parent_item, [item_name])
                tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item_path))
                
                if is_dir:
                    # Add placeholder to allow expansion
                    placeholder = QTreeWidgetItem(tree_item, ["..."])
                    tree_item.setExpanded(False)
        except PermissionError:
            pass
        except Exception as e:
            print(f"Error populating tree item: {e}")

    def _capture_tree_state(self):
        """Capture expanded nodes and selected path."""
        expanded = set()
        selected = None

        def recurse(item):
            nonlocal selected
            path = item.data(0, Qt.ItemDataRole.UserRole)
            if item.isExpanded() and path:
                expanded.add(path)
            if item.isSelected() and path:
                selected = path
            for i in range(item.childCount()):
                recurse(item.child(i))

        root = self.file_tree.invisibleRootItem()
        for i in range(root.childCount()):
            recurse(root.child(i))
        return expanded, selected

    def _restore_tree_state(self, expanded_paths, selected_path):
        """Restore expanded nodes and selection."""
        if not expanded_paths and not selected_path:
            return

        def recurse(item):
            path = item.data(0, Qt.ItemDataRole.UserRole)
            if path in expanded_paths:
                item.setExpanded(True)
            if path == selected_path:
                item.setSelected(True)
                self.file_tree.scrollToItem(item)
            for i in range(item.childCount()):
                recurse(item.child(i))

        root = self.file_tree.invisibleRootItem()
        for i in range(root.childCount()):
            recurse(root.child(i))
    
    def on_item_double_clicked(self, item, column):
        """Handle double-click on tree item."""
        item_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_path:
            return
        
        path = Path(item_path)
        
        if path.is_dir():
            # Expand/collapse directory
            if item.isExpanded():
                item.setExpanded(False)
            else:
                # Load children if not already loaded
                if item.childCount() == 1 and item.child(0).text(0) == "...":
                    item.takeChild(0)  # Remove placeholder
                    self.populate_tree_item(item, path)
                elif item.childCount() == 0:
                    # No children yet, populate
                    self.populate_tree_item(item, path)
                item.setExpanded(True)
        elif path.suffix.lower() == '.parquet':
            # Load parquet file info
            self.load_parquet_file(path)

    def on_item_expanded(self, item):
        """Load children when a folder is expanded via the arrow."""
        item_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_path:
            return
        path = Path(item_path)
        if path.is_dir():
            # If placeholder exists, replace with actual children
            if item.childCount() == 1 and item.child(0).text(0) == "...":
                item.takeChild(0)
                self.populate_tree_item(item, path)
            elif item.childCount() == 0:
                self.populate_tree_item(item, path)
    
    def on_item_selected(self):
        """Handle selection change in tree."""
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        item_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not item_path:
            return
        
        path = Path(item_path)
        
        if path.is_file() and path.suffix.lower() == '.parquet':
            self.load_parquet_file(path)
    
    def load_parquet_file(self, file_path):
        """Load and display parquet file information."""
        try:
            self.current_file = file_path
            self.file_path_label.setText(f"File: {file_path.name}")
            self.file_path_label.setStyleSheet("font-weight: bold; color: #00d4aa;")
            
            # Update path label
            try:
                relative_path = file_path.relative_to(DATA_DIR)
                if relative_path.parent != Path('.'):
                    path_str = f"data/{relative_path.parent}"
                else:
                    path_str = "data"
            except ValueError:
                # File is outside data directory
                path_str = str(file_path.parent)
            self.path_label.setText(f"Path: {path_str}")
            
            # Load parquet file
            self.current_df = pd.read_parquet(file_path)
            
            # Display basic info
            self.display_basic_info()
            
            # Display column information
            self.display_column_info()
            
            # Display preview
            self.display_preview()
            
            # Find most recent trading day
            self.find_most_recent_trading_day()
            
            # Enable inspect button
            self.inspect_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load parquet file:\n{str(e)}")
            self.current_file = None
            self.current_df = None
            self.inspect_btn.setEnabled(False)
    
    def display_basic_info(self):
        """Display basic file information."""
        if self.current_df is None:
            return
        
        info_lines = [
            f"Rows: {len(self.current_df):,}",
            f"Columns: {len(self.current_df.columns)}",
            f"Index: {self.current_df.index.name or '(unnamed)'}",
        ]
        
        # Add date range if index is datetime
        if isinstance(self.current_df.index, pd.DatetimeIndex):
            info_lines.append(f"Date Range: {self.current_df.index.min()} to {self.current_df.index.max()}")
        
        self.basic_info_text.setPlainText("\n".join(info_lines))
    
    def display_column_info(self):
        """Display column information with populated and null counts."""
        if self.current_df is None:
            return
        
        self.column_table.setRowCount(len(self.current_df.columns))
        
        for idx, col in enumerate(self.current_df.columns):
            # Column name
            name_item = QTableWidgetItem(str(col))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.column_table.setItem(idx, 0, name_item)
            
            # Data type
            dtype_item = QTableWidgetItem(str(self.current_df[col].dtype))
            dtype_item.setFlags(dtype_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.column_table.setItem(idx, 1, dtype_item)
            
            # Populated count (non-null)
            non_null_count = self.current_df[col].notna().sum()
            populated_item = QTableWidgetItem(f"{non_null_count:,}")
            populated_item.setFlags(populated_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            populated_item.setForeground(Qt.GlobalColor.darkGreen)
            self.column_table.setItem(idx, 2, populated_item)
            
            # Null count
            null_count = self.current_df[col].isna().sum()
            null_item = QTableWidgetItem(f"{null_count:,}")
            null_item.setFlags(null_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if null_count > 0:
                null_item.setForeground(Qt.GlobalColor.red)
            self.column_table.setItem(idx, 3, null_item)
    
    def display_preview(self):
        """Display preview of first 10 rows."""
        if self.current_df is None or self.current_df.empty:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return
        
        # Get first 200 rows (table height unchanged)
        preview_df = self.current_df.head(200)
        
        # Set up table
        self.preview_table.setRowCount(len(preview_df))
        self.preview_table.setColumnCount(len(preview_df.columns) + 1)  # +1 for index
        
        # Set headers
        headers = ["Index"] + [str(col) for col in preview_df.columns]
        self.preview_table.setHorizontalHeaderLabels(headers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        # Populate table
        for row_idx, (idx_val, row) in enumerate(preview_df.iterrows()):
            # Index column
            idx_item = QTableWidgetItem(str(idx_val))
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.preview_table.setItem(row_idx, 0, idx_item)
            
            # Data columns
            for col_idx, col in enumerate(preview_df.columns):
                value = row[col]
                if pd.isna(value):
                    display_value = "NaN"
                else:
                    display_value = str(value)
                
                item = QTableWidgetItem(display_value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.preview_table.setItem(row_idx, col_idx + 1, item)
    
    def find_most_recent_trading_day(self):
        """Find and display the most recent trading day in the data."""
        if self.current_df is None or self.current_df.empty:
            self.trading_day_label.setText("")
            return
        
        most_recent = None
        
        # Check if index is datetime
        if isinstance(self.current_df.index, pd.DatetimeIndex):
            most_recent = self.current_df.index.max()
        
        # Check date columns
        date_columns = []
        for col in self.current_df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.current_df[col]):
                date_columns.append(col)
            elif self.current_df[col].dtype == 'object':
                # Try to parse as date
                try:
                    sample = self.current_df[col].dropna().iloc[0] if len(self.current_df[col].dropna()) > 0 else None
                    if sample and isinstance(sample, str):
                        pd.to_datetime(sample)
                        date_columns.append(col)
                except:
                    pass
        
        # Find max date in date columns
        for col in date_columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(self.current_df[col]):
                    col_max = self.current_df[col].max()
                else:
                    col_max = pd.to_datetime(self.current_df[col], errors='coerce').max()
                
                if pd.notna(col_max):
                    if most_recent is None or col_max > most_recent:
                        most_recent = col_max
            except:
                pass
        
        # Display result
        if most_recent is not None and pd.notna(most_recent):
            if isinstance(most_recent, pd.Timestamp):
                date_str = most_recent.strftime('%Y-%m-%d')
            else:
                date_str = str(most_recent)
            self.trading_day_label.setText(f"Most Recent Trading Day: {date_str}")
        else:
            self.trading_day_label.setText("Most Recent Trading Day: Not found")
    
    def export_and_open(self):
        """Export current parquet to CSV and open in Excel."""
        if self.current_df is None:
            QMessageBox.warning(self, "No Data", "No file loaded.")
            return
        
        try:
            # Ensure export directory exists
            export_dir = TEMP_EXPORT_FILE.parent
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV (overwrite existing)
            self.inspect_btn.setEnabled(False)
            self.inspect_btn.setText("Exporting...")
            self.inspect_btn.repaint()  # Update button text immediately
            
            self.current_df.to_csv(TEMP_EXPORT_FILE, index=True)
            
            # Open in Excel
            self.open_file_in_excel(TEMP_EXPORT_FILE)
            
            self.inspect_btn.setEnabled(True)
            self.inspect_btn.setText("Inspect (Export to CSV & Open Excel)")
            
            QMessageBox.information(
                self, 
                "Success", 
                f"CSV exported and opened in Excel:\n{TEMP_EXPORT_FILE}"
            )
            
        except Exception as e:
            self.inspect_btn.setEnabled(True)
            self.inspect_btn.setText("Inspect (Export to CSV & Open Excel)")
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")
    
    def open_file_in_excel(self, file_path):
        """Open a file in Excel (Windows)."""
        try:
            if platform.system() == 'Windows':
                os.startfile(str(file_path))
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', '-a', 'Microsoft Excel', str(file_path)])
            else:  # Linux
                subprocess.run(['xdg-open', str(file_path)])
        except Exception as e:
            QMessageBox.warning(
                self, 
                "Open Failed", 
                f"Could not open file automatically:\n{str(e)}\n\nFile saved to:\n{file_path}"
            )
