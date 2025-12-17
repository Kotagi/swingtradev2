"""
Model Comparison Tab

Allows users to view and compare multiple trained models side-by-side.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QMessageBox, QScrollArea, QCheckBox, QLineEdit, QComboBox,
    QDialog, QTextEdit, QInputDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPalette
from pathlib import Path
from datetime import datetime
import joblib
import yaml

from gui.utils.model_registry import ModelRegistry
from gui.tabs.metrics_help_dialog import MetricsHelpDialog


class ModelComparisonTab(QWidget):
    """Tab for comparing trained models."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.registry = ModelRegistry()
        self.selected_models = []  # List of model IDs for comparison
        self.init_ui()
        self.refresh_model_list()
    
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
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget
        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title row with help button
        title_row = QHBoxLayout()
        title = QLabel("Model Comparison")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        title_row.addWidget(title)
        title_row.addStretch()
        
        # Help button - using QLabel styled as button for better text visibility
        help_btn_container = QWidget()
        help_btn_container.setFixedSize(35, 35)
        help_btn_container.setToolTip("Click for help on understanding model metrics")
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
        self.search_edit.setPlaceholderText("Search by name...")
        self.search_edit.textChanged.connect(self.refresh_model_list)
        filter_layout.addWidget(self.search_edit)
        
        filter_layout.addWidget(QLabel("Min ROC AUC:"))
        self.min_auc_combo = QComboBox()
        self.min_auc_combo.addItems(["Any", "0.5", "0.6", "0.7", "0.8", "0.9"])
        self.min_auc_combo.currentTextChanged.connect(self.refresh_model_list)
        filter_layout.addWidget(self.min_auc_combo)
        
        filter_layout.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_model_list)
        filter_layout.addWidget(refresh_btn)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Model list table
        list_label = QLabel("All Models")
        list_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(list_label)
        
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(17)  # Added Features column
        # Abbreviated headers (tooltips not supported per-section in QHeaderView)
        self.models_table.setHorizontalHeaderLabels([
            "Select", "Name", "Date", "ROC AUC", "Accuracy", 
            "Precision", "Recall", "F1", "Avg Prec",
            "Feature Set", "Features", "Horizon", "Imbalance", "Tuned", "CV", "Iterations", "CV Folds"
        ])
        
        # Set Select column to fixed width (just for checkbox)
        self.models_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.models_table.setColumnWidth(0, 50)
        # Use ResizeToContents for better column sizing, with minimum widths
        for col in range(1, 17):
            self.models_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
            # Set minimum widths for important columns
            if col == 1:  # Name
                self.models_table.setColumnWidth(col, 150)
            elif col == 2:  # Date
                self.models_table.setColumnWidth(col, 120)
            elif col in [3, 4, 5, 6, 7, 8]:  # Metrics
                self.models_table.setColumnWidth(col, 80)
            elif col == 10:  # Features
                self.models_table.setColumnWidth(col, 80)
            else:
                self.models_table.setColumnWidth(col, 70)
        
        # Enable horizontal scrolling
        self.models_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.models_table.setAlternatingRowColors(True)
        self.models_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.models_table.setMinimumHeight(400)
        layout.addWidget(self.models_table)
        
        # Comparison section
        comparison_group = QGroupBox("Comparison")
        comparison_layout = QVBoxLayout()
        
        # Selected models info
        self.selected_label = QLabel("No models selected for comparison")
        self.selected_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        comparison_layout.addWidget(self.selected_label)
        
        # Comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(0)  # Will be set dynamically
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setMinimumHeight(300)
        comparison_layout.addWidget(self.comparison_table)
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        compare_btn = QPushButton("Compare Selected")
        compare_btn.clicked.connect(self.compare_selected_models)
        action_layout.addWidget(compare_btn)
        
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self.clear_selection)
        action_layout.addWidget(clear_btn)
        
        rename_btn = QPushButton("Rename Selected")
        # Match the height of other buttons
        rename_btn.setFixedHeight(compare_btn.sizeHint().height())
        rename_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                color: white;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #ff9800;
                color: #000000;
            }
        """)
        rename_btn.clicked.connect(self.rename_selected_model)
        action_layout.addWidget(rename_btn)
        
        delete_btn = QPushButton("Delete Selected")
        delete_btn.setStyleSheet("background-color: #f44336; color: white;")
        delete_btn.clicked.connect(self.delete_selected_models)
        action_layout.addWidget(delete_btn)
        
        comparison_layout.addLayout(action_layout)
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
    
    def refresh_model_list(self):
        """Refresh the model list table."""
        # Reload registry from disk to get latest models
        self.registry._registry = self.registry._load_registry()
        
        # Get filter criteria
        name_filter = self.search_edit.text().strip()
        min_auc_text = self.min_auc_combo.currentText()
        min_auc = float(min_auc_text) if min_auc_text != "Any" else None
        
        # Get filtered models
        models = self.registry.search_models(
            name_filter=name_filter if name_filter else None,
            min_roc_auc=min_auc
        )
        
        # Populate table
        self.models_table.setRowCount(len(models))
        self.models_table.setSortingEnabled(False)
        
        for row_idx, model in enumerate(models):
            # Select checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(model.get("id") in self.selected_models)
            checkbox.stateChanged.connect(
                lambda state, mid=model.get("id"): self.toggle_model_selection(mid, state == Qt.CheckState.Checked.value)
            )
            # Center the checkbox in the cell
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.models_table.setCellWidget(row_idx, 0, checkbox_widget)
            
            # Name
            name_item = QTableWidgetItem(model.get("name", "Unknown"))
            self.models_table.setItem(row_idx, 1, name_item)
            
            # Date
            training_date = model.get("training_date", "")
            if training_date:
                try:
                    dt = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = training_date
            else:
                date_str = "Unknown"
            date_item = QTableWidgetItem(date_str)
            self.models_table.setItem(row_idx, 2, date_item)
            
            # Test ROC AUC
            test_metrics = model.get("metrics", {}).get("test", {})
            roc_auc = test_metrics.get("roc_auc", 0.0)
            roc_item = QTableWidgetItem(f"{roc_auc:.4f}" if roc_auc > 0 else "N/A")
            roc_item.setData(Qt.ItemDataRole.EditRole, roc_auc)  # For sorting
            if roc_auc > 0.8:
                roc_item.setForeground(Qt.GlobalColor.green)
            elif roc_auc > 0.6:
                roc_item.setForeground(Qt.GlobalColor.yellow)
            self.models_table.setItem(row_idx, 3, roc_item)
            
            # Test Accuracy
            accuracy = test_metrics.get("accuracy", 0.0)
            acc_item = QTableWidgetItem(f"{accuracy:.4f}" if accuracy > 0 else "N/A")
            acc_item.setData(Qt.ItemDataRole.EditRole, accuracy)
            self.models_table.setItem(row_idx, 4, acc_item)
            
            # Test Precision
            precision = test_metrics.get("precision", 0.0)
            precision_item = QTableWidgetItem(f"{precision:.4f}" if precision > 0 else "N/A")
            precision_item.setData(Qt.ItemDataRole.EditRole, precision)
            self.models_table.setItem(row_idx, 5, precision_item)
            
            # Test Recall
            recall = test_metrics.get("recall", 0.0)
            recall_item = QTableWidgetItem(f"{recall:.4f}" if recall > 0 else "N/A")
            recall_item.setData(Qt.ItemDataRole.EditRole, recall)
            self.models_table.setItem(row_idx, 6, recall_item)
            
            # Test F1
            f1 = test_metrics.get("f1_score", 0.0)
            f1_item = QTableWidgetItem(f"{f1:.4f}" if f1 > 0 else "N/A")
            f1_item.setData(Qt.ItemDataRole.EditRole, f1)
            self.models_table.setItem(row_idx, 7, f1_item)
            
            # Test Average Precision
            avg_precision = test_metrics.get("average_precision", 0.0)
            ap_item = QTableWidgetItem(f"{avg_precision:.4f}" if avg_precision > 0 else "N/A")
            ap_item.setData(Qt.ItemDataRole.EditRole, avg_precision)
            self.models_table.setItem(row_idx, 8, ap_item)
            
            # Feature Set
            feature_set = model.get("parameters", {}).get("feature_set", "default")
            feature_item = QTableWidgetItem(feature_set or "default")
            self.models_table.setItem(row_idx, 9, feature_item)
            
            # Features (enabled/total)
            feature_count_str = self._get_feature_count_display(model)
            features_item = QTableWidgetItem(feature_count_str)
            self.models_table.setItem(row_idx, 10, features_item)
            
            # Horizon (extract from parameters or label_col)
            horizon = model.get("parameters", {}).get("horizon", "N/A")
            horizon_item = QTableWidgetItem(str(horizon) if horizon else "N/A")
            self.models_table.setItem(row_idx, 11, horizon_item)
            
            # Class Imbalance Multiplier
            imbalance = model.get("parameters", {}).get("imbalance_multiplier", "N/A")
            if imbalance is not None and imbalance != "N/A":
                imbalance_item = QTableWidgetItem(f"{imbalance:.2f}")
                imbalance_item.setData(Qt.ItemDataRole.EditRole, float(imbalance))  # For sorting
            else:
                imbalance_item = QTableWidgetItem("N/A")
                imbalance_item.setData(Qt.ItemDataRole.EditRole, 0.0)
            self.models_table.setItem(row_idx, 12, imbalance_item)
            
            # Tuned (Yes/No)
            tuned = model.get("parameters", {}).get("tune", False)
            tuned_item = QTableWidgetItem("Yes" if tuned else "No")
            tuned_item.setData(Qt.ItemDataRole.EditRole, 1 if tuned else 0)  # For sorting
            self.models_table.setItem(row_idx, 13, tuned_item)
            
            # CV (Yes/No)
            cv = model.get("parameters", {}).get("cv", False)
            cv_item = QTableWidgetItem("Yes" if cv else "No")
            cv_item.setData(Qt.ItemDataRole.EditRole, 1 if cv else 0)  # For sorting
            self.models_table.setItem(row_idx, 14, cv_item)
            
            # Iterations (if tuned)
            n_iter = model.get("parameters", {}).get("n_iter")
            if n_iter is not None and tuned:
                iter_item = QTableWidgetItem(str(n_iter))
                iter_item.setData(Qt.ItemDataRole.EditRole, int(n_iter))  # For sorting
            else:
                iter_item = QTableWidgetItem("N/A")
                iter_item.setData(Qt.ItemDataRole.EditRole, 0)  # For sorting
            self.models_table.setItem(row_idx, 15, iter_item)
            
            # CV Folds (if CV)
            cv_folds = model.get("parameters", {}).get("cv_folds")
            if cv_folds is not None and cv:
                folds_item = QTableWidgetItem(str(cv_folds))
                folds_item.setData(Qt.ItemDataRole.EditRole, int(cv_folds))  # For sorting
            else:
                folds_item = QTableWidgetItem("N/A")
                folds_item.setData(Qt.ItemDataRole.EditRole, 0)  # For sorting
            self.models_table.setItem(row_idx, 16, folds_item)
        
        self.models_table.setSortingEnabled(True)
        # Sort by date descending by default
        self.models_table.sortItems(2, Qt.SortOrder.DescendingOrder)
    
    def toggle_model_selection(self, model_id: str, selected: bool):
        """Toggle model selection."""
        if selected and model_id not in self.selected_models:
            self.selected_models.append(model_id)
        elif not selected and model_id in self.selected_models:
            self.selected_models.remove(model_id)
        self.update_selected_label()
    
    def clear_selection(self):
        """Clear all model selections."""
        self.selected_models.clear()
        self.refresh_model_list()
        self.update_selected_label()
        self.comparison_table.setRowCount(0)
        self.comparison_table.setColumnCount(0)
    
    def update_selected_label(self):
        """Update the selected models label."""
        count = len(self.selected_models)
        if count == 0:
            self.selected_label.setText("No models selected for comparison")
        else:
            self.selected_label.setText(f"{count} model(s) selected for comparison")
    
    def compare_selected_models(self):
        """Compare selected models side-by-side."""
        if len(self.selected_models) < 2:
            QMessageBox.warning(self, "Not Enough Models", "Please select at least 2 models to compare.")
            return
        
        # Get model data
        models = []
        for model_id in self.selected_models:
            model = self.registry.get_model(model_id)
            if model:
                models.append(model)
        
        if len(models) < 2:
            QMessageBox.warning(self, "Error", "Could not load all selected models.")
            return
        
        # Prepare comparison data
        # Metrics to compare
        metric_names = [
            "ROC AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Average Precision"
        ]
        
        # Set up comparison table
        self.comparison_table.setColumnCount(len(models) + 1)  # +1 for metric names
        headers = ["Metric"] + [m.get("name", f"Model {i+1}") for i, m in enumerate(models)]
        self.comparison_table.setHorizontalHeaderLabels(headers)
        self.comparison_table.setRowCount(len(metric_names))
        
        # Populate comparison table
        for row_idx, metric_name in enumerate(metric_names):
            # Metric name
            metric_item = QTableWidgetItem(metric_name)
            metric_item.setFlags(metric_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.comparison_table.setItem(row_idx, 0, metric_item)
            
            # Values for each model
            for col_idx, model in enumerate(models):
                test_metrics = model.get("metrics", {}).get("test", {})
                
                # Map metric names to keys
                metric_key_map = {
                    "ROC AUC": "roc_auc",
                    "Accuracy": "accuracy",
                    "Precision": "precision",
                    "Recall": "recall",
                    "F1 Score": "f1_score",
                    "Average Precision": "average_precision"
                }
                
                metric_key = metric_key_map.get(metric_name)
                value = test_metrics.get(metric_key, 0.0) if metric_key else 0.0
                
                value_item = QTableWidgetItem(f"{value:.4f}" if value > 0 else "N/A")
                value_item.setData(Qt.ItemDataRole.EditRole, value)  # For sorting
                
                # Color code based on value
                if metric_name == "ROC AUC" and value > 0.8:
                    value_item.setForeground(Qt.GlobalColor.green)
                elif metric_name == "ROC AUC" and value > 0.6:
                    value_item.setForeground(Qt.GlobalColor.yellow)
                
                self.comparison_table.setItem(row_idx, col_idx + 1, value_item)
        
        # Highlight best values in each row
        for row_idx in range(len(metric_names)):
            best_value = -1
            best_col = -1
            for col_idx in range(1, len(models) + 1):
                item = self.comparison_table.item(row_idx, col_idx)
                if item:
                    value = item.data(Qt.ItemDataRole.EditRole)
                    if isinstance(value, (int, float)) and value > best_value:
                        best_value = value
                        best_col = col_idx
            
            # Highlight best value
            if best_col > 0:
                best_item = self.comparison_table.item(row_idx, best_col)
                if best_item:
                    best_item.setBackground(Qt.GlobalColor.darkGreen)
                    best_item.setForeground(Qt.GlobalColor.white)
    
    def rename_selected_model(self):
        """Rename a selected model."""
        if not self.selected_models:
            QMessageBox.warning(self, "No Selection", "Please select a model to rename.")
            return
        
        if len(self.selected_models) > 1:
            QMessageBox.warning(self, "Multiple Selection", "Please select only one model to rename.")
            return
        
        # Get the selected model
        model_id = self.selected_models[0]
        model = self.registry.get_model(model_id)
        if not model:
            QMessageBox.warning(self, "Error", "Could not find selected model.")
            return
        
        # Get current name
        current_name = model.get("name", "Unknown")
        
        # Get new name from user
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Model",
            "Enter new name:",
            text=current_name
        )
        
        if not ok or not new_name.strip():
            return
        
        new_name = new_name.strip()
        
        # Check if name already exists (optional - could allow duplicates)
        # For now, we'll allow duplicates
        
        # Update model name
        if self.registry.update_model(model_id, {"name": new_name}):
            QMessageBox.information(self, "Success", f"Model renamed to '{new_name}'.")
            self.refresh_model_list()
            self.update_selected_label()
        else:
            QMessageBox.warning(self, "Error", "Failed to rename model.")
    
    def delete_selected_models(self):
        """Delete selected models from registry."""
        if not self.selected_models:
            QMessageBox.warning(self, "No Selection", "Please select models to delete.")
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {len(self.selected_models)} model(s)?\n\n"
            "Note: This only removes them from the registry, not the model files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            deleted_count = 0
            for model_id in self.selected_models[:]:  # Copy list to avoid modification during iteration
                if self.registry.delete_model(model_id):
                    deleted_count += 1
                    self.selected_models.remove(model_id)
            
            QMessageBox.information(self, "Deleted", f"Deleted {deleted_count} model(s) from registry.")
            self.refresh_model_list()
            self.update_selected_label()
            self.comparison_table.setRowCount(0)
            self.comparison_table.setColumnCount(0)
    
    def show_metrics_help(self):
        """Show help dialog explaining model metrics."""
        dialog = MetricsHelpDialog(self)
        dialog.exec()
    
    def _get_feature_count_display(self, model: dict) -> str:
        """
        Get feature count display string (enabled/total).
        Tries to extract from registry, then from model file, then shows N/A.
        """
        # Try to get from training_info first
        training_info = model.get("training_info", {})
        feature_count = training_info.get("feature_count")
        total_features = training_info.get("total_features")
        
        # If we have both, return formatted string
        if feature_count is not None and total_features is not None:
            return f"{feature_count}/{total_features}"
        
        # Fallback: try to extract from model file
        model_path = model.get("file_path")
        if model_path:
            try:
                model_path_obj = Path(model_path)
                if model_path_obj.exists():
                    # Load the model pickle
                    model_data = joblib.load(model_path_obj)
                    # Model is saved as dict with "features" list and "metadata" dict
                    if isinstance(model_data, dict):
                        # Try to get from features list
                        features_list = model_data.get("features", [])
                        if features_list:
                            enabled_count = len(features_list)
                            # Get total from config/features.yaml
                            total = self._count_total_features()
                            return f"{enabled_count}/{total}"
                        # Try to get from metadata
                        metadata = model_data.get("metadata", {})
                        n_features = metadata.get("n_features")
                        if n_features is not None:
                            total = self._count_total_features()
                            return f"{n_features}/{total}"
                    # Fallback: try XGBoost model directly (if model_data is the model itself)
                    elif hasattr(model_data, 'feature_names_in_'):
                        enabled_count = len(model_data.feature_names_in_)
                        total = self._count_total_features()
                        return f"{enabled_count}/{total}"
            except Exception:
                pass  # Fall through to N/A
        
        # If all else fails, return N/A
        return "N/A"
    
    def _count_total_features(self) -> int:
        """Count total available features from config/features.yaml."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "features.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                features = cfg.get("features", {})
                # Count all features (enabled or disabled, excluding comments)
                return len([k for k in features.keys() if not k.startswith('#')])
            return 57  # Default fallback (current total)
        except Exception:
            return 57  # Default fallback

