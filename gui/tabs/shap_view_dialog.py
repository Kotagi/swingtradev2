"""
SHAP View Dialog

Displays SHAP explanations for a single model or compares two models.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QScrollArea, QWidget,
    QGroupBox, QTextEdit, QMessageBox, QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from pathlib import Path
from typing import Dict, Optional, List, Any
import json

from gui.services import SHAPService


class SHAPViewDialog(QDialog):
    """Dialog for viewing SHAP explanations."""
    
    def __init__(self, model_id: str, model_name: str, shap_service: SHAPService, parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.model_name = model_name
        self.shap_service = shap_service
        self.setWindowTitle(f"SHAP Analysis: {model_name}")
        self.setMinimumSize(900, 700)
        
        self.init_ui()
        self.load_shap_data()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel(f"SHAP Explanations for: {self.model_name}")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        
        # Feature importance table
        importance_group = QGroupBox("Feature Importance Ranking")
        importance_layout = QVBoxLayout()
        
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(4)
        self.importance_table.setHorizontalHeaderLabels(["Rank", "Feature", "Importance", "Importance %"])
        self.importance_table.setAlternatingRowColors(True)
        self.importance_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.importance_table.horizontalHeader().setStretchLastSection(True)
        importance_layout.addWidget(self.importance_table)
        
        importance_group.setLayout(importance_layout)
        content_layout.addWidget(importance_group)
        
        # Summary statistics
        stats_group = QGroupBox("Summary Statistics")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        content_layout.addWidget(stats_group)
        
        # SHAP plot (if available)
        plot_group = QGroupBox("SHAP Summary Plot")
        plot_layout = QVBoxLayout()
        self.plot_label = QLabel("SHAP plot not available")
        self.plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_label.setMinimumHeight(400)
        plot_layout.addWidget(self.plot_label)
        plot_group.setLayout(plot_layout)
        content_layout.addWidget(plot_group)
        
        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_shap_data(self):
        """Load and display SHAP data."""
        artifacts = self.shap_service.load_artifacts(self.model_id)
        
        if not artifacts:
            self.stats_text.setText("SHAP artifacts not found for this model.")
            return
        
        # Display importance ranking
        ranking = artifacts.get("importance_ranking", [])
        if ranking:
            self.importance_table.setRowCount(len(ranking))
            for row_idx, item in enumerate(ranking):
                self.importance_table.setItem(row_idx, 0, QTableWidgetItem(str(item.get("rank", row_idx + 1))))
                self.importance_table.setItem(row_idx, 1, QTableWidgetItem(item.get("feature", "")))
                self.importance_table.setItem(row_idx, 2, QTableWidgetItem(f"{item.get('importance', 0):.6f}"))
                self.importance_table.setItem(row_idx, 3, QTableWidgetItem(f"{item.get('importance_pct', 0):.2f}%"))
            
            self.importance_table.resizeColumnsToContents()
        
        # Display summary statistics
        summary = artifacts.get("summary", {})
        stats = summary.get("statistics", {})
        metadata = artifacts.get("metadata", {})
        
        stats_text = f"""
<b>Computation Details:</b><br>
• Data Split: {metadata.get('data_split', 'Unknown')}<br>
• Sample Size: {metadata.get('sample_size', 'Unknown')} samples<br>
• Total Features: {metadata.get('n_features', 'Unknown')}<br>
• Computation Date: {metadata.get('computation_date', 'Unknown')[:10] if metadata.get('computation_date') else 'Unknown'}<br>
<br>
<b>Importance Statistics:</b><br>
• Total Importance: {stats.get('total_importance', 0):.6f}<br>
• Mean Importance: {stats.get('mean_importance', 0):.6f}<br>
• Max Importance: {stats.get('max_importance', 0):.6f}<br>
• Top 10 Concentration: {metadata.get('top_n_concentration', {}).get('percentage', 0):.1f}%<br>
"""
        self.stats_text.setHtml(stats_text)
        
        # Display plot if available
        plot_path = artifacts.get("plot_path")
        if plot_path and Path(plot_path).exists():
            pixmap = QPixmap(plot_path)
            if not pixmap.isNull():
                # Scale to fit while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(
                    self.plot_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.plot_label.setPixmap(scaled_pixmap)
                self.plot_label.setText("")
            else:
                self.plot_label.setText("Failed to load SHAP plot image")
        else:
            self.plot_label.setText("SHAP plot not available")


class SHAPComparisonDialog(QDialog):
    """Dialog for comparing SHAP explanations between two models."""
    
    def __init__(self, model1_id: str, model1_name: str, model2_id: str, model2_name: str,
                 shap_service: SHAPService, parent=None):
        super().__init__(parent)
        self.model1_id = model1_id
        self.model1_name = model1_name
        self.model2_id = model2_id
        self.model2_name = model2_name
        self.shap_service = shap_service
        self.setWindowTitle(f"SHAP Comparison: {model1_name} vs {model2_name}")
        self.setMinimumSize(1200, 800)
        
        self.init_ui()
        self.load_comparison_data()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel(f"SHAP Comparison: {self.model1_name} vs {self.model2_name}")
        header.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Splitter for side-by-side comparison
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Model 1 panel
        model1_widget = self._create_model_panel(self.model1_name, "model1")
        splitter.addWidget(model1_widget)
        
        # Model 2 panel
        model2_widget = self._create_model_panel(self.model2_name, "model2")
        splitter.addWidget(model2_widget)
        
        splitter.setSizes([600, 600])  # Equal widths
        layout.addWidget(splitter)
        
        # Comparison summary
        comparison_group = QGroupBox("Comparison Summary")
        comparison_layout = QVBoxLayout()
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        self.comparison_text.setMaximumHeight(150)
        comparison_layout.addWidget(self.comparison_text)
        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _create_model_panel(self, model_name: str, prefix: str) -> QWidget:
        """Create a panel for displaying one model's SHAP data."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Model name header
        header = QLabel(model_name)
        header.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(header)
        
        # Feature importance table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Rank", "Feature", "Importance", "Importance %"])
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        
        # Store reference for later use
        setattr(self, f"{prefix}_table", table)
        
        widget.setLayout(layout)
        return widget
    
    def load_comparison_data(self):
        """Load and display comparison data."""
        artifacts1 = self.shap_service.load_artifacts(self.model1_id)
        artifacts2 = self.shap_service.load_artifacts(self.model2_id)
        
        # Load model 1 data
        if artifacts1:
            ranking1 = artifacts1.get("importance_ranking", [])
            self._populate_table(self.model1_table, ranking1)
        
        # Load model 2 data
        if artifacts2:
            ranking2 = artifacts2.get("importance_ranking", [])
            self._populate_table(self.model2_table, ranking2)
        
        # Generate comparison summary
        if artifacts1 and artifacts2:
            self._generate_comparison_summary(artifacts1, artifacts2)
        else:
            missing = []
            if not artifacts1:
                missing.append(self.model1_name)
            if not artifacts2:
                missing.append(self.model2_name)
            self.comparison_text.setText(f"SHAP artifacts not found for: {', '.join(missing)}")
    
    def _populate_table(self, table: QTableWidget, ranking: List[Dict]):
        """Populate a table with ranking data."""
        table.setRowCount(len(ranking))
        for row_idx, item in enumerate(ranking):
            table.setItem(row_idx, 0, QTableWidgetItem(str(item.get("rank", row_idx + 1))))
            table.setItem(row_idx, 1, QTableWidgetItem(item.get("feature", "")))
            table.setItem(row_idx, 2, QTableWidgetItem(f"{item.get('importance', 0):.6f}"))
            table.setItem(row_idx, 3, QTableWidgetItem(f"{item.get('importance_pct', 0):.2f}%"))
        
        table.resizeColumnsToContents()
    
    def _generate_comparison_summary(self, artifacts1: Dict, artifacts2: Dict):
        """Generate comparison summary text."""
        ranking1 = {item["feature"]: item for item in artifacts1.get("importance_ranking", [])}
        ranking2 = {item["feature"]: item for item in artifacts2.get("importance_ranking", [])}
        
        # Find features that changed rank significantly
        significant_changes = []
        common_features = set(ranking1.keys()) & set(ranking2.keys())
        
        for feature in common_features:
            rank1 = ranking1[feature]["rank"]
            rank2 = ranking2[feature]["rank"]
            rank_diff = abs(rank1 - rank2)
            
            if rank_diff >= 5:  # Significant change
                direction = "↑" if rank2 < rank1 else "↓"
                significant_changes.append(
                    f"{feature}: Rank {rank1} → {rank2} {direction} ({rank_diff} positions)"
                )
        
        # Calculate similarity (Spearman correlation of ranks)
        common_features_list = sorted(common_features)
        ranks1 = [ranking1[f]["rank"] for f in common_features_list]
        ranks2 = [ranking2[f]["rank"] for f in common_features_list]
        
        try:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(ranks1, ranks2)
            similarity = f"{correlation:.3f}"
        except ImportError:
            similarity = "N/A (scipy not available)"
        
        summary = f"""
<b>Comparison Summary:</b><br>
• Common Features: {len(common_features)}<br>
• Rank Similarity (Spearman): {similarity}<br>
<br>
<b>Significant Rank Changes (≥5 positions):</b><br>
{chr(10).join(significant_changes[:10]) if significant_changes else "No significant rank changes"}
"""
        self.comparison_text.setHtml(summary)

