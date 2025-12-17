"""
SHAP Dialog - Display SHAP explanations for models

Shows feature importance rankings, summary plots, and metadata for model interpretability.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QScrollArea, QGroupBox,
    QTextEdit, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from pathlib import Path
from typing import Dict, Any, Optional


class SHAPDialog(QDialog):
    """Dialog to display SHAP explanations for a single model."""
    
    def __init__(self, model: Dict[str, Any], artifacts: Dict[str, Any], parent=None):
        """
        Initialize SHAP dialog.
        
        Args:
            model: Model dictionary from registry
            artifacts: SHAP artifacts dictionary (from SHAPService.load_artifacts)
            parent: Parent widget
        """
        super().__init__(parent)
        self.model = model
        self.artifacts = artifacts
        self.setWindowTitle(f"SHAP Explanations - {model.get('name', 'Unknown Model')}")
        self.setMinimumSize(900, 700)
        
        self.init_ui()
        self._center_dialog()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel(f"<b>Model:</b> {self.model.get('name', 'Unknown')}"))
        info_layout.addWidget(QLabel(f"<b>Training Date:</b> {self.model.get('training_date', 'Unknown')}"))
        
        # SHAP metadata
        metadata = self.artifacts.get("metadata", {})
        if metadata:
            info_layout.addWidget(QLabel(f"<b>Data Split:</b> {metadata.get('data_split', 'Unknown')}"))
            info_layout.addWidget(QLabel(f"<b>Samples:</b> {metadata.get('sample_size', 'Unknown')} / {metadata.get('total_samples_available', 'Unknown')}"))
            info_layout.addWidget(QLabel(f"<b>Computation Date:</b> {metadata.get('computation_date', 'Unknown')}"))
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Tabs for different views
        tabs = QTabWidget()
        
        # Tab 1: Feature Importance Ranking
        ranking_tab = QWidget()
        ranking_layout = QVBoxLayout()
        
        ranking_label = QLabel("<b>Feature Importance Ranking (SHAP)</b>")
        ranking_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        ranking_layout.addWidget(ranking_label)
        
        self.ranking_table = QTableWidget()
        self.ranking_table.setColumnCount(4)
        self.ranking_table.setHorizontalHeaderLabels(["Rank", "Feature", "Importance", "Importance %"])
        self.ranking_table.horizontalHeader().setSectionResizeMode(1, QTableWidget.ResizeMode.Stretch)
        self.ranking_table.setAlternatingRowColors(True)
        self._populate_ranking_table()
        ranking_layout.addWidget(self.ranking_table)
        
        ranking_tab.setLayout(ranking_layout)
        tabs.addTab(ranking_tab, "Feature Ranking")
        
        # Tab 2: Summary Plot
        plot_tab = QWidget()
        plot_layout = QVBoxLayout()
        
        plot_label = QLabel("<b>SHAP Summary Plot</b>")
        plot_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        plot_layout.addWidget(plot_label)
        
        plot_image_label = QLabel()
        plot_path = self.artifacts.get("plot_path")
        if plot_path and Path(plot_path).exists():
            pixmap = QPixmap(plot_path)
            # Scale to fit dialog width while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(850, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            plot_image_label.setPixmap(scaled_pixmap)
            plot_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        else:
            plot_image_label.setText("Summary plot not available")
            plot_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plot_image_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        
        plot_layout.addWidget(plot_image_label)
        plot_tab.setLayout(plot_layout)
        tabs.addTab(plot_tab, "Summary Plot")
        
        # Tab 3: Statistics
        stats_tab = QWidget()
        stats_layout = QVBoxLayout()
        
        stats_label = QLabel("<b>SHAP Statistics</b>")
        stats_label.setStyleSheet("font-size: 14px; margin-bottom: 10px;")
        stats_layout.addWidget(stats_label)
        
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setPlainText(self._format_statistics())
        stats_layout.addWidget(stats_text)
        
        stats_tab.setLayout(stats_layout)
        tabs.addTab(stats_tab, "Statistics")
        
        layout.addWidget(tabs)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _populate_ranking_table(self):
        """Populate the feature importance ranking table."""
        ranking = self.artifacts.get("importance_ranking", [])
        
        self.ranking_table.setRowCount(len(ranking))
        
        for row, item in enumerate(ranking):
            # Rank
            rank_item = QTableWidgetItem(str(item.get("rank", row + 1)))
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.ranking_table.setItem(row, 0, rank_item)
            
            # Feature name
            feature_item = QTableWidgetItem(item.get("feature", "Unknown"))
            self.ranking_table.setItem(row, 1, feature_item)
            
            # Importance (raw value)
            importance = item.get("importance", 0.0)
            importance_item = QTableWidgetItem(f"{importance:.6f}")
            importance_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            # Store numeric value for sorting
            importance_item.setData(Qt.ItemDataRole.EditRole, importance)
            self.ranking_table.setItem(row, 2, importance_item)
            
            # Importance percentage
            importance_pct = item.get("importance_pct", 0.0)
            pct_item = QTableWidgetItem(f"{importance_pct:.2f}%")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            # Store numeric value for sorting
            pct_item.setData(Qt.ItemDataRole.EditRole, importance_pct)
            self.ranking_table.setItem(row, 3, pct_item)
        
        # Enable sorting
        self.ranking_table.setSortingEnabled(True)
        # Sort by rank by default
        self.ranking_table.sortItems(0, Qt.SortOrder.AscendingOrder)
    
    def _format_statistics(self) -> str:
        """Format SHAP statistics as text."""
        summary = self.artifacts.get("summary", {})
        metadata = self.artifacts.get("metadata", {})
        stats = summary.get("statistics", {})
        
        lines = []
        lines.append("=== SHAP Computation Metadata ===")
        lines.append(f"Model ID: {metadata.get('model_id', 'Unknown')}")
        lines.append(f"Computation Date: {metadata.get('computation_date', 'Unknown')}")
        lines.append(f"Data Split: {metadata.get('data_split', 'Unknown')}")
        lines.append(f"Sample Size: {metadata.get('sample_size', 'Unknown')}")
        lines.append(f"Total Samples Available: {metadata.get('total_samples_available', 'Unknown')}")
        lines.append(f"Number of Features: {metadata.get('n_features', 'Unknown')}")
        lines.append(f"SHAP Version: {metadata.get('shap_version', 'Unknown')}")
        
        lines.append("\n=== Feature Concentration ===")
        top_n_info = metadata.get("top_n_concentration", {})
        if top_n_info:
            lines.append(f"Top {top_n_info.get('n', 10)} features account for {top_n_info.get('percentage', 0):.2f}% of total importance")
        
        lines.append("\n=== SHAP Statistics ===")
        if stats:
            lines.append(f"Total Importance: {stats.get('total_importance', 0):.6f}")
            lines.append(f"Mean Importance: {stats.get('mean_importance', 0):.6f}")
            lines.append(f"Std Importance: {stats.get('std_importance', 0):.6f}")
            lines.append(f"Max Importance: {stats.get('max_importance', 0):.6f}")
            lines.append(f"Min Importance: {stats.get('min_importance', 0):.6f}")
        
        return "\n".join(lines)
    
    def _center_dialog(self):
        """Center the dialog on the screen."""
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        dialog_geometry = self.frameGeometry()
        dialog_geometry.moveCenter(screen.center())
        self.move(dialog_geometry.topLeft())


class SHAPComparisonDialog(QDialog):
    """Dialog to compare SHAP explanations between two models."""
    
    def __init__(
        self,
        model1: Dict[str, Any],
        artifacts1: Dict[str, Any],
        model2: Dict[str, Any],
        artifacts2: Dict[str, Any],
        parent=None
    ):
        """
        Initialize SHAP comparison dialog.
        
        Args:
            model1: First model dictionary
            artifacts1: First model's SHAP artifacts
            model2: Second model dictionary
            artifacts2: Second model's SHAP artifacts
            parent: Parent widget
        """
        super().__init__(parent)
        self.model1 = model1
        self.artifacts1 = artifacts1
        self.model2 = model2
        self.artifacts2 = artifacts2
        
        self.setWindowTitle(f"SHAP Comparison: {model1.get('name', 'Model 1')} vs {model2.get('name', 'Model 2')}")
        self.setMinimumSize(1200, 800)
        
        self.init_ui()
        self._center_dialog()
    
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Models info
        info_group = QGroupBox("Models Being Compared")
        info_layout = QHBoxLayout()
        
        model1_info = QVBoxLayout()
        model1_info.addWidget(QLabel(f"<b>Model 1:</b> {self.model1.get('name', 'Unknown')}"))
        model1_info.addWidget(QLabel(f"Training Date: {self.model1.get('training_date', 'Unknown')}"))
        metadata1 = self.artifacts1.get("metadata", {})
        if metadata1:
            model1_info.addWidget(QLabel(f"SHAP Data: {metadata1.get('data_split', 'Unknown')} ({metadata1.get('sample_size', 'Unknown')} samples)"))
        
        model2_info = QVBoxLayout()
        model2_info.addWidget(QLabel(f"<b>Model 2:</b> {self.model2.get('name', 'Unknown')}"))
        model2_info.addWidget(QLabel(f"Training Date: {self.model2.get('training_date', 'Unknown')}"))
        metadata2 = self.artifacts2.get("metadata", {})
        if metadata2:
            model2_info.addWidget(QLabel(f"SHAP Data: {metadata2.get('data_split', 'Unknown')} ({metadata2.get('sample_size', 'Unknown')} samples)"))
        
        info_layout.addLayout(model1_info)
        info_layout.addLayout(model2_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Comparison table
        comparison_label = QLabel("<b>Feature Importance Comparison</b>")
        comparison_label.setStyleSheet("font-size: 14px; margin-top: 10px;")
        layout.addWidget(comparison_label)
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(6)
        self.comparison_table.setHorizontalHeaderLabels([
            "Feature", "Model 1 Rank", "Model 1 %", "Model 2 Rank", "Model 2 %", "Delta"
        ])
        self.comparison_table.horizontalHeader().setSectionResizeMode(0, QTableWidget.ResizeMode.Stretch)
        self.comparison_table.setAlternatingRowColors(True)
        self._populate_comparison_table()
        layout.addWidget(self.comparison_table)
        
        # Statistics
        stats_group = QGroupBox("Comparison Statistics")
        stats_layout = QVBoxLayout()
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setPlainText(self._format_comparison_statistics())
        stats_layout.addWidget(stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _populate_comparison_table(self):
        """Populate the comparison table."""
        ranking1 = {item["feature"]: item for item in self.artifacts1.get("importance_ranking", [])}
        ranking2 = {item["feature"]: item for item in self.artifacts2.get("importance_ranking", [])}
        
        # Get all unique features
        all_features = set(ranking1.keys()) | set(ranking2.keys())
        
        # Create comparison data
        comparison_data = []
        for feature in all_features:
            item1 = ranking1.get(feature, {})
            item2 = ranking2.get(feature, {})
            
            rank1 = item1.get("rank", None)
            pct1 = item1.get("importance_pct", 0.0)
            rank2 = item2.get("rank", None)
            pct2 = item2.get("importance_pct", 0.0)
            
            # Calculate delta (change in percentage)
            delta = pct2 - pct1
            
            comparison_data.append({
                "feature": feature,
                "rank1": rank1,
                "pct1": pct1,
                "rank2": rank2,
                "pct2": pct2,
                "delta": delta
            })
        
        # Sort by absolute delta (biggest changes first)
        comparison_data.sort(key=lambda x: abs(x["delta"]), reverse=True)
        
        self.comparison_table.setRowCount(len(comparison_data))
        
        for row, data in enumerate(comparison_data):
            # Feature name
            feature_item = QTableWidgetItem(data["feature"])
            self.comparison_table.setItem(row, 0, feature_item)
            
            # Model 1 rank
            rank1_str = str(data["rank1"]) if data["rank1"] is not None else "N/A"
            rank1_item = QTableWidgetItem(rank1_str)
            rank1_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if data["rank1"] is not None:
                rank1_item.setData(Qt.ItemDataRole.EditRole, data["rank1"])
            self.comparison_table.setItem(row, 1, rank1_item)
            
            # Model 1 percentage
            pct1_item = QTableWidgetItem(f"{data['pct1']:.2f}%")
            pct1_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct1_item.setData(Qt.ItemDataRole.EditRole, data["pct1"])
            self.comparison_table.setItem(row, 2, pct1_item)
            
            # Model 2 rank
            rank2_str = str(data["rank2"]) if data["rank2"] is not None else "N/A"
            rank2_item = QTableWidgetItem(rank2_str)
            rank2_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if data["rank2"] is not None:
                rank2_item.setData(Qt.ItemDataRole.EditRole, data["rank2"])
            self.comparison_table.setItem(row, 3, rank2_item)
            
            # Model 2 percentage
            pct2_item = QTableWidgetItem(f"{data['pct2']:.2f}%")
            pct2_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct2_item.setData(Qt.ItemDataRole.EditRole, data["pct2"])
            self.comparison_table.setItem(row, 4, pct2_item)
            
            # Delta
            delta_str = f"{data['delta']:+.2f}%"
            delta_item = QTableWidgetItem(delta_str)
            delta_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            delta_item.setData(Qt.ItemDataRole.EditRole, data["delta"])
            # Color code: green for increase, red for decrease
            if data["delta"] > 0:
                delta_item.setForeground(Qt.GlobalColor.green)
            elif data["delta"] < 0:
                delta_item.setForeground(Qt.GlobalColor.red)
            self.comparison_table.setItem(row, 5, delta_item)
        
        # Enable sorting
        self.comparison_table.setSortingEnabled(True)
        # Sort by absolute delta by default (biggest changes first)
        self.comparison_table.sortItems(5, Qt.SortOrder.DescendingOrder)
    
    def _format_comparison_statistics(self) -> str:
        """Format comparison statistics as text."""
        ranking1 = {item["feature"]: item for item in self.artifacts1.get("importance_ranking", [])}
        ranking2 = {item["feature"]: item for item in self.artifacts2.get("importance_ranking", [])}
        
        # Get top 10 features from each
        top10_1 = sorted(ranking1.values(), key=lambda x: x.get("rank", 999))[:10]
        top10_2 = sorted(ranking2.values(), key=lambda x: x.get("rank", 999))[:10]
        
        # Calculate similarity (Spearman correlation of ranks)
        common_features = set(ranking1.keys()) & set(ranking2.keys())
        if len(common_features) >= 2:
            ranks1 = [ranking1[f].get("rank", 999) for f in common_features]
            ranks2 = [ranking2[f].get("rank", 999) for f in common_features]
            try:
                from scipy.stats import spearmanr
                correlation, p_value = spearmanr(ranks1, ranks2)
                similarity = f"{correlation:.3f} (p={p_value:.3f})"
            except ImportError:
                # Fallback: simple rank difference calculation
                rank_diffs = [abs(r1 - r2) for r1, r2 in zip(ranks1, ranks2)]
                avg_diff = sum(rank_diffs) / len(rank_diffs) if rank_diffs else 0
                similarity = f"Average rank difference: {avg_diff:.1f} (scipy not available for correlation)"
            except Exception:
                similarity = "Could not calculate"
        else:
            similarity = "Not enough common features"
        
        lines = []
        lines.append("=== Top 10 Features Comparison ===")
        lines.append("\nModel 1 Top 10:")
        for item in top10_1:
            lines.append(f"  {item.get('rank', '?'):2d}. {item.get('feature', 'Unknown'):30s} {item.get('importance_pct', 0):6.2f}%")
        
        lines.append("\nModel 2 Top 10:")
        for item in top10_2:
            lines.append(f"  {item.get('rank', '?'):2d}. {item.get('feature', 'Unknown'):30s} {item.get('importance_pct', 0):6.2f}%")
        
        lines.append(f"\n=== Similarity ===")
        lines.append(f"Rank Correlation: {similarity}")
        lines.append(f"Common Features: {len(common_features)}")
        
        return "\n".join(lines)
    
    def _center_dialog(self):
        """Center the dialog on the screen."""
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen().availableGeometry()
        dialog_geometry = self.frameGeometry()
        dialog_geometry.moveCenter(screen.center())
        self.move(dialog_geometry.topLeft())

