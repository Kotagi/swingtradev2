"""
Stop-Loss Analysis Tab

Analyzes stop-loss trades to identify patterns and generate filter recommendations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QFileDialog, QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit,
    QProgressBar, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from gui.services import StopLossAnalysisService, DataService

PROJECT_ROOT = Path(__file__).parent.parent.parent


class FeatureExtractionWorker(QThread):
    """Worker thread for extracting features at entry time."""
    
    finished = pyqtSignal(bool, pd.DataFrame, str)  # success, features_df, message
    progress = pyqtSignal(int, int)  # completed, total
    progress_message = pyqtSignal(str)  # message
    
    def __init__(self, service: StopLossAnalysisService, trades_df: pd.DataFrame, data_dir: Path):
        super().__init__()
        self.service = service
        self.trades_df = trades_df
        self.data_dir = data_dir
    
    def run(self):
        """Extract features in background thread."""
        try:
            def progress_callback(completed, total, message=None):
                self.progress.emit(completed, total)
                if message:
                    self.progress_message.emit(message)
            
            features_df = self.service.get_entry_features(
                self.trades_df,
                self.data_dir,
                progress_callback=progress_callback
            )
            
            if features_df.empty:
                self.finished.emit(False, pd.DataFrame(), 
                                 "No features extracted. Check that feature files exist.")
            else:
                self.finished.emit(True, features_df, 
                                 f"Extracted features for {len(features_df)} trades")
        except Exception as e:
            self.finished.emit(False, pd.DataFrame(), f"Error: {str(e)}")


class AnalysisWorker(QThread):
    """Worker thread for running stop-loss analysis."""
    
    finished = pyqtSignal(bool, dict, str)  # success, results, message
    progress = pyqtSignal(int, int)  # completed, total
    progress_message = pyqtSignal(str)  # message
    
    def __init__(self, service: StopLossAnalysisService, trades_df: pd.DataFrame, 
                 features_df: pd.DataFrame, effect_size_threshold: float):
        super().__init__()
        self.service = service
        self.trades_df = trades_df
        self.features_df = features_df
        self.effect_size_threshold = effect_size_threshold
    
    def run(self):
        """Run analysis in background thread."""
        try:
            def progress_callback(completed, total, message=None):
                self.progress.emit(completed, total)
                if message:
                    self.progress_message.emit(message)
            
            results = self.service.analyze_stop_losses(
                self.trades_df,
                self.features_df,
                self.effect_size_threshold,
                progress_callback=progress_callback
            )
            
            self.finished.emit(True, results, 
                             f"Analysis complete: {results.get('stop_loss_count', 0)} stop-losses found")
        except Exception as e:
            self.finished.emit(False, {}, f"Error: {str(e)}")


class StopLossAnalysisTab(QWidget):
    """Tab for analyzing stop-loss trades and generating filter recommendations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = StopLossAnalysisService()
        self.data_service = DataService()
        self.current_csv_path = None
        self.current_trades_df = None
        self.current_features_df = None
        self.analysis_results = None
        self.feature_worker = None
        self.analysis_worker = None
        self.init_ui()
    
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
        
        # Title
        title = QLabel("Stop-Loss Analysis")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        
        csv_row = QHBoxLayout()
        csv_row.addWidget(QLabel("Backtest CSV:"))
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setReadOnly(True)
        self.csv_path_edit.setPlaceholderText("No backtest CSV loaded")
        csv_row.addWidget(self.csv_path_edit)
        change_csv_btn = QPushButton("Change...")
        change_csv_btn.clicked.connect(self.browse_csv)
        csv_row.addWidget(change_csv_btn)
        input_layout.addLayout(csv_row)
        
        self.analyze_btn = QPushButton("Analyze Stop-Losses")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #000000;
                font-weight: bold;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #00c896;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        self.analyze_btn.clicked.connect(self.run_analysis)
        input_layout.addWidget(self.analyze_btn)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Summary cards (placeholder - will be populated in Unit 1.4)
        summary_group = QGroupBox("Summary")
        summary_layout = QHBoxLayout()
        
        self.total_trades_card, self.total_trades_label = self.create_summary_card("Total Trades", "0")
        self.stop_loss_card, self.stop_loss_label = self.create_summary_card("Stop-Losses", "0 (0%)")
        self.winners_card, self.winners_label = self.create_summary_card("Winners", "0 (0%)")
        self.target_card, self.target_label = self.create_summary_card("Target Reached", "0 (0%)")
        
        summary_layout.addWidget(self.total_trades_card)
        summary_layout.addWidget(self.stop_loss_card)
        summary_layout.addWidget(self.winners_card)
        summary_layout.addWidget(self.target_card)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Analysis settings (collapsible - will be implemented in Phase 2)
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QVBoxLayout()
        
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Effect Size Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 1.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(0.3)
        self.threshold_spin.setDecimals(2)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        settings_layout.addLayout(threshold_row)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Feature comparison table (placeholder - will be populated in Unit 1.5)
        table_group = QGroupBox("Top Differentiating Features")
        table_layout = QVBoxLayout()
        
        self.features_table = QTableWidget()
        self.features_table.setColumnCount(5)
        self.features_table.setHorizontalHeaderLabels([
            "Feature", "Stop-Loss Mean", "Winner Mean", "Difference", "Effect Size"
        ])
        self.features_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.features_table.setAlternatingRowColors(True)
        self.features_table.setMinimumHeight(300)
        table_layout.addWidget(self.features_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Recommendations panel
        recommendations_group = QGroupBox("Recommendations")
        recommendations_layout = QVBoxLayout()
        
        # Summary label showing count
        self.recommendations_summary_label = QLabel("Run analysis to generate recommendations")
        self.recommendations_summary_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        recommendations_layout.addWidget(self.recommendations_summary_label)
        
        # Scroll area for recommendations
        recommendations_scroll = QScrollArea()
        recommendations_scroll.setWidgetResizable(True)
        recommendations_scroll.setMinimumHeight(300)
        recommendations_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        # Container widget for recommendations
        self.recommendations_container = QWidget()
        self.recommendations_layout = QVBoxLayout()
        self.recommendations_layout.setSpacing(10)
        self.recommendations_container.setLayout(self.recommendations_layout)
        
        recommendations_scroll.setWidget(self.recommendations_container)
        recommendations_layout.addWidget(recommendations_scroll)
        
        recommendations_group.setLayout(recommendations_layout)
        layout.addWidget(recommendations_group)
        
        # Store references to recommendation sections
        self.strong_section = None
        self.moderate_section = None
        self.weak_section = None
        self.selected_filters = []  # List of selected filter dictionaries
        
        # Immediate Stop-Loss Analysis Section (collapsible)
        immediate_group = QGroupBox("Immediate Stop-Loss Analysis")
        immediate_group.setCheckable(True)
        immediate_group.setChecked(False)
        immediate_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ff9800;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        immediate_layout = QVBoxLayout()
        
        # Summary label
        self.immediate_summary_label = QLabel("Immediate stops (≤1 day) will be analyzed after running main analysis")
        self.immediate_summary_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        self.immediate_summary_label.setWordWrap(True)
        immediate_layout.addWidget(self.immediate_summary_label)
        
        # Analyze button
        analyze_immediate_btn = QPushButton("Analyze Immediate Stops")
        analyze_immediate_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: #000000;
                font-weight: bold;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #f57c00;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #808080;
            }
        """)
        analyze_immediate_btn.clicked.connect(self.analyze_immediate_stops)
        immediate_layout.addWidget(analyze_immediate_btn)
        
        # Special recommendations container
        self.immediate_recommendations_container = QWidget()
        self.immediate_recommendations_layout = QVBoxLayout()
        self.immediate_recommendations_layout.setSpacing(5)
        self.immediate_recommendations_container.setLayout(self.immediate_recommendations_layout)
        immediate_layout.addWidget(self.immediate_recommendations_container)
        
        # Action buttons for immediate stops
        immediate_action_row = QHBoxLayout()
        
        include_in_main_btn = QPushButton("Include in Main Recommendations")
        include_in_main_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #00d4aa;
                color: #000000;
            }
        """)
        include_in_main_btn.clicked.connect(self.include_immediate_in_main)
        immediate_action_row.addWidget(include_in_main_btn)
        
        exclude_immediate_btn = QPushButton("Exclude")
        exclude_immediate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #f44336;
                color: #ffffff;
            }
        """)
        exclude_immediate_btn.clicked.connect(self.exclude_immediate_recommendations)
        immediate_action_row.addWidget(exclude_immediate_btn)
        
        immediate_action_widget = QWidget()
        immediate_action_widget.setLayout(immediate_action_row)
        immediate_layout.addWidget(immediate_action_widget)
        
        immediate_group.setLayout(immediate_layout)
        immediate_group.toggled.connect(lambda checked: immediate_group.setVisible(checked))
        immediate_group.setVisible(False)
        layout.addWidget(immediate_group)
        
        self.immediate_group = immediate_group
        self.analyze_immediate_btn = analyze_immediate_btn
        self.immediate_stop_recommendations = []  # Store immediate stop recommendations
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setFormat("%p% (%v/%m)")
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("Ready - Load a backtest CSV to begin")
        self.status_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def create_summary_card(self, title: str, value: str) -> Tuple[QWidget, QLabel]:
        """Create a summary card widget and return both the widget and value label."""
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(10, 10, 10, 10)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #b0b0b0; font-size: 12px;")
        card_layout.addWidget(title_label)
        
        value_label = QLabel(value)
        value_label.setStyleSheet("color: #ffffff; font-size: 18px; font-weight: bold;")
        card_layout.addWidget(value_label)
        
        card.setLayout(card_layout)
        return card, value_label
    
    def browse_csv(self):
        """Browse for backtest CSV file."""
        backtest_dir = PROJECT_ROOT / "data" / "backtest_results"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Backtest CSV",
            str(backtest_dir),
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.current_csv_path = file_path
            self.csv_path_edit.setText(Path(file_path).name)
            self.status_label.setText(f"Loaded: {Path(file_path).name}")
            self.status_label.setStyleSheet("color: #4caf50;")
    
    def run_analysis(self):
        """Run stop-loss analysis."""
        if not self.current_csv_path:
            QMessageBox.warning(self, "No CSV Loaded", "Please load a backtest CSV file first.")
            return
        
        # Disable analyze button
        self.analyze_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Loading backtest CSV...")
        self.status_label.setStyleSheet("color: #ff9800;")
        
        try:
            # Load trades from CSV
            self.current_trades_df = pd.read_csv(self.current_csv_path)
            
            # Handle entry_date column (convert to datetime if needed)
            if 'entry_date' in self.current_trades_df.columns:
                self.current_trades_df['entry_date'] = pd.to_datetime(
                    self.current_trades_df['entry_date'], errors='coerce'
                )
                # Set entry_date as index for easier lookup
                self.current_trades_df = self.current_trades_df.set_index('entry_date')
            
            self.status_label.setText(f"Loaded {len(self.current_trades_df)} trades. Extracting features...")
            
            # Get data directory for features
            data_dir = PROJECT_ROOT / "data" / "features_labeled"
            
            # Start feature extraction in background thread
            self.feature_worker = FeatureExtractionWorker(
                self.service,
                self.current_trades_df,
                data_dir
            )
            self.feature_worker.finished.connect(self.on_feature_extraction_finished)
            self.feature_worker.progress.connect(self.on_feature_extraction_progress)
            self.feature_worker.progress_message.connect(self.on_feature_extraction_message)
            self.feature_worker.start()
            
        except Exception as e:
            self.status_label.setText(f"Error loading CSV: {str(e)}")
            self.status_label.setStyleSheet("color: #f44336;")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")
            self.analyze_btn.setEnabled(True)
    
    def on_feature_extraction_progress(self, completed: int, total: int):
        """Handle feature extraction progress updates."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(completed)
    
    def on_feature_extraction_message(self, message: str):
        """Handle feature extraction progress messages."""
        self.status_label.setText(message)
    
    def on_feature_extraction_finished(self, success: bool, features_df: pd.DataFrame, message: str):
        """Handle completion of feature extraction."""
        if not success:
            self.progress_bar.setVisible(False)
            self.analyze_btn.setEnabled(True)
            self.status_label.setText(f"✗ {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            QMessageBox.warning(self, "Feature Extraction Failed", message)
            return
        
        self.current_features_df = features_df
        
        # Now run the analysis
        self.status_label.setText("Running stop-loss analysis...")
        self.status_label.setStyleSheet("color: #ff9800;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Run analysis in background thread
        self.analysis_worker = AnalysisWorker(
            self.service,
            self.current_trades_df,
            self.current_features_df,
            self.threshold_spin.value()
        )
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.progress_message.connect(self.on_analysis_message)
        self.analysis_worker.start()
    
    def on_analysis_progress(self, completed: int, total: int):
        """Handle analysis progress updates."""
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(completed)
    
    def on_analysis_message(self, message: str):
        """Handle analysis progress messages."""
        self.status_label.setText(message)
    
    def on_analysis_finished(self, success: bool, results: dict, message: str):
        """Handle completion of analysis."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        if success:
            self.analysis_results = results
            self.update_summary_cards()
            self.update_feature_comparison_table()
            self.update_recommendations_display()
            self.update_immediate_stop_summary()
            self.status_label.setText(f"✓ {message}")
            self.status_label.setStyleSheet("color: #4caf50;")
        else:
            self.status_label.setText(f"✗ {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            QMessageBox.warning(self, "Analysis Failed", message)
    
    def update_immediate_stop_summary(self):
        """Update immediate stop summary label."""
        if not self.analysis_results:
            return
        
        immediate_count = self.analysis_results.get('immediate_stop_count', 0)
        immediate_rate = self.analysis_results.get('immediate_stop_rate', 0.0) * 100
        stop_loss_count = self.analysis_results.get('stop_loss_count', 0)
        
        if immediate_count > 0:
            self.immediate_summary_label.setText(
                f"Immediate Stops (≤1 day): {immediate_count} ({immediate_rate:.1f}% of stop-losses, "
                f"{immediate_count}/{stop_loss_count} total stop-losses)"
            )
            self.immediate_summary_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.immediate_group.setVisible(True)
            self.analyze_immediate_btn.setEnabled(True)
        else:
            self.immediate_summary_label.setText("No immediate stops (≤1 day) found")
            self.immediate_summary_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
            self.immediate_group.setVisible(False)
    
    def update_summary_cards(self):
        """Update summary cards with analysis results."""
        if not self.analysis_results:
            return
        
        results = self.analysis_results
        
        # Update total trades card
        self.total_trades_label.setText(str(results.get('total_trades', 0)))
        
        # Update stop-loss card
        sl_count = results.get('stop_loss_count', 0)
        sl_rate = results.get('stop_loss_rate', 0.0) * 100
        self.stop_loss_label.setText(f"{sl_count} ({sl_rate:.1f}%)")
        
        # Update winners card
        win_count = results.get('winning_count', 0)
        win_rate = (win_count / results.get('total_trades', 1)) * 100 if results.get('total_trades', 0) > 0 else 0
        self.winners_label.setText(f"{win_count} ({win_rate:.1f}%)")
        
        # Update target card
        target_count = results.get('target_count', 0)
        target_rate = (target_count / results.get('total_trades', 1)) * 100 if results.get('total_trades', 0) > 0 else 0
        self.target_label.setText(f"{target_count} ({target_rate:.1f}%)")
    
    def update_feature_comparison_table(self):
        """Update feature comparison table with analysis results."""
        if not self.analysis_results or not self.analysis_results.get('feature_comparisons'):
            self.features_table.setRowCount(0)
            return
        
        comparisons = self.analysis_results['feature_comparisons']
        self.features_table.setRowCount(len(comparisons))
        
        for row_idx, comp in enumerate(comparisons):
            # Feature name
            self.features_table.setItem(row_idx, 0, QTableWidgetItem(comp['feature']))
            
            # Stop-loss mean
            sl_item = QTableWidgetItem(f"{comp['stop_loss_mean']:.4f}")
            sl_item.setData(Qt.ItemDataRole.EditRole, comp['stop_loss_mean'])
            self.features_table.setItem(row_idx, 1, sl_item)
            
            # Winner mean
            win_item = QTableWidgetItem(f"{comp['winner_mean']:.4f}")
            win_item.setData(Qt.ItemDataRole.EditRole, comp['winner_mean'])
            self.features_table.setItem(row_idx, 2, win_item)
            
            # Difference
            diff_item = QTableWidgetItem(f"{comp['difference']:.4f}")
            diff_item.setData(Qt.ItemDataRole.EditRole, comp['difference'])
            self.features_table.setItem(row_idx, 3, diff_item)
            
            # Effect size
            effect_item = QTableWidgetItem(f"{comp['abs_effect']:.3f}")
            effect_item.setData(Qt.ItemDataRole.EditRole, comp['abs_effect'])
            self.features_table.setItem(row_idx, 4, effect_item)
        
        # Enable sorting
        self.features_table.setSortingEnabled(True)
    
    def update_recommendations_display(self):
        """Update recommendations display, grouped by effect size."""
        # Clear existing recommendations
        while self.recommendations_layout.count():
            child = self.recommendations_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.analysis_results or not self.analysis_results.get('recommendations'):
            self.recommendations_summary_label.setText("No recommendations generated")
            self.recommendations_summary_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
            return
        
        recommendations = self.analysis_results['recommendations']
        
        # Group by category
        strong_recs = [r for r in recommendations if r.get('category') == 'strong']
        moderate_recs = [r for r in recommendations if r.get('category') == 'moderate']
        weak_recs = [r for r in recommendations if r.get('category') == 'weak']
        
        total_above_threshold = len(recommendations)
        showing_count = len(strong_recs) + len(moderate_recs) + len(weak_recs)
        
        # Update summary label
        if total_above_threshold > showing_count:
            self.recommendations_summary_label.setText(
                f"Showing: {showing_count} recommendations ({total_above_threshold} total above threshold)"
            )
        else:
            self.recommendations_summary_label.setText(f"Showing: {showing_count} recommendations")
        self.recommendations_summary_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        
        # Strong recommendations (always visible)
        if strong_recs:
            self.strong_section = self.create_recommendation_section(
                "Strong Recommendations", strong_recs, "strong", expanded=True
            )
            self.recommendations_layout.addWidget(self.strong_section)
        
        # Moderate recommendations (always visible)
        if moderate_recs:
            self.moderate_section = self.create_recommendation_section(
                "Moderate Recommendations", moderate_recs, "moderate", expanded=True
            )
            self.recommendations_layout.addWidget(self.moderate_section)
        
        # Weak recommendations (collapsed by default)
        if weak_recs:
            self.weak_section = self.create_recommendation_section(
                "Weak Recommendations", weak_recs, "weak", expanded=False
            )
            self.recommendations_layout.addWidget(self.weak_section)
        
        # Impact Preview Panel (expandable)
        self.impact_preview_group = QGroupBox("Impact Preview")
        self.impact_preview_group.setCheckable(True)
        self.impact_preview_group.setChecked(False)
        self.impact_preview_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #00d4aa;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        impact_layout = QVBoxLayout()
        self.impact_preview_label = QLabel("Select filters to see impact preview")
        self.impact_preview_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        self.impact_preview_label.setWordWrap(True)
        impact_layout.addWidget(self.impact_preview_label)
        
        self.impact_preview_group.setLayout(impact_layout)
        self.impact_preview_group.toggled.connect(lambda checked: self.impact_preview_group.setVisible(checked))
        self.impact_preview_group.setVisible(False)
        self.recommendations_layout.addWidget(self.impact_preview_group)
        
        # Action buttons
        action_row = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_recommendations)
        action_row.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all_recommendations)
        action_row.addWidget(deselect_all_btn)
        
        action_row.addStretch()
        
        apply_selected_btn = QPushButton("Apply Selected")
        apply_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #000000;
                font-weight: bold;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #00c896;
            }
        """)
        apply_selected_btn.clicked.connect(self.apply_selected_filters)
        action_row.addWidget(apply_selected_btn)
        
        save_preset_btn = QPushButton("Save Selected as Preset")
        save_preset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #00d4aa;
                color: #000000;
            }
        """)
        save_preset_btn.clicked.connect(self.save_selected_as_preset)
        action_row.addWidget(save_preset_btn)
        
        action_widget = QWidget()
        action_widget.setLayout(action_row)
        self.recommendations_layout.addWidget(action_widget)
        
        # Add spacer
        self.recommendations_layout.addStretch()
    
    def create_recommendation_section(self, title: str, recommendations: list, category: str, expanded: bool = True) -> QGroupBox:
        """Create a collapsible section for recommendations."""
        section = QGroupBox(title)
        section.setCheckable(True)
        section.setChecked(expanded)
        section.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        # Set category-specific colors
        if category == "strong":
            section.setStyleSheet(section.styleSheet() + "QGroupBox { border-color: #4caf50; }")
        elif category == "moderate":
            section.setStyleSheet(section.styleSheet() + "QGroupBox { border-color: #ff9800; }")
        else:
            section.setStyleSheet(section.styleSheet() + "QGroupBox { border-color: #b0b0b0; }")
        
        section_layout = QVBoxLayout()
        section_layout.setSpacing(5)
        
        # Store recommendations in section for access
        section.recommendations = recommendations
        
        # Add each recommendation
        for rec in recommendations:
            rec_widget = self.create_recommendation_item(rec)
            section_layout.addWidget(rec_widget)
        
        section.setLayout(section_layout)
        
        # Connect toggle signal
        section.toggled.connect(lambda checked, s=section: s.setVisible(checked))
        
        return section
    
    def create_recommendation_item(self, recommendation: dict) -> QWidget:
        """Create a widget for a single recommendation."""
        item_widget = QWidget()
        item_layout = QHBoxLayout()
        item_layout.setContentsMargins(10, 5, 10, 5)
        
        # Checkbox for selection (will be implemented in Unit 2.2)
        checkbox = QCheckBox()
        checkbox.setChecked(False)
        checkbox.recommendation = recommendation  # Store reference
        checkbox.stateChanged.connect(lambda state, rec=recommendation: self.on_recommendation_toggled(rec, state == 2))
        item_layout.addWidget(checkbox)
        
        # Description label
        desc_label = QLabel(recommendation.get('description', ''))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #ffffff;")
        item_layout.addWidget(desc_label, 1)
        
        # Info button (will be implemented later with feature descriptions)
        info_btn = QPushButton("ℹ")
        info_btn.setFixedSize(25, 25)
        info_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #00d4aa;
                border-radius: 12px;
                color: #00d4aa;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00d4aa;
                color: #000000;
            }
        """)
        info_btn.setToolTip("Show feature information")
        # Will connect to feature info dialog in later phase
        item_layout.addWidget(info_btn)
        
        # Preview Impact button (will be implemented in Unit 2.3)
        preview_btn = QPushButton("Preview Impact")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #00d4aa;
                color: #000000;
            }
        """)
        preview_btn.setToolTip("Preview impact of this filter")
        preview_btn.recommendation = recommendation
        # Will connect to impact preview in Unit 2.3
        item_layout.addWidget(preview_btn)
        
        item_widget.setLayout(item_layout)
        return item_widget
    
    def on_recommendation_toggled(self, recommendation: dict, checked: bool):
        """Handle recommendation checkbox toggle."""
        if checked:
            if recommendation not in self.selected_filters:
                self.selected_filters.append(recommendation)
        else:
            if recommendation in self.selected_filters:
                self.selected_filters.remove(recommendation)
        
        # Update impact preview
        self.update_impact_preview()
    
    def select_all_recommendations(self):
        """Select all recommendation checkboxes."""
        if not self.analysis_results or not self.analysis_results.get('recommendations'):
            return
        
        # Clear and rebuild selected_filters
        self.selected_filters.clear()
        self.selected_filters = self.analysis_results['recommendations'].copy()
        
        # Find all checkboxes in recommendations container
        for i in range(self.recommendations_layout.count()):
            item = self.recommendations_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QGroupBox):
                    # Find checkboxes in this group
                    self._set_checkboxes_in_widget(widget, True)
        
        # Update impact preview
        self.update_impact_preview()
    
    def deselect_all_recommendations(self):
        """Deselect all recommendation checkboxes."""
        # Find all checkboxes in recommendations container
        for i in range(self.recommendations_layout.count()):
            item = self.recommendations_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QGroupBox):
                    # Find checkboxes in this group
                    self._set_checkboxes_in_widget(widget, False)
        
        self.selected_filters.clear()
        
        # Update impact preview
        self.update_impact_preview()
    
    def _set_checkboxes_in_widget(self, widget: QWidget, checked: bool):
        """Recursively set all checkboxes in a widget."""
        if isinstance(widget, QCheckBox):
            widget.setChecked(checked)
        elif isinstance(widget, QGroupBox):
            layout = widget.layout()
            if layout:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        self._set_checkboxes_in_widget(item.widget(), checked)
        else:
            layout = widget.layout()
            if layout:
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        self._set_checkboxes_in_widget(item.widget(), checked)
    
    def update_impact_preview(self):
        """Update the impact preview panel based on selected filters."""
        if not self.selected_filters:
            self.impact_preview_group.setVisible(False)
            return
        
        if self.current_trades_df is None or self.current_features_df is None:
            return
        
        # Convert selected filters to (feature, operator, value) tuples
        filter_tuples = [
            (f['feature'], f['operator'], f['value'])
            for f in self.selected_filters
        ]
        
        # Calculate impact
        try:
            impact = self.service.calculate_impact(
                filter_tuples,
                self.current_trades_df,
                self.current_features_df
            )
            
            # Build preview text
            preview_text = f"<b>Selected filters would:</b><br><br>"
            preview_text += f"• Exclude ~{impact['stop_loss_excluded_pct']:.1f}% of stop-loss trades ({impact.get('stop_loss_count_before', 0) - impact.get('stop_loss_count_after', 0)} trades)<br>"
            preview_text += f"• Exclude ~{impact['winner_excluded_pct']:.1f}% of winning trades<br>"
            preview_text += f"• Estimated new stop-loss rate: <b>{impact['estimated_new_sl_rate']:.1f}%</b> "
            
            # Calculate improvement
            if self.analysis_results:
                original_sl_rate = self.analysis_results.get('stop_loss_rate', 0.0) * 100
                improvement = original_sl_rate - impact['estimated_new_sl_rate']
                if improvement > 0:
                    preview_text += f"(<span style='color: #4caf50;'>↓ {improvement:.1f}%</span>)<br>"
                else:
                    preview_text += f"(<span style='color: #f44336;'>↑ {abs(improvement):.1f}%</span>)<br>"
            else:
                preview_text += "<br>"
            
            preview_text += f"• Estimated total trades: <b>{impact['estimated_total_trades']}</b> (down from {impact.get('total_trades_before', 0)})<br>"
            
            # Add warnings
            if impact['warnings']:
                preview_text += "<br>"
                for warning in impact['warnings']:
                    preview_text += f"<span style='color: #ff9800;'>{warning}</span><br>"
            
            self.impact_preview_label.setText(preview_text)
            self.impact_preview_label.setStyleSheet("color: #ffffff;")
            self.impact_preview_group.setVisible(True)
            self.impact_preview_group.setChecked(True)
            
        except Exception as e:
            self.impact_preview_label.setText(f"Error calculating impact: {str(e)}")
            self.impact_preview_label.setStyleSheet("color: #f44336;")
            self.impact_preview_group.setVisible(True)
    
    def apply_selected_filters(self):
        """Apply selected filters - shows impact preview."""
        if not self.selected_filters:
            QMessageBox.information(self, "No Filters Selected", "Please select at least one filter to apply.")
            return
        
        # Impact preview is already shown, just ensure it's visible
        self.impact_preview_group.setChecked(True)
        self.impact_preview_group.setVisible(True)
    
    def save_selected_as_preset(self):
        """Save selected filters as a preset (will be implemented in Unit 4.1)."""
        if not self.selected_filters:
            QMessageBox.information(self, "No Filters Selected", "Please select at least one filter to save as a preset.")
            return
        
        # This will be implemented in Unit 4.1
        QMessageBox.information(self, "Save Preset", 
                               "Preset saving will be implemented in Unit 4.1.")
    
    def analyze_immediate_stops(self):
        """Analyze immediate stops (≤1 day) separately."""
        if self.current_features_df is None or self.current_features_df.empty:
            QMessageBox.warning(self, "No Data", "Please run main analysis first.")
            return
        
        # Filter for immediate stops
        immediate_mask = self.current_features_df.get('holding_days', pd.Series([999] * len(self.current_features_df))) <= 1
        immediate_stops = self.current_features_df[immediate_mask].copy()
        
        if immediate_stops.empty:
            QMessageBox.information(self, "No Immediate Stops", "No immediate stops found to analyze.")
            return
        
        # Get winners for comparison
        if 'exit_reason' in self.current_features_df.columns:
            winners = self.current_features_df[
                (self.current_features_df['exit_reason'] != 'stop_loss') & 
                (self.current_features_df.get('return', 0) > 0)
            ].copy()
        else:
            winners = self.current_features_df[self.current_features_df.get('return', 0) > 0].copy()
        
        if winners.empty:
            QMessageBox.warning(self, "No Winners", "No winning trades found for comparison.")
            return
        
        # Compare features
        exclude_cols = {'ticker', 'entry_date', 'exit_reason', 'return', 'pnl', 'holding_days'}
        feature_cols = [col for col in self.current_features_df.columns if col not in exclude_cols]
        
        recommendations = []
        for feat in feature_cols:
            try:
                immediate_values = immediate_stops[feat].dropna()
                winner_values = winners[feat].dropna()
                
                if len(immediate_values) > 0 and len(winner_values) > 0:
                    immediate_mean = immediate_values.mean()
                    winner_mean = winner_values.mean()
                    immediate_std = immediate_values.std()
                    winner_std = winner_values.std()
                    
                    if pd.notna(immediate_mean) and pd.notna(winner_mean) and immediate_std > 0:
                        pooled_std = np.sqrt((immediate_std**2 + winner_std**2) / 2)
                        if pooled_std > 0:
                            cohens_d = (immediate_mean - winner_mean) / pooled_std
                            abs_effect = abs(cohens_d)
                            
                            # Only include if effect size is significant (≥0.2 for immediate stops)
                            if abs_effect >= 0.2:
                                # Determine operator and value
                                if immediate_mean > winner_mean:
                                    operator = "<"
                                    threshold_value = winner_mean + (immediate_mean - winner_mean) * 0.3
                                else:
                                    operator = ">"
                                    threshold_value = immediate_mean + (winner_mean - immediate_mean) * 0.3
                                
                                # Categorize
                                if abs_effect > 0.5:
                                    category = "strong"
                                elif abs_effect >= 0.3:
                                    category = "moderate"
                                else:
                                    category = "weak"
                                
                                recommendations.append({
                                    'feature': feat,
                                    'operator': operator,
                                    'value': float(threshold_value),
                                    'effect_size': float(abs_effect),
                                    'cohens_d': float(cohens_d),
                                    'category': category,
                                    'description': f"Filter: {feat} {operator} {threshold_value:.4f} (effect size: {abs_effect:.3f})"
                                })
            except Exception:
                continue
        
        # Sort by effect size
        recommendations.sort(key=lambda x: x['effect_size'], reverse=True)
        self.immediate_stop_recommendations = recommendations
        
        # Display recommendations
        self.display_immediate_recommendations()
    
    def display_immediate_recommendations(self):
        """Display immediate stop recommendations."""
        # Clear existing
        while self.immediate_recommendations_layout.count():
            child = self.immediate_recommendations_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not self.immediate_stop_recommendations:
            no_recs_label = QLabel("No significant patterns found in immediate stops")
            no_recs_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
            self.immediate_recommendations_layout.addWidget(no_recs_label)
            return
        
        # Group by category
        strong_recs = [r for r in self.immediate_stop_recommendations if r.get('category') == 'strong']
        moderate_recs = [r for r in self.immediate_stop_recommendations if r.get('category') == 'moderate']
        weak_recs = [r for r in self.immediate_stop_recommendations if r.get('category') == 'weak']
        
        # Display recommendations
        if strong_recs:
            strong_label = QLabel(f"<b>Strong Recommendations ({len(strong_recs)}):</b>")
            strong_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.immediate_recommendations_layout.addWidget(strong_label)
            for rec in strong_recs:
                rec_widget = self.create_immediate_recommendation_item(rec)
                self.immediate_recommendations_layout.addWidget(rec_widget)
        
        if moderate_recs:
            moderate_label = QLabel(f"<b>Moderate Recommendations ({len(moderate_recs)}):</b>")
            moderate_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.immediate_recommendations_layout.addWidget(moderate_label)
            for rec in moderate_recs:
                rec_widget = self.create_immediate_recommendation_item(rec)
                self.immediate_recommendations_layout.addWidget(rec_widget)
        
        if weak_recs:
            weak_label = QLabel(f"<b>Weak Recommendations ({len(weak_recs)}):</b>")
            weak_label.setStyleSheet("color: #b0b0b0; font-weight: bold;")
            self.immediate_recommendations_layout.addWidget(weak_label)
            for rec in weak_recs:
                rec_widget = self.create_immediate_recommendation_item(rec)
                self.immediate_recommendations_layout.addWidget(rec_widget)
        
        self.immediate_recommendations_layout.addStretch()
    
    def create_immediate_recommendation_item(self, recommendation: dict) -> QWidget:
        """Create a widget for a single immediate stop recommendation."""
        item_widget = QWidget()
        item_layout = QHBoxLayout()
        item_layout.setContentsMargins(10, 5, 10, 5)
        
        # Description label
        desc_label = QLabel(recommendation.get('description', ''))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #ffffff;")
        item_layout.addWidget(desc_label, 1)
        
        item_widget.setLayout(item_layout)
        return item_widget
    
    def include_immediate_in_main(self):
        """Include immediate stop recommendations in main recommendations."""
        if not self.immediate_stop_recommendations:
            QMessageBox.information(self, "No Recommendations", "No immediate stop recommendations to include.")
            return
        
        # Add to main recommendations
        if not self.analysis_results:
            self.analysis_results = {}
        
        if 'recommendations' not in self.analysis_results:
            self.analysis_results['recommendations'] = []
        
        # Add immediate recommendations (mark them as from immediate stops)
        for rec in self.immediate_stop_recommendations:
            rec_copy = rec.copy()
            rec_copy['from_immediate'] = True
            if rec_copy not in self.analysis_results['recommendations']:
                self.analysis_results['recommendations'].append(rec_copy)
        
        # Re-display main recommendations
        self.update_recommendations_display()
        
        QMessageBox.information(self, "Included", 
                               f"{len(self.immediate_stop_recommendations)} immediate stop recommendation(s) added to main recommendations.")
    
    def exclude_immediate_recommendations(self):
        """Exclude immediate stop recommendations from main recommendations."""
        if not self.analysis_results or 'recommendations' not in self.analysis_results:
            return
        
        # Remove recommendations marked as from immediate stops
        original_count = len(self.analysis_results['recommendations'])
        self.analysis_results['recommendations'] = [
            rec for rec in self.analysis_results['recommendations'] 
            if not rec.get('from_immediate', False)
        ]
        
        removed_count = original_count - len(self.analysis_results['recommendations'])
        
        if removed_count > 0:
            # Re-display main recommendations
            self.update_recommendations_display()
            QMessageBox.information(self, "Excluded", 
                                   f"{removed_count} immediate stop recommendation(s) removed from main recommendations.")
        else:
            QMessageBox.information(self, "No Change", "No immediate stop recommendations found in main recommendations.")

