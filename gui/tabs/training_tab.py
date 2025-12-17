"""
Model Training Tab

Allows users to train ML models with various configurations.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QMessageBox, QProgressBar, QTextEdit, QLineEdit, QFileDialog,
    QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path
from datetime import datetime

from gui.services import TrainingService, DataService
from gui.widgets import PresetManagerWidget
from gui.utils.model_registry import ModelRegistry
from gui.tabs.feature_selection_dialog import FeatureSelectionDialog
import yaml


class TrainingWorker(QThread):
    """Worker thread for model training."""
    
    finished = pyqtSignal(bool, str, dict)  # success, message, metrics_dict
    progress = pyqtSignal(int, int)  # current_stage, total_stages (for progress bar)
    progress_message = pyqtSignal(str)  # progress message text
    
    def __init__(self, service: TrainingService, **kwargs):
        super().__init__()
        self.service = service
        self.kwargs = kwargs
    
    def training_progress_callback(self, current_stage: int, total_stages: int, message: str):
        """Callback for training progress updates."""
        self.progress.emit(current_stage, total_stages)
        self.progress_message.emit(message)
    
    def run(self):
        """Run model training in background thread."""
        try:
            # Add progress callback
            self.kwargs['progress_callback'] = self.training_progress_callback
            success, message, metrics_dict = self.service.train_model(**self.kwargs)
            self.finished.emit(success, message, metrics_dict)
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}", {})


class TrainingTab(QWidget):
    """Tab for model training."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = TrainingService()
        self.data_service = DataService()
        self.registry = ModelRegistry()
        self.worker = None
        
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
        title = QLabel("Model Training")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Training options group
        options_group = QGroupBox("Training Options")
        options_layout = QVBoxLayout()
        
        # Hyperparameter tuning
        tune_row = QHBoxLayout()
        self.tune_check = QCheckBox("Enable Hyperparameter Tuning")
        self.tune_check.setChecked(False)
        tune_row.addWidget(self.tune_check)
        tune_row.addStretch()
        options_layout.addLayout(tune_row)
        
        # Tuning iterations (shown when tuning enabled)
        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Tuning Iterations:"))
        self.n_iter_spin = QSpinBox()
        self.n_iter_spin.setRange(5, 100)
        self.n_iter_spin.setValue(20)
        iter_row.addWidget(self.n_iter_spin)
        iter_row.addStretch()
        options_layout.addLayout(iter_row)
        
        # Cross-validation
        cv_row = QHBoxLayout()
        self.cv_check = QCheckBox("Use Cross-Validation")
        cv_row.addWidget(self.cv_check)
        cv_row.addStretch()
        options_layout.addLayout(cv_row)
        
        # CV folds
        folds_row = QHBoxLayout()
        folds_row.addWidget(QLabel("CV Folds:"))
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(2, 10)
        self.cv_folds_spin.setValue(3)
        folds_row.addWidget(self.cv_folds_spin)
        folds_row.addStretch()
        options_layout.addLayout(folds_row)
        
        # Fast mode
        fast_row = QHBoxLayout()
        self.fast_check = QCheckBox("Fast Mode (faster but less optimal)")
        fast_row.addWidget(self.fast_check)
        fast_row.addStretch()
        options_layout.addLayout(fast_row)
        
        # Other options
        other_row = QHBoxLayout()
        self.plots_check = QCheckBox("Generate Plots")
        self.diagnostics_check = QCheckBox("SHAP Diagnostics")
        self.no_early_stop_check = QCheckBox("Disable Early Stopping")
        other_row.addWidget(self.plots_check)
        other_row.addWidget(self.diagnostics_check)
        other_row.addWidget(self.no_early_stop_check)
        other_row.addStretch()
        options_layout.addLayout(other_row)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Advanced parameters group
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_layout = QVBoxLayout()
        
        # Imbalance multiplier
        imbalance_row = QHBoxLayout()
        imbalance_row.addWidget(QLabel("Class Imbalance Multiplier:"))
        self.imbalance_spin = QDoubleSpinBox()
        self.imbalance_spin.setRange(0.5, 5.0)
        self.imbalance_spin.setSingleStep(0.1)
        self.imbalance_spin.setValue(1.0)
        self.imbalance_spin.setDecimals(1)
        imbalance_row.addWidget(self.imbalance_spin)
        imbalance_row.addStretch()
        advanced_layout.addLayout(imbalance_row)
        
        # Horizon (trading days)
        horizon_row = QHBoxLayout()
        horizon_row.addWidget(QLabel("Horizon (Trading Days):"))
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(1, 365)
        self.horizon_spin.setValue(30)
        self.horizon_spin.setSuffix(" days")
        self.horizon_spin.setToolTip("Trade horizon in trading days (used to select label column, e.g., label_30d)")
        horizon_row.addWidget(self.horizon_spin)
        horizon_row.addStretch()
        advanced_layout.addLayout(horizon_row)
        
        # Return threshold (for metadata tracking)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Return Threshold (%):"))
        self.return_threshold_spin = QSpinBox()
        self.return_threshold_spin.setRange(0, 100)
        self.return_threshold_spin.setValue(5)
        self.return_threshold_spin.setSuffix("%")
        self.return_threshold_spin.setToolTip("Return threshold used for labeling (for metadata tracking only)")
        threshold_row.addWidget(self.return_threshold_spin)
        threshold_row.addStretch()
        advanced_layout.addLayout(threshold_row)
        
        # Feature set
        featureset_row = QHBoxLayout()
        featureset_row.addWidget(QLabel("Feature Set:"))
        self.featureset_combo = QComboBox()
        self.featureset_combo.setEditable(True)
        self.featureset_combo.setMinimumWidth(200)
        self.featureset_combo.lineEdit().setPlaceholderText("Leave empty for default")
        # Only one feature set now; keep empty for default
        self.featureset_combo.addItems([""])
        featureset_row.addWidget(self.featureset_combo)
        featureset_row.addStretch()
        advanced_layout.addLayout(featureset_row)
        
        # Model output
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Model Output Path:"))
        self.model_output_edit = QLineEdit()
        self.model_output_edit.setPlaceholderText("Leave empty for default naming")
        output_row.addWidget(self.model_output_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_model_output)
        output_row.addWidget(browse_output_btn)
        advanced_layout.addLayout(output_row)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # Feature selection
        feature_row = QHBoxLayout()
        feature_row.addWidget(QLabel("Training Features:"))
        self.feature_selection_btn = QPushButton("Select Features")
        self.feature_selection_btn.clicked.connect(self.open_feature_selection)
        self._update_feature_button_text()
        feature_row.addWidget(self.feature_selection_btn)
        feature_row.addStretch()
        layout.addLayout(feature_row)
        
        # Preset manager
        preset_row = QHBoxLayout()
        self.preset_manager = PresetManagerWidget("training", self)
        self.preset_manager.set_callbacks(self.load_config, self.save_config)
        preset_row.addWidget(self.preset_manager)
        preset_row.addStretch()
        layout.addLayout(preset_row)
        
        # Train button
        self.train_btn = QPushButton("Train Model")
        self.train_btn.setStyleSheet("""
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
        self.train_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setFormat("%p% (%v/%m)")
        layout.addWidget(self.progress_bar)
        
        # Status message
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #b0b0b0; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # Log output
        log_label = QLabel("Output Log")
        log_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 10px;")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setMaximumHeight(300)
        layout.addWidget(self.log_text)
        
        # Set layout on content widget
        content_widget.setLayout(layout)
        
        # Set content widget in scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        # Set main layout
        self.setLayout(main_layout)
    
    def browse_model_output(self):
        """Browse for model output file."""
        models_dir = Path(self.data_service.get_models_dir())
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model As",
            str(models_dir),
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if file_path:
            self.model_output_edit.setText(file_path)
    
    def save_config(self):
        """Save current configuration as a dictionary."""
        return {
            "tune": self.tune_check.isChecked(),
            "n_iter": self.n_iter_spin.value(),
            "cv": self.cv_check.isChecked(),
            "cv_folds": self.cv_folds_spin.value(),
            "fast": self.fast_check.isChecked(),
            "plots": self.plots_check.isChecked(),
            "diagnostics": self.diagnostics_check.isChecked(),
            "no_early_stop": self.no_early_stop_check.isChecked(),
            "imbalance_multiplier": self.imbalance_spin.value(),
            "horizon": self.horizon_spin.value(),
            "return_threshold": self.return_threshold_spin.value(),
            "feature_set": self.featureset_combo.currentText().strip(),
            "model_output": self.model_output_edit.text().strip()
        }
    
    def load_config(self, config: dict):
        """Load configuration from a dictionary."""
        if "tune" in config:
            self.tune_check.setChecked(config["tune"])
        if "n_iter" in config:
            self.n_iter_spin.setValue(config["n_iter"])
        if "cv" in config:
            self.cv_check.setChecked(config["cv"])
        if "cv_folds" in config:
            self.cv_folds_spin.setValue(config["cv_folds"])
        if "fast" in config:
            self.fast_check.setChecked(config["fast"])
        if "plots" in config:
            self.plots_check.setChecked(config["plots"])
        if "diagnostics" in config:
            self.diagnostics_check.setChecked(config["diagnostics"])
        if "no_early_stop" in config:
            self.no_early_stop_check.setChecked(config["no_early_stop"])
        if "imbalance_multiplier" in config:
            self.imbalance_spin.setValue(config["imbalance_multiplier"])
        if "horizon" in config:
            self.horizon_spin.setValue(config["horizon"])
        if "return_threshold" in config:
            self.return_threshold_spin.setValue(config["return_threshold"])
        if "feature_set" in config:
            index = self.featureset_combo.findText(config["feature_set"])
            if index >= 0:
                self.featureset_combo.setCurrentIndex(index)
            else:
                self.featureset_combo.setCurrentText(config["feature_set"])
        if "model_output" in config:
            self.model_output_edit.setText(config["model_output"])
    
    def train_model(self):
        """Start model training."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", "Model training is already in progress.")
            return
        
        # Disable button
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing training...")
        self.status_label.setStyleSheet("color: #ff9800;")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting model training...")
        
        # Calculate total stages (base 7, +1 for tuning, +1 for diagnostics)
        total_stages = 7
        if self.tune_check.isChecked():
            total_stages += 1
        if self.diagnostics_check.isChecked():
            total_stages += 1
        
        self.progress_bar.setRange(0, total_stages)
        self.progress_bar.setValue(0)
        
        # Get parameters
        kwargs = {
            "tune": self.tune_check.isChecked(),
            "n_iter": self.n_iter_spin.value() if self.tune_check.isChecked() else 20,
            "cv": self.cv_check.isChecked(),
            "cv_folds": self.cv_folds_spin.value() if self.cv_check.isChecked() else None,
            "fast": self.fast_check.isChecked(),
            "plots": self.plots_check.isChecked(),
            "diagnostics": self.diagnostics_check.isChecked(),
            "no_early_stop": self.no_early_stop_check.isChecked(),
            "imbalance_multiplier": self.imbalance_spin.value()
        }
        
        # Horizon
        horizon = self.horizon_spin.value()
        if horizon > 0:
            kwargs["horizon"] = horizon
        
        # Return threshold (convert from percentage to decimal)
        return_threshold_pct = self.return_threshold_spin.value()
        if return_threshold_pct > 0:
            kwargs["return_threshold"] = return_threshold_pct / 100.0
        
        # Feature set
        feature_set = self.featureset_combo.currentText().strip()
        if feature_set:
            kwargs["feature_set"] = feature_set
        
        # Model output
        model_output = self.model_output_edit.text().strip()
        if model_output:
            kwargs["model_output"] = model_output
        
        # Log the command being run (for debugging)
        import sys
        from pathlib import Path
        scripts_dir = Path(__file__).parent.parent.parent / "src"
        train_script = scripts_dir / "train_model.py"
        cmd_preview = f"{sys.executable} {train_script}"
        if kwargs.get("tune"):
            cmd_preview += " --tune"
        if kwargs.get("feature_set"):
            cmd_preview += f" --feature-set {kwargs['feature_set']}"
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Command: {cmd_preview}")
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Script exists: {train_script.exists()}")
        
        # Create worker
        self.worker = TrainingWorker(self.service, **kwargs)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.progress.connect(self.on_training_progress)
        self.worker.progress_message.connect(self.on_progress_message)
        self.worker.start()
    
    def on_training_progress(self, current_stage: int, total_stages: int):
        """Handle training progress update."""
        if total_stages > 0:
            self.progress_bar.setRange(0, total_stages)
            self.progress_bar.setValue(current_stage)
    
    def on_progress_message(self, message: str):
        """Handle progress message update."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #ff9800;")
        # Also log progress updates
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
    
    def on_training_finished(self, success: bool, message: str, metrics_dict: dict = None):
        """Handle completion of model training."""
        # Re-enable button
        self.train_btn.setEnabled(True)
        
        # Set progress to 100% if successful
        if success and self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(self.progress_bar.maximum())
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if success:
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: #4caf50;")
            self.log_text.append(f"[{timestamp}] ✓ {message}")
            
            # Save model to registry if we have metrics
            if metrics_dict and metrics_dict.get('model_path'):
                try:
                    # Extract parameters from current UI state
                    # Use horizon from UI (or fallback to extracted from label_col if not set)
                    horizon = self.horizon_spin.value() if self.horizon_spin.value() > 0 else metrics_dict.get('horizon')
                    
                    # Return threshold from UI (convert from percentage to decimal)
                    return_threshold_pct = self.return_threshold_spin.value()
                    return_threshold = return_threshold_pct / 100.0 if return_threshold_pct > 0 else None
                    
                    parameters = {
                        'horizon': horizon,
                        'return_threshold': return_threshold,
                        'feature_set': self.featureset_combo.currentText().strip() or None,
                        'tune': self.tune_check.isChecked(),
                        'cv': self.cv_check.isChecked(),
                        'imbalance_multiplier': self.imbalance_spin.value(),
                        'n_iter': self.n_iter_spin.value() if self.tune_check.isChecked() else None,
                        'cv_folds': self.cv_folds_spin.value() if self.cv_check.isChecked() else None,
                    }
                    
                    # Get feature count from metrics_dict (parsed from training output)
                    feature_count = metrics_dict.get('feature_count')
                    
                    # Count total features from config/features.yaml
                    total_features = self._count_total_features()
                    
                    # Extract training info
                    training_info = {
                        'training_time': metrics_dict.get('training_time'),
                        'feature_count': feature_count,
                        'total_features': total_features,
                    }
                    
                    # Prepare metrics for registry
                    registry_metrics = {
                        'test': metrics_dict.get('test_metrics', {}),
                        'validation': metrics_dict.get('validation_metrics', {})
                    }
                    
                    # Register the model
                    model_id = self.registry.register_model(
                        model_path=metrics_dict['model_path'],
                        metrics=registry_metrics,
                        parameters=parameters,
                        training_info=training_info
                    )
                    
                    self.log_text.append(f"[{timestamp}] ✓ Model registered in registry (ID: {model_id})")
                except Exception as e:
                    self.log_text.append(f"[{timestamp}] ⚠ Failed to register model: {str(e)}")
        else:
            self.status_label.setText(f"Error: {message}")
            self.status_label.setStyleSheet("color: #f44336;")
            self.log_text.append(f"[{timestamp}] ✗ Error: {message}")
            # Show more details in message box
            error_details = message
            if len(error_details) > 500:
                error_details = error_details[:500] + "..."
            QMessageBox.critical(self, "Training Failed", error_details)
    
    def _count_total_features(self) -> int:
        """Count total available features from config/features.yaml."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "features.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                features = cfg.get("features", {})
                # Count all features (enabled or disabled)
                return len([k for k in features.keys() if not k.startswith('#')])
            return 57  # Default fallback (current total)
        except Exception:
            return 57  # Default fallback
    
    def _count_enabled_features(self) -> int:
        """Count enabled features from config/train_features.yaml."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "train_features.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                features = cfg.get("features", {})
                # Count enabled features (value == 1)
                return sum(1 for v in features.values() if v == 1)
            # If train_features.yaml doesn't exist, assume all are enabled
            return self._count_total_features()
        except Exception:
            return self._count_total_features()
    
    def _update_feature_button_text(self):
        """Update the feature selection button text with count."""
        enabled = self._count_enabled_features()
        total = self._count_total_features()
        self.feature_selection_btn.setText(f"Select Features ({enabled} enabled)")
    
    def open_feature_selection(self):
        """Open the feature selection dialog."""
        dialog = FeatureSelectionDialog(self)
        dialog.features_saved.connect(self._update_feature_button_text)
        dialog.exec()

