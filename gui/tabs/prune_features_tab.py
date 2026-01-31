"""
Prune Features Tab

Disable low/zero-importance features in build and training configs.
Lives under Analysis tab.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox, QFileDialog,
    QStackedWidget, QFormLayout, QSizePolicy,
)
from PyQt6.QtCore import Qt
from pathlib import Path
from typing import List, Optional

from gui.services import PruneService
from gui.utils.model_registry import ModelRegistry

PROJECT_ROOT = Path(__file__).parent.parent.parent


class PruneFeaturesTab(QWidget):
    """Tab for pruning low/zero-importance features from configs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = PruneService()
        self.preview_to_disable: List[str] = []
        self.preview_feature_set_used: str = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)

        title = QLabel("Prune features (disable low/zero importance)")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00d4aa;")
        layout.addWidget(title)

        # Source: model
        source_group = QGroupBox("Source model")
        source_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(400)
        self.model_combo.setToolTip("Select a trained model; its feature importances will be used.")
        self._populate_models()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_model)
        source_row = QHBoxLayout()
        source_row.addWidget(self.model_combo)
        source_row.addWidget(self.browse_btn)
        source_layout.addRow("Model:", source_row)
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        # Feature set
        fs_group = QGroupBox("Feature set")
        fs_layout = QFormLayout()
        self.feature_set_combo = QComboBox()
        self._populate_feature_sets()
        fs_layout.addRow("Feature set:", self.feature_set_combo)
        fs_group.setLayout(fs_layout)
        layout.addWidget(fs_group)

        # Features ranked by importance (when model selected)
        importance_group = QGroupBox("Features by importance")
        importance_layout = QVBoxLayout()
        self.importance_text = QTextEdit()
        self.importance_text.setReadOnly(True)
        self.importance_text.setPlaceholderText("Select a model above to list features ranked by importance (full values).")
        self.importance_text.setMinimumHeight(180)
        self.importance_text.setMaximumHeight(280)
        importance_layout.addWidget(self.importance_text)
        importance_group.setLayout(importance_layout)
        layout.addWidget(importance_group)

        # Rule
        rule_group = QGroupBox("Prune rule")
        rule_layout = QFormLayout()
        self.rule_combo = QComboBox()
        self.rule_combo.addItem("Drop zero importance", "drop_zero")
        self.rule_combo.addItem("Keep top N features", "keep_top_n")
        self.rule_combo.addItem("Drop below threshold", "drop_below_threshold")
        self.rule_combo.currentIndexChanged.connect(self._on_rule_changed)
        rule_layout.addRow("Rule:", self.rule_combo)
        self.param_stack = QStackedWidget()
        self.param_stack.setMaximumWidth(140)
        self.param_stack.setMaximumHeight(32)
        self.param_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.param_n = QSpinBox()
        self.param_n.setRange(1, 10000)
        self.param_n.setValue(200)
        self.param_n.setMaximumWidth(120)
        self.param_n.setMaximumHeight(28)
        self.param_n.setToolTip("Keep only the top N features by importance; disable the rest.")
        self.param_stack.addWidget(self.param_n)
        self.param_threshold = QDoubleSpinBox()
        self.param_threshold.setRange(0.0, 1.0)
        self.param_threshold.setDecimals(6)
        self.param_threshold.setSingleStep(0.001)
        self.param_threshold.setValue(0.001)
        self.param_threshold.setMaximumWidth(120)
        self.param_threshold.setMaximumHeight(28)
        self.param_threshold.setToolTip("Disable features with importance below this value.")
        self.param_stack.addWidget(self.param_threshold)
        self.param_placeholder = QWidget()
        self.param_placeholder.setMaximumHeight(28)
        self.param_stack.addWidget(self.param_placeholder)
        rule_layout.addRow("Parameter:", self.param_stack)
        rule_group.setLayout(rule_layout)
        layout.addWidget(rule_group)
        self._on_rule_changed()

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self._preview)
        preview_layout.addWidget(self.preview_btn)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("Click Preview to see which features would be disabled.")
        self.preview_text.setMinimumHeight(120)
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Apply
        apply_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply pruning")
        self.apply_btn.clicked.connect(self._apply)
        self.apply_btn.setToolTip("Set the previewed features to 0 in build and train configs.")
        apply_layout.addWidget(self.apply_btn)
        apply_layout.addStretch()
        layout.addLayout(apply_layout)

        layout.addStretch()
        self.setLayout(layout)
        # Connect after importance_text exists; then populate if a model is selected
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._on_model_changed()

    def _on_model_changed(self):
        """Load feature importances for the selected model, set feature set from model, and list features ranked by importance."""
        path = self.model_combo.currentData()
        if not path or not Path(path).exists():
            self.importance_text.clear()
            self.importance_text.setPlaceholderText("Select a model above to list features ranked by importance (full values).")
            return
        importances, feature_set = self.service.get_importances_from_model(path)
        # Auto-set feature set combo to the model's feature set when available
        if feature_set and self.feature_set_combo.count() > 0:
            for i in range(self.feature_set_combo.count()):
                if self.feature_set_combo.itemData(i) == feature_set:
                    self.feature_set_combo.setCurrentIndex(i)
                    break
        # Update features-by-importance list (full numbers)
        if not importances:
            self.importance_text.setPlainText("No feature importances in this model.")
            return
        items = sorted(importances.items(), key=lambda x: -x[1])
        lines = []
        for i, (name, imp) in enumerate(items, 1):
            lines.append(f"{i}\t{name}\t{repr(imp)}")
        self.importance_text.setPlainText("\n".join(lines))

    def _populate_models(self):
        self.model_combo.clear()
        try:
            registry = ModelRegistry()
            models = registry.get_all_models()
            for m in models:
                name = m.get("name", "?")
                path = m.get("file_path", "")
                if path and Path(path).exists():
                    self.model_combo.addItem(name, path)
            if self.model_combo.count() == 0:
                self.model_combo.addItem("(No models in registry)", "")
        except Exception:
            self.model_combo.addItem("(Load failed)", "")

    def _populate_feature_sets(self):
        self.feature_set_combo.clear()
        try:
            from feature_set_manager import list_feature_sets
            for fs in list_feature_sets():
                self.feature_set_combo.addItem(fs, fs)
            if self.feature_set_combo.count() == 0:
                self.feature_set_combo.addItem("(None)", "")
        except Exception:
            self.feature_set_combo.addItem("(Load failed)", "")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model",
            str(PROJECT_ROOT / "models"),
            "Pickle (*.pkl);;All (*)",
        )
        if path:
            self.model_combo.insertItem(0, Path(path).name, path)
            self.model_combo.setCurrentIndex(0)

    def _on_rule_changed(self):
        idx = self.rule_combo.currentData()
        if idx == "keep_top_n":
            self.param_stack.setCurrentWidget(self.param_n)
        elif idx == "drop_below_threshold":
            self.param_stack.setCurrentWidget(self.param_threshold)
        else:
            self.param_stack.setCurrentWidget(self.param_placeholder)

    def _get_model_path(self) -> Optional[str]:
        path = self.model_combo.currentData()
        if path and Path(path).exists():
            return path
        return None

    def _get_feature_set(self) -> str:
        fs = self.feature_set_combo.currentData()
        return fs or ""

    def _preview(self):
        model_path = self._get_model_path()
        if not model_path:
            QMessageBox.warning(self, "No model", "Select a model (or browse to a .pkl file).")
            return
        feature_set = self._get_feature_set()
        rule = self.rule_combo.currentData()
        if rule == "keep_top_n":
            param = float(self.param_n.value())
        elif rule == "drop_below_threshold":
            param = self.param_threshold.value()
        else:
            param = 0.0
        to_disable, msg, feature_set_used = self.service.preview_prune(feature_set, model_path, rule, param)
        self.preview_to_disable = to_disable
        self.preview_feature_set_used = feature_set_used or feature_set
        lines = [msg, ""]
        if feature_set_used:
            lines.append(f"Feature set: {feature_set_used}")
            lines.append("")
        if to_disable:
            lines.append(f"Features to disable ({len(to_disable)}):")
            for n in to_disable[:50]:
                lines.append(f"  {n}")
            if len(to_disable) > 50:
                lines.append(f"  ... and {len(to_disable) - 50} more")
        self.preview_text.setText("\n".join(lines))

    def _apply(self):
        if not self.preview_to_disable:
            QMessageBox.information(
                self,
                "Nothing to apply",
                "Run Preview first to see which features would be disabled, then click Apply.",
            )
            return
        feature_set = self._get_feature_set() or self.preview_feature_set_used
        if not feature_set:
            QMessageBox.warning(self, "No feature set", "Select a feature set (or run Preview with a model that has feature_set in metadata).")
            return
        ok = QMessageBox.question(
            self,
            "Apply pruning",
            f"Disable {len(self.preview_to_disable)} feature(s) in build and train configs for '{feature_set}'?\n\n"
            "Next: rebuild features and retrain to use the pruned set.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) == QMessageBox.StandardButton.Ok
        if not ok:
            return
        count, message = self.service.apply_prune(feature_set, self.preview_to_disable, dry_run=False)
        QMessageBox.information(self, "Prune result", message)
        if count > 0:
            self.preview_text.append("\n" + message)
            self.preview_to_disable = []
