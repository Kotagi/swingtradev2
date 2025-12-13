"""
Metrics Help Dialog

Explains what each model metric means and how to interpret them.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit
)
from PyQt6.QtCore import Qt


class MetricsHelpDialog(QDialog):
    """Dialog explaining model metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Metrics Help")
        self.setMinimumSize(700, 600)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Understanding Model Metrics")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00d4aa; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Help text
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #555;
                border-radius: 5px;
                color: #e0e0e0;
                font-size: 13px;
                padding: 10px;
            }
        """)
        
        help_content = """
<h2 style="color: #00d4aa;">Model Performance Metrics Explained</h2>

<p>These metrics help you evaluate how well your trained model performs at predicting profitable trading opportunities.</p>

<h3 style="color: #00d4aa;">📊 ROC AUC (Receiver Operating Characteristic - Area Under Curve)</h3>
<p><b>Range:</b> 0.0 to 1.0 | <b>Higher is better</b></p>
<p><b>What it measures:</b> The model's ability to distinguish between profitable trades (positive class) and unprofitable trades (negative class).</p>
<p><b>Interpretation:</b></p>
<ul>
    <li><b>0.9 - 1.0:</b> Excellent - Model is very good at separating profitable from unprofitable trades</li>
    <li><b>0.8 - 0.9:</b> Good - Model performs well</li>
    <li><b>0.7 - 0.8:</b> Fair - Model has some predictive power</li>
    <li><b>0.6 - 0.7:</b> Poor - Model is barely better than random guessing</li>
    <li><b>0.5:</b> Random - Model is no better than flipping a coin</li>
    <li><b>&lt; 0.5:</b> Worse than random - Model is performing worse than chance</li>
</ul>
<p><b>Why it matters:</b> ROC AUC is the most important metric for binary classification. It tells you how well your model can rank opportunities from most to least promising.</p>

<h3 style="color: #00d4aa;">🎯 Accuracy</h3>
<p><b>Range:</b> 0.0 to 1.0 | <b>Higher is better</b></p>
<p><b>What it measures:</b> The percentage of predictions that are correct (both profitable and unprofitable trades).</p>
<p><b>Interpretation:</b></p>
<ul>
    <li><b>0.9+:</b> Excellent - Model is correct 90%+ of the time</li>
    <li><b>0.7 - 0.9:</b> Good - Model is correct most of the time</li>
    <li><b>0.5 - 0.7:</b> Fair - Model is better than random</li>
    <li><b>&lt; 0.5:</b> Poor - Model is worse than random</li>
</ul>
<p><b>Note:</b> Accuracy can be misleading if your data is imbalanced (e.g., 90% of trades are unprofitable). A model that always predicts "unprofitable" would have 90% accuracy but be useless.</p>

<h3 style="color: #00d4aa;">🎪 Precision</h3>
<p><b>Range:</b> 0.0 to 1.0 | <b>Higher is better</b></p>
<p><b>What it measures:</b> Of all the trades the model predicts as profitable, what percentage actually were profitable?</p>
<p><b>Formula:</b> True Positives / (True Positives + False Positives)</p>
<p><b>Interpretation:</b></p>
<ul>
    <li><b>0.8+:</b> Excellent - 80%+ of predicted profitable trades actually are profitable</li>
    <li><b>0.6 - 0.8:</b> Good - Most predicted profitable trades are correct</li>
    <li><b>0.4 - 0.6:</b> Fair - About half of predicted profitable trades are correct</li>
    <li><b>&lt; 0.4:</b> Poor - Most predicted profitable trades are actually unprofitable</li>
</ul>
<p><b>Why it matters:</b> High precision means when your model says "this is a good trade," it's usually right. This reduces wasted capital on bad trades.</p>

<h3 style="color: #00d4aa;">🔍 Recall (Sensitivity)</h3>
<p><b>Range:</b> 0.0 to 1.0 | <b>Higher is better</b></p>
<p><b>What it measures:</b> Of all the actually profitable trades, what percentage did the model correctly identify?</p>
<p><b>Formula:</b> True Positives / (True Positives + False Negatives)</p>
<p><b>Interpretation:</b></p>
<ul>
    <li><b>0.8+:</b> Excellent - Model catches 80%+ of profitable opportunities</li>
    <li><b>0.6 - 0.8:</b> Good - Model catches most profitable opportunities</li>
    <li><b>0.4 - 0.6:</b> Fair - Model catches about half of profitable opportunities</li>
    <li><b>&lt; 0.4:</b> Poor - Model misses most profitable opportunities</li>
</ul>
<p><b>Why it matters:</b> High recall means you're not missing many good trading opportunities. Low recall means you're leaving money on the table.</p>

<h3 style="color: #00d4aa;">⚖️ F1 Score</h3>
<p><b>Range:</b> 0.0 to 1.0 | <b>Higher is better</b></p>
<p><b>What it measures:</b> The harmonic mean of Precision and Recall. It balances both metrics.</p>
<p><b>Formula:</b> 2 × (Precision × Recall) / (Precision + Recall)</p>
<p><b>Interpretation:</b></p>
<ul>
    <li><b>0.8+:</b> Excellent - Model has both high precision and recall</li>
    <li><b>0.6 - 0.8:</b> Good - Model balances precision and recall well</li>
    <li><b>0.4 - 0.6:</b> Fair - Model has moderate precision and recall</li>
    <li><b>&lt; 0.4:</b> Poor - Model struggles with both precision and recall</li>
</ul>
<p><b>Why it matters:</b> F1 Score gives you a single number that considers both how accurate your predictions are (precision) and how many opportunities you catch (recall). It's useful when you need to balance both concerns.</p>

<h3 style="color: #00d4aa;">📈 Average Precision (AP)</h3>
<p><b>Range:</b> 0.0 to 1.0 | <b>Higher is better</b></p>
<p><b>What it measures:</b> The area under the Precision-Recall curve. Similar to ROC AUC but focuses on precision-recall tradeoff.</p>
<p><b>Interpretation:</b></p>
<ul>
    <li><b>0.8+:</b> Excellent - Model maintains high precision across different recall levels</li>
    <li><b>0.6 - 0.8:</b> Good - Model has good precision-recall balance</li>
    <li><b>0.4 - 0.6:</b> Fair - Model has moderate precision-recall performance</li>
    <li><b>&lt; 0.4:</b> Poor - Model struggles with precision-recall tradeoff</li>
</ul>
<p><b>Why it matters:</b> Average Precision is especially useful when your data is imbalanced (many more unprofitable than profitable trades). It focuses on how well the model performs on the positive class (profitable trades).</p>

<h2 style="color: #00d4aa;">💡 Tips for Model Comparison</h2>
<ul>
    <li><b>ROC AUC</b> is generally the most important metric - prioritize models with higher ROC AUC</li>
    <li>If you want to minimize bad trades, focus on <b>Precision</b></li>
    <li>If you want to catch more opportunities, focus on <b>Recall</b></li>
    <li><b>F1 Score</b> is good when you need to balance precision and recall</li>
    <li>For imbalanced datasets, <b>Average Precision</b> can be more informative than accuracy</li>
    <li>Remember: These are test set metrics. A model that performs well on test data should perform well on new, unseen data</li>
</ul>
        """
        
        help_text.setHtml(help_content)
        layout.addWidget(help_text)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4aa;
                color: #000000;
                font-weight: bold;
                padding: 8px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00c896;
            }
        """)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

