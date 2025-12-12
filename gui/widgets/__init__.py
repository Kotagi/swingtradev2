"""
GUI Widgets Package

Reusable widgets for the GUI application.
"""

from gui.widgets.preset_manager import PresetManagerWidget
from gui.widgets.chart_widget import (
    ChartWidget, EquityCurveWidget, ReturnsDistributionWidget, PerformanceMetricsWidget
)
from gui.widgets.drag_drop_lineedit import DragDropLineEdit
from gui.widgets.drawdown_chart import DrawdownChartWidget, RollingMetricsWidget

__all__ = [
    'PresetManagerWidget',
    'ChartWidget',
    'EquityCurveWidget',
    'ReturnsDistributionWidget',
    'PerformanceMetricsWidget',
    'DragDropLineEdit',
    'DrawdownChartWidget',
    'RollingMetricsWidget'
]

