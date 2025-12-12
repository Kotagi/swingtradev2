"""
Chart Widget

Embedded matplotlib charts for PyQt6.
"""

try:
    import matplotlib
    matplotlib.use('QtAgg')  # Use Qt backend for embedding
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    FigureCanvas = None
    Figure = None
    plt = None

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
import pandas as pd
import numpy as np


class ChartWidget(QWidget):
    """Base widget for matplotlib charts."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        super().__init__(parent)
        
        if not HAS_MATPLOTLIB:
            layout = QVBoxLayout()
            label = QLabel("Matplotlib not available. Install with: pip install matplotlib")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color: #ff9800; padding: 20px;")
            layout.addWidget(label)
            self.setLayout(layout)
            return
        
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def clear(self):
        """Clear the chart."""
        if HAS_MATPLOTLIB and self.figure:
            self.figure.clear()
            self.canvas.draw()
    
    def draw(self):
        """Redraw the chart."""
        if HAS_MATPLOTLIB and self.canvas:
            self.canvas.draw()


class EquityCurveWidget(ChartWidget):
    """Widget for displaying equity curve from backtest results."""
    
    def plot_equity_curve(self, trades_df: pd.DataFrame, start_value: float = 1.0):
        """
        Plot a normalized equity curve from completed trades.
        
        Behavior:
        - Starts at `start_value` (default 1.0, can be read as 1.0 = 100%).
        - Changes only on trade exit dates.
        - Trades are treated as sequential, equal-sized bets and compounded.
        - Multiple trades exiting on the same date are aggregated multiplicatively.
        - Flat between exit dates.
        
        Args:
            trades_df: DataFrame with columns:
                - return (decimal, e.g., 0.05 for +5%)
                - exit_date (used for timing; falls back to entry_date/index if absent)
            start_value: Normalized starting equity value.
        """
        if not HAS_MATPLOTLIB:
            return
        
        self.clear()
        ax = self.figure.add_subplot(111)
        
        if trades_df.empty:
            ax.text(0.5, 0.5, 'No trades to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.draw()
            return
        
        # Ensure we have returns
        df = trades_df.copy()
        if 'return' not in df.columns:
            # Try to derive returns from pnl if available (best-effort)
            if 'pnl' in df.columns:
                # Approximate position size from median absolute pnl / 5%
                pos_size = max(1e-9, df['pnl'].abs().median() / 0.05) if not df['pnl'].abs().median() == 0 else 1.0
                df['return'] = df['pnl'] / pos_size
            else:
                ax.text(0.5, 0.5, 'No return data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14)
                self.draw()
                return
        
        # Clean returns
        df['return'] = pd.to_numeric(df['return'], errors='coerce').fillna(0.0)
        
        # Use exit_date if present, else entry_date, else index
        if 'exit_date' in df.columns:
            dates = pd.to_datetime(df['exit_date'])
        elif 'entry_date' in df.columns:
            dates = pd.to_datetime(df['entry_date'])
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            # No dates; plot steps over trade index
            dates = pd.RangeIndex(start=0, stop=len(df))
        
        df = df.assign(exit_dt=dates)
        df = df.sort_values('exit_dt')
        
        # Aggregate trades by exit date: compound returns per date
        agg = df.groupby('exit_dt')['return'].apply(lambda r: (1.0 + r).prod()).reset_index()
        agg.rename(columns={'return': 'factor'}, inplace=True)
        
        # Build equity over time (stepwise)
        equity_points = []
        current_equity = start_value
        for _, row in agg.iterrows():
            current_equity *= row['factor']
            equity_points.append((row['exit_dt'], current_equity))
        
        # Create a stepwise series over the full date range
        if isinstance(dates, (pd.DatetimeIndex, pd.Series)) and pd.api.types.is_datetime64_any_dtype(dates):
            full_range = pd.date_range(start=agg['exit_dt'].min(), end=agg['exit_dt'].max(), freq='D')
            equity_series = pd.Series(index=full_range, dtype=float)
            for dt, val in equity_points:
                equity_series.loc[dt] = val
            equity_series = equity_series.ffill().fillna(start_value)
            plot_dates = equity_series.index
            plot_values = equity_series.values
            drawstyle = 'steps-post'
        else:
            # Use trade index if no dates
            plot_dates = [i for i, _ in enumerate(equity_points)]
            plot_values = [v for _, v in equity_points]
            drawstyle = 'default'
        
        ax.plot(plot_dates, plot_values, linewidth=2, color='#00d4aa', label='Equity (normalized)', drawstyle=drawstyle)
        ax.axhline(y=start_value, color='#808080', linestyle='--', alpha=0.5, label='Start')
        ax.set_xlabel('Date' if drawstyle == 'steps-post' else 'Trade Index')
        ax.set_ylabel('Normalized Equity')
        ax.set_title('Equity Curve (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#b0b0b0')
        ax.xaxis.label.set_color('#b0b0b0')
        ax.yaxis.label.set_color('#b0b0b0')
        ax.title.set_color('#00d4aa')
        
        # Rotate x-axis labels if dates
        if drawstyle == 'steps-post':
            self.figure.autofmt_xdate()
        
        self.draw()
    
    def plot_cumulative_pnl(self, trades_df: pd.DataFrame):
        """
        Plot cumulative total return (P&L) over time from completed trades.
        
        This shows the running total of gains/losses from all trades, starting at 0.
        The line goes up when trades are profitable and down when they lose money.
        
        Args:
            trades_df: DataFrame with columns:
                - pnl (profit/loss in dollars)
                - exit_date (date when trade closed)
        """
        if not HAS_MATPLOTLIB:
            return
        
        try:
            self.clear()
            ax = self.figure.add_subplot(111)
            
            if trades_df.empty:
                ax.text(0.5, 0.5, 'No trades to display', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, color='#b0b0b0')
                self.draw()
                return
            
            # Ensure we have P&L data
            df = trades_df.copy()
            if 'pnl' not in df.columns:
                # Try to derive P&L from return if available
                if 'return' in df.columns:
                    # Estimate position size - use a reasonable default
                    position_size = 1000.0  # Default
                    if 'position_size' in df.columns:
                        position_size = df['position_size'].median()
                    elif 'entry_price' in df.columns:
                        # Rough estimate: assume $1000 position
                        position_size = 1000.0
                    else:
                        # If we have return but no entry_price, estimate from return distribution
                        # Assume median return of 5% corresponds to $50 P&L
                        if not df['return'].empty:
                            median_return = df['return'].abs().median()
                            if median_return > 0:
                                position_size = 50.0 / median_return
                    df['pnl'] = df['return'] * position_size
                else:
                    ax.text(0.5, 0.5, f'No P&L data available\nColumns: {", ".join(df.columns)}', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12, color='#ff9800')
                    self.draw()
                    return
            
            # Clean P&L
            df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
        
            # Use exit_date if present, else entry_date, else index
            if 'exit_date' in df.columns:
                dates = pd.to_datetime(df['exit_date'], errors='coerce')
            elif 'entry_date' in df.columns:
                dates = pd.to_datetime(df['entry_date'], errors='coerce')
            elif isinstance(df.index, pd.DatetimeIndex):
                dates = df.index
            else:
                # No dates; use trade index
                dates = pd.RangeIndex(start=0, stop=len(df))
            
            # Filter out rows with invalid dates if using dates
            if pd.api.types.is_datetime64_any_dtype(dates):
                valid_mask = dates.notna()
                df = df[valid_mask].copy()
                dates = dates[valid_mask]
            
            if df.empty:
                ax.text(0.5, 0.5, 'No valid trades with dates', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, color='#ff9800')
                self.draw()
                return
            
            df = df.assign(exit_dt=dates)
            df = df.sort_values('exit_dt')
            
            # Calculate cumulative P&L
            df['cumulative_pnl'] = df['pnl'].cumsum()
            
            # Create a continuous time series for smooth line plotting
            if pd.api.types.is_datetime64_any_dtype(df['exit_dt']):
                # Create full date range from first to last trade
                min_date = df['exit_dt'].min()
                max_date = df['exit_dt'].max()
                
                full_range = pd.date_range(start=min_date, end=max_date, freq='D')
                
                # Create series with cumulative P&L at each exit date
                pnl_series = pd.Series(index=full_range, dtype=float)
                for _, row in df.iterrows():
                    pnl_series.loc[row['exit_dt']] = row['cumulative_pnl']
                
                # Forward fill to show flat lines between trades
                pnl_series = pnl_series.ffill().fillna(0.0)
                
                plot_dates = pnl_series.index
                plot_values = pnl_series.values
            else:
                # Use trade index if no dates
                plot_dates = range(len(df))
                plot_values = df['cumulative_pnl'].values
            
            # Ensure we have valid data to plot
            if len(plot_values) == 0 or all(pd.isna(plot_values)):
                ax.text(0.5, 0.5, 'No valid data to plot', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=14, color='#ff9800')
                self.draw()
                return
            
            # Plot the cumulative P&L line
            ax.plot(plot_dates, plot_values, linewidth=2, color='#00d4aa', label='Cumulative P&L')
            
            # Add zero line
            ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.5, label='Break Even')
            
            # Color the area above/below zero
            ax.fill_between(plot_dates, 0, plot_values, 
                            where=(plot_values >= 0), 
                            color='#00d4aa', alpha=0.2, label='Profit')
            ax.fill_between(plot_dates, 0, plot_values, 
                            where=(plot_values < 0), 
                            color='#f44336', alpha=0.2, label='Loss')
            
            # Check if we're using dates
            is_datetime = isinstance(plot_dates, pd.DatetimeIndex) or (
                hasattr(plot_dates, 'dtype') and pd.api.types.is_datetime64_any_dtype(plot_dates)
            )
            
            ax.set_xlabel('Date' if is_datetime else 'Trade Index')
            ax.set_ylabel('Cumulative P&L ($)')
            ax.set_title('Cumulative Total Return Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#1e1e1e')
            self.figure.patch.set_facecolor('#1e1e1e')
            ax.tick_params(colors='#b0b0b0')
            ax.xaxis.label.set_color('#b0b0b0')
            ax.yaxis.label.set_color('#b0b0b0')
            ax.title.set_color('#00d4aa')
            
            # Format y-axis as currency
            if plt:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Rotate x-axis labels if dates
            if is_datetime:
                self.figure.autofmt_xdate()
            
            self.draw()
        except Exception as e:
            import traceback
            error_msg = f"Error plotting cumulative P&L: {str(e)}"
            self.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error plotting chart:\n{str(e)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='#f44336')
            self.draw()
            # Re-raise to be caught by caller for logging
            raise


class ReturnsDistributionWidget(ChartWidget):
    """Widget for displaying returns distribution."""
    
    def plot_returns_distribution(self, trades_df: pd.DataFrame):
        """
        Plot returns distribution histogram.
        
        Args:
            trades_df: DataFrame with 'return' or 'pnl' column
        """
        if not HAS_MATPLOTLIB:
            return
        
        self.clear()
        ax = self.figure.add_subplot(111)
        
        if trades_df.empty:
            ax.text(0.5, 0.5, 'No trades to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.draw()
            return
        
        # Get returns
        if 'return' in trades_df.columns:
            returns = trades_df['return'] * 100  # Convert to percentage
            xlabel = 'Return (%)'
        elif 'pnl' in trades_df.columns:
            # Convert P&L to approximate return (assuming position_size)
            position_size = trades_df['pnl'].abs().median() / 0.05  # Rough estimate
            returns = (trades_df['pnl'] / position_size) * 100
            xlabel = 'Return (%)'
        else:
            ax.text(0.5, 0.5, 'No return or P&L data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.draw()
            return
        
        # Plot histogram
        ax.hist(returns, bins=30, color='#00d4aa', alpha=0.7, edgecolor='#00a080')
        ax.axvline(x=0, color='#ff9800', linestyle='--', linewidth=2, label='Break Even')
        ax.axvline(x=returns.mean(), color='#4caf50', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.set_title('Returns Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#b0b0b0')
        ax.xaxis.label.set_color('#b0b0b0')
        ax.yaxis.label.set_color('#b0b0b0')
        ax.title.set_color('#00d4aa')
        
        self.draw()


class PerformanceMetricsWidget(ChartWidget):
    """Widget for displaying performance metrics as bar chart."""
    
    def plot_metrics(self, metrics: dict):
        """
        Plot performance metrics as bar chart.
        
        Args:
            metrics: Dictionary with metric names and values
        """
        if not HAS_MATPLOTLIB:
            return
        
        self.clear()
        ax = self.figure.add_subplot(111)
        
        if not metrics:
            ax.text(0.5, 0.5, 'No metrics to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.draw()
            return
        
        # Filter and format metrics
        display_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                # Format key for display
                display_key = key.replace('_', ' ').title()
                display_metrics[display_key] = value
        
        if not display_metrics:
            ax.text(0.5, 0.5, 'No valid metrics to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.draw()
            return
        
        # Create bar chart
        keys = list(display_metrics.keys())
        values = list(display_metrics.values())
        
        colors = ['#00d4aa' if v >= 0 else '#f44336' for v in values]
        bars = ax.barh(keys, values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (key, value) in enumerate(display_metrics.items()):
            ax.text(value, i, f' {value:.2f}', va='center', color='#b0b0b0')
        
        ax.set_xlabel('Value')
        ax.set_title('Performance Metrics')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#b0b0b0')
        ax.xaxis.label.set_color('#b0b0b0')
        ax.yaxis.label.set_color('#b0b0b0')
        ax.title.set_color('#00d4aa')
        
        self.figure.tight_layout()
        self.draw()

