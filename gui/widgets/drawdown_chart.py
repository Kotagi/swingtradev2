"""
Drawdown Chart Widget

Displays drawdown visualization from backtest results.
"""

try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    FigureCanvas = None
    Figure = None

from gui.widgets.chart_widget import ChartWidget
import pandas as pd
import numpy as np


class DrawdownChartWidget(ChartWidget):
    """Widget for displaying drawdown chart."""
    
    def plot_drawdown(self, trades_df: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Plot drawdown chart from trades DataFrame.
        
        Args:
            trades_df: DataFrame with columns: entry_date, pnl (or return)
            initial_capital: Starting capital
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
        
        # Calculate equity curve
        if 'pnl' in trades_df.columns:
            cumulative_pnl = trades_df['pnl'].cumsum()
            equity = initial_capital + cumulative_pnl
        elif 'return' in trades_df.columns:
            position_size = initial_capital / 10
            pnl = trades_df['return'] * position_size
            cumulative_pnl = pnl.cumsum()
            equity = initial_capital + cumulative_pnl
        else:
            ax.text(0.5, 0.5, 'No P&L or return data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14)
            self.draw()
            return
        
        # Get dates
        if 'entry_date' in trades_df.columns:
            dates = pd.to_datetime(trades_df['entry_date'])
        elif trades_df.index.name == 'entry_date' or isinstance(trades_df.index, pd.DatetimeIndex):
            dates = trades_df.index
        else:
            dates = range(len(trades_df))
        
        # Calculate drawdown
        running_max = equity.cummax()
        drawdown = ((equity - running_max) / running_max) * 100  # Percentage drawdown
        
        # Plot
        ax.fill_between(dates, drawdown, 0, color='#f44336', alpha=0.3, label='Drawdown')
        ax.plot(dates, drawdown, linewidth=1.5, color='#f44336', label='Drawdown %')
        ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.5)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Chart')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#b0b0b0')
        ax.xaxis.label.set_color('#b0b0b0')
        ax.yaxis.label.set_color('#b0b0b0')
        ax.title.set_color('#00d4aa')
        
        # Rotate x-axis labels if dates
        if isinstance(dates, pd.DatetimeIndex):
            self.figure.autofmt_xdate()
        
        self.draw()


class RollingMetricsWidget(ChartWidget):
    """Widget for displaying rolling performance metrics."""
    
    def plot_rolling_metrics(self, trades_df: pd.DataFrame, window: int = 20):
        """
        Plot rolling Sharpe ratio and win rate.
        
        Args:
            trades_df: DataFrame with 'return' and 'pnl' columns
            window: Rolling window size
        """
        if not HAS_MATPLOTLIB:
            return
        
        self.clear()
        fig = self.figure
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        if trades_df.empty:
            ax1.text(0.5, 0.5, 'No trades to display', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14)
            self.draw()
            return
        
        # Get dates
        if 'entry_date' in trades_df.columns:
            dates = pd.to_datetime(trades_df['entry_date'])
        elif trades_df.index.name == 'entry_date' or isinstance(trades_df.index, pd.DatetimeIndex):
            dates = trades_df.index
        else:
            dates = range(len(trades_df))
        
        # Calculate rolling metrics
        if 'return' in trades_df.columns:
            returns = trades_df['return']
        elif 'pnl' in trades_df.columns:
            # Approximate returns from P&L
            position_size = trades_df['pnl'].abs().median() / 0.05
            returns = trades_df['pnl'] / position_size
        else:
            ax1.text(0.5, 0.5, 'No return data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14)
            self.draw()
            return
        
        # Rolling Sharpe (simplified)
        rolling_mean = returns.rolling(window=window).mean()
        rolling_std = returns.rolling(window=window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252) if rolling_std.std() > 0 else pd.Series(0, index=returns.index)
        
        # Rolling win rate
        wins = (returns > 0).astype(int)
        rolling_win_rate = wins.rolling(window=window).mean() * 100
        
        # Plot rolling Sharpe
        ax1.plot(dates, rolling_sharpe, linewidth=2, color='#00d4aa', label=f'Rolling Sharpe ({window} trades)')
        ax1.axhline(y=0, color='#808080', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Rolling Performance Metrics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#1e1e1e')
        
        # Plot rolling win rate
        ax2.plot(dates, rolling_win_rate, linewidth=2, color='#4caf50', label=f'Rolling Win Rate ({window} trades)')
        ax2.axhline(y=50, color='#808080', linestyle='--', alpha=0.5, label='50%')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#1e1e1e')
        
        fig.patch.set_facecolor('#1e1e1e')
        for ax in [ax1, ax2]:
            ax.tick_params(colors='#b0b0b0')
            ax.xaxis.label.set_color('#b0b0b0')
            ax.yaxis.label.set_color('#b0b0b0')
        ax1.title.set_color('#00d4aa')
        
        # Rotate x-axis labels if dates
        if isinstance(dates, pd.DatetimeIndex):
            self.figure.autofmt_xdate()
        
        self.figure.tight_layout()
        self.draw()

