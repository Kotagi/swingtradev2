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
        
        Shows drawdown (peak-to-trough decline) from the cumulative P&L curve,
        matching the cumulative P&L chart's timeline (sorted by exit_date).
        
        Args:
            trades_df: DataFrame with columns: exit_date (or entry_date), pnl (or return)
            initial_capital: Not used (kept for compatibility), drawdown calculated from $0 start
        """
        if not HAS_MATPLOTLIB:
            return
        
        self.clear()
        ax = self.figure.add_subplot(111)
        
        if trades_df.empty:
            ax.text(0.5, 0.5, 'No trades to display', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='#b0b0b0')
            self.draw()
            return
        
        # Prepare data - sort by exit_date to match cumulative P&L chart
        df = trades_df.copy()
        
        # Get dates - use exit_date (when P&L is realized), fall back to entry_date
        if 'exit_date' in df.columns:
            dates = pd.to_datetime(df['exit_date'], errors='coerce')
        elif 'entry_date' in df.columns:
            dates = pd.to_datetime(df['entry_date'], errors='coerce')
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            dates = pd.RangeIndex(start=0, stop=len(df))
        
        # Filter out invalid dates
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
        
        # Sort by date to ensure chronological order (matching cumulative P&L chart)
        df = df.assign(exit_dt=dates)
        df = df.sort_values('exit_dt')
        dates = df['exit_dt']
        
        # Calculate cumulative P&L (starting from $0, matching cumulative P&L chart)
        if 'pnl' in df.columns:
            df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
            cumulative_pnl = df['pnl'].cumsum()
        elif 'return' in df.columns:
            # Estimate position size to derive P&L
            position_size = 1000.0  # Default
            if 'position_size' in df.columns:
                position_size = df['position_size'].median()
            df['return'] = pd.to_numeric(df['return'], errors='coerce').fillna(0.0)
            cumulative_pnl = (df['return'] * position_size).cumsum()
        else:
            ax.text(0.5, 0.5, 'No P&L or return data available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='#ff9800')
            self.draw()
            return
        
        # Calculate drawdown from cumulative P&L curve
        # Running peak (highest cumulative P&L reached so far)
        running_max = cumulative_pnl.cummax()
        # Drawdown = peak - current (always >= 0, shows how far below peak we are)
        drawdown = running_max - cumulative_pnl
        
        # Create continuous time series for smooth plotting (matching cumulative P&L chart)
        if pd.api.types.is_datetime64_any_dtype(dates):
            min_date = dates.min()
            max_date = dates.max()
            full_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # Create series with drawdown at each exit date
            drawdown_series = pd.Series(index=full_range, dtype=float)
            for i, (dt, dd) in enumerate(zip(dates, drawdown)):
                drawdown_series.loc[dt] = dd
            
            # Forward fill to show flat lines between trades
            drawdown_series = drawdown_series.ffill().fillna(0.0)
            
            plot_dates = drawdown_series.index
            plot_values = drawdown_series.values
        else:
            plot_dates = range(len(df))
            plot_values = drawdown.values
        
        # Plot drawdown (shaded area below zero line)
        ax.fill_between(plot_dates, 0, plot_values, color='#f44336', alpha=0.3, label='Drawdown')
        ax.plot(plot_dates, plot_values, linewidth=1.5, color='#f44336', label='Drawdown ($)')
        ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.5, label='No Drawdown')
        ax.set_xlabel('Date' if pd.api.types.is_datetime64_any_dtype(plot_dates) else 'Trade Index')
        ax.set_ylabel('Drawdown ($)')
        ax.set_title('Drawdown Chart (Peak-to-Trough)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#b0b0b0')
        ax.xaxis.label.set_color('#b0b0b0')
        ax.yaxis.label.set_color('#b0b0b0')
        ax.title.set_color('#00d4aa')
        
        # Format y-axis as currency
        try:
            import matplotlib.pyplot as plt
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        except:
            pass
        
        # Rotate x-axis labels if dates
        if pd.api.types.is_datetime64_any_dtype(plot_dates):
            self.figure.autofmt_xdate()
        
        self.draw()


class RollingMetricsWidget(ChartWidget):
    """Widget for displaying rolling performance metrics."""
    
    def plot_rolling_metrics(self, trades_df: pd.DataFrame, window: int = 20):
        """
        Plot rolling Sharpe ratio and win rate over time.
        
        Shows rolling metrics calculated over the last N trades (chronologically by exit date).
        This helps identify periods of strong/weak performance.
        
        Args:
            trades_df: DataFrame with 'exit_date' (or 'entry_date'), 'return' (or 'pnl') columns
            window: Rolling window size (number of trades)
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
                    transform=ax1.transAxes, fontsize=14, color='#b0b0b0')
            self.draw()
            return
        
        # Prepare data - sort by exit_date to ensure chronological order
        df = trades_df.copy()
        
        # Get dates - use exit_date (when trade completes), fall back to entry_date
        if 'exit_date' in df.columns:
            dates = pd.to_datetime(df['exit_date'], errors='coerce')
        elif 'entry_date' in df.columns:
            dates = pd.to_datetime(df['entry_date'], errors='coerce')
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index
        else:
            dates = pd.RangeIndex(start=0, stop=len(df))
        
        # Filter out invalid dates
        if pd.api.types.is_datetime64_any_dtype(dates):
            valid_mask = dates.notna()
            df = df[valid_mask].copy()
            dates = dates[valid_mask]
        
        if df.empty:
            ax1.text(0.5, 0.5, 'No valid trades with dates', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14, color='#ff9800')
            self.draw()
            return
        
        # Sort by date to ensure chronological order (critical for rolling calculations)
        df = df.assign(exit_dt=dates)
        df = df.sort_values('exit_dt')
        dates = df['exit_dt']
        
        # Get returns - must be sorted chronologically
        if 'return' in df.columns:
            df['return'] = pd.to_numeric(df['return'], errors='coerce').fillna(0.0)
            returns = df['return']
        elif 'pnl' in df.columns:
            # Approximate returns from P&L
            df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
            position_size = df['pnl'].abs().median()
            if position_size <= 0:
                position_size = max(1e-9, df['pnl'].abs().mean())
            if position_size <= 0:
                position_size = 1000.0  # Default fallback
            returns = df['pnl'] / position_size
        else:
            ax1.text(0.5, 0.5, 'No return data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=14, color='#ff9800')
            self.draw()
            return
        
        # Calculate rolling metrics (now on chronologically sorted data)
        # Rolling Sharpe ratio: (mean return / std return) * sqrt(252) annualized
        rolling_mean = returns.rolling(window=window, min_periods=1).mean()
        rolling_std = returns.rolling(window=window, min_periods=1).std()
        # Only calculate Sharpe when we have valid std (avoid division by zero)
        rolling_sharpe = pd.Series(0.0, index=returns.index)
        valid_sharpe_mask = rolling_std > 0
        rolling_sharpe[valid_sharpe_mask] = (rolling_mean[valid_sharpe_mask] / rolling_std[valid_sharpe_mask]) * np.sqrt(252)
        
        # Rolling win rate: percentage of winning trades in the window
        wins = (returns > 0).astype(int)
        rolling_win_rate = wins.rolling(window=window, min_periods=1).mean() * 100
        
        # Plot rolling Sharpe
        ax1.plot(dates, rolling_sharpe, linewidth=2, color='#00d4aa', label=f'Rolling Sharpe ({window} trades)')
        ax1.axhline(y=0, color='#808080', linestyle='--', alpha=0.5)
        ax1.axhline(y=1, color='#4caf50', linestyle='--', alpha=0.3, label='Sharpe = 1')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Rolling Performance Metrics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#1e1e1e')
        
        # Plot rolling win rate
        ax2.plot(dates, rolling_win_rate, linewidth=2, color='#4caf50', label=f'Rolling Win Rate ({window} trades)')
        ax2.axhline(y=50, color='#808080', linestyle='--', alpha=0.5, label='50%')
        ax2.set_xlabel('Date' if pd.api.types.is_datetime64_any_dtype(dates) else 'Trade Index')
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
        if pd.api.types.is_datetime64_any_dtype(dates):
            self.figure.autofmt_xdate()
        
        self.figure.tight_layout()
        self.draw()

