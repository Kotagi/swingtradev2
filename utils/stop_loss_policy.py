#!/usr/bin/env python3
"""
stop_loss_policy.py

Centralized stop-loss policy module for adaptive ATR-based stop losses.

This module provides a single source of truth for stop-loss calculations,
supporting both constant and adaptive ATR-based stop losses.
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class StopLossConfig:
    """
    Configuration for stop-loss behavior.
    
    Attributes:
        mode: "constant", "adaptive_atr", or "swing_atr"
        constant_stop_loss_pct: Fixed stop percentage (e.g., 0.075 for 7.5%)
        atr_stop_k: Multiplier for ATR-based stops (e.g., 1.8)
        atr_stop_min_pct: Minimum stop distance (e.g., 0.04 = 4%)
        atr_stop_max_pct: Maximum stop distance (e.g., 0.10 = 10%)
        swing_lookback_days: Days to look back for swing low (e.g., 10)
        swing_atr_buffer_k: ATR multiplier for swing_atr buffer (e.g., 0.75)
    """
    
    def __init__(
        self,
        mode: str = "constant",
        constant_stop_loss_pct: float = 0.075,
        atr_stop_k: float = 1.8,
        atr_stop_min_pct: float = 0.04,
        atr_stop_max_pct: float = 0.10,
        swing_lookback_days: int = 10,
        swing_atr_buffer_k: float = 0.75
    ):
        if mode not in ["constant", "adaptive_atr", "swing_atr"]:
            raise ValueError(f"Invalid stop_loss_mode: {mode}. Must be 'constant', 'adaptive_atr', or 'swing_atr'")
        
        self.mode = mode
        self.constant_stop_loss_pct = constant_stop_loss_pct
        self.atr_stop_k = atr_stop_k
        self.atr_stop_min_pct = atr_stop_min_pct
        self.atr_stop_max_pct = atr_stop_max_pct
        self.swing_lookback_days = swing_lookback_days
        self.swing_atr_buffer_k = swing_atr_buffer_k
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'StopLossConfig':
        """Create StopLossConfig from a dictionary."""
        return cls(
            mode=config_dict.get("stop_loss_mode", "constant"),
            constant_stop_loss_pct=config_dict.get("constant_stop_loss_pct", 0.075),
            atr_stop_k=config_dict.get("atr_stop_k", 1.8),
            atr_stop_min_pct=config_dict.get("atr_stop_min_pct", 0.04),
            atr_stop_max_pct=config_dict.get("atr_stop_max_pct", 0.10),
            swing_lookback_days=config_dict.get("swing_lookback_days", 10),
            swing_atr_buffer_k=config_dict.get("swing_atr_buffer_k", 0.75)
        )


def calculate_stop_loss_pct(
    entry_row: pd.Series,
    config: StopLossConfig,
    entry_price: Optional[float] = None
) -> Tuple[float, Dict]:
    """
    Calculate stop-loss percentage for a single trade entry.
    
    This is the central stop-loss calculation function. All backtesting
    code should call this function instead of hard-coding stop logic.
    
    Args:
        entry_row: Series containing features at entry time, including
                   'atr14_normalized' if using adaptive_atr or swing_atr mode,
                   and 'swing_low_10d' (or swing_low_{N}d) if using swing_atr mode.
        config: StopLossConfig object with mode and parameters.
        entry_price: Entry price (optional, will try to extract from entry_row if not provided).
    
    Returns:
        Tuple of (stop_pct, metadata_dict) where:
        - stop_pct: Stop-loss percentage as a positive fraction (e.g., 0.054 for 5.4%).
                    The caller should apply this as a negative value (e.g., -0.054).
        - metadata_dict: Dictionary with additional info about the calculation:
            - 'used_swing_low': bool (True if swing_atr mode used swing low, False if fallback)
            - 'swing_distance_pct': float (distance to swing low, if applicable)
            - 'buffer_pct': float (ATR buffer added, if applicable)
            - 'raw_stop_pct': float (raw stop before clamping, if applicable)
    
    Examples:
        >>> config = StopLossConfig(mode="constant", constant_stop_loss_pct=0.075)
        >>> entry_row = pd.Series({})
        >>> stop_pct, meta = calculate_stop_loss_pct(entry_row, config)
        >>> stop_pct
        0.075
        
        >>> config = StopLossConfig(mode="adaptive_atr", atr_stop_k=1.8, 
        ...                         atr_stop_min_pct=0.04, atr_stop_max_pct=0.10)
        >>> entry_row = pd.Series({"atr14_normalized": 0.03})
        >>> stop_pct, meta = calculate_stop_loss_pct(entry_row, config)
        >>> stop_pct  # 1.8 * 0.03 = 0.054, within [0.04, 0.10]
        0.054
    """
    metadata = {}
    
    if config.mode == "constant":
        return config.constant_stop_loss_pct, metadata
    
    elif config.mode == "adaptive_atr":
        # Get ATR normalized value from entry row
        atr_pct = entry_row.get("atr14_normalized")
        
        if pd.isna(atr_pct) or atr_pct is None:
            # Fallback to minimum if ATR is not available
            metadata['fallback_reason'] = 'atr_missing'
            return config.atr_stop_min_pct, metadata
        
        # Calculate raw stop distance: stop_k * atr_pct
        raw_stop_pct = config.atr_stop_k * atr_pct
        
        # Clamp to reasonable range
        stop_pct = np.clip(raw_stop_pct, config.atr_stop_min_pct, config.atr_stop_max_pct)
        
        metadata['raw_stop_pct'] = raw_stop_pct
        return stop_pct, metadata
    
    elif config.mode == "swing_atr":
        # Get entry price (try multiple column names)
        if entry_price is None:
            entry_price = entry_row.get('open') or entry_row.get('Open') or entry_row.get('close') or entry_row.get('Close')
        
        if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
            # Fallback: use adaptive_atr logic if entry price is unavailable
            metadata['fallback_reason'] = 'entry_price_missing'
            atr_pct = entry_row.get("atr14_normalized")
            if pd.isna(atr_pct) or atr_pct is None:
                return config.atr_stop_min_pct, metadata
            raw_stop_pct = config.atr_stop_k * atr_pct
            stop_pct = np.clip(raw_stop_pct, config.atr_stop_min_pct, config.atr_stop_max_pct)
            metadata['raw_stop_pct'] = raw_stop_pct
            return stop_pct, metadata
        
        # Get swing low feature (try swing_low_{N}d format first, then swing_low_10d)
        swing_low_feature_name = f"swing_low_{config.swing_lookback_days}d"
        swing_low = entry_row.get(swing_low_feature_name)
        
        # Fallback to swing_low_10d if the specific lookback feature doesn't exist
        if pd.isna(swing_low) or swing_low is None:
            swing_low = entry_row.get("swing_low_10d")
        
        # Get ATR normalized value
        atr_pct = entry_row.get("atr14_normalized")
        
        # Check if swing low is valid and usable
        if pd.isna(swing_low) or swing_low is None or swing_low <= 0:
            # Fallback: use adaptive_atr logic if swing low is unavailable
            metadata['used_swing_low'] = False
            metadata['fallback_reason'] = 'swing_low_missing'
            if pd.isna(atr_pct) or atr_pct is None:
                return config.atr_stop_min_pct, metadata
            raw_stop_pct = config.atr_stop_k * atr_pct
            stop_pct = np.clip(raw_stop_pct, config.atr_stop_min_pct, config.atr_stop_max_pct)
            metadata['raw_stop_pct'] = raw_stop_pct
            return stop_pct, metadata
        
        # Calculate swing distance: (entry_price - swing_low) / entry_price
        swing_distance_pct = (entry_price - swing_low) / entry_price
        
        # Check if swing distance is valid and positive (swing low should be below entry)
        if swing_distance_pct <= 0 or pd.isna(swing_distance_pct):
            # Fallback: swing low is at or above entry price (invalid)
            metadata['used_swing_low'] = False
            metadata['fallback_reason'] = 'swing_low_invalid'
            if pd.isna(atr_pct) or atr_pct is None:
                return config.atr_stop_min_pct, metadata
            raw_stop_pct = config.atr_stop_k * atr_pct
            stop_pct = np.clip(raw_stop_pct, config.atr_stop_min_pct, config.atr_stop_max_pct)
            metadata['raw_stop_pct'] = raw_stop_pct
            return stop_pct, metadata
        
        # Calculate ATR buffer
        if pd.isna(atr_pct) or atr_pct is None:
            buffer_pct = 0.0
            metadata['buffer_missing'] = True
        else:
            buffer_pct = config.swing_atr_buffer_k * atr_pct
        
        # Combine swing distance and buffer
        raw_stop_pct = swing_distance_pct + buffer_pct
        
        # Clamp to global bounds
        stop_pct = np.clip(raw_stop_pct, config.atr_stop_min_pct, config.atr_stop_max_pct)
        
        # Store metadata
        metadata['used_swing_low'] = True
        metadata['swing_distance_pct'] = swing_distance_pct
        metadata['buffer_pct'] = buffer_pct
        metadata['raw_stop_pct'] = raw_stop_pct
        
        return stop_pct, metadata
    
    else:
        raise ValueError(f"Unknown stop_loss_mode: {config.mode}")


def calculate_stop_losses_for_dataframe(
    df: pd.DataFrame,
    config: StopLossConfig,
    entry_signal_col: str = "entry_signal"
) -> pd.Series:
    """
    Calculate stop-loss percentages for all potential entry points in a DataFrame.
    
    This is useful for batch processing or when you need to pre-calculate
    all stop losses before running the backtest.
    
    Args:
        df: DataFrame with features and entry signals.
        config: StopLossConfig object with mode and parameters.
        entry_signal_col: Name of the boolean column indicating entry signals.
    
    Returns:
        Series with stop-loss percentages (as positive fractions) for each row.
        Only rows with entry signals will have calculated stops; others will be NaN.
    """
    stop_losses = pd.Series(index=df.index, dtype=float)
    
    # Only calculate stops for rows with entry signals
    entry_mask = df.get(entry_signal_col, pd.Series(False, index=df.index))
    
    for idx, row in df.iterrows():
        if entry_mask.loc[idx]:
            stop_pct, _ = calculate_stop_loss_pct(row, config)
            stop_losses.loc[idx] = stop_pct
    
    return stop_losses


def create_stop_loss_config_from_args(
    stop_loss_mode: Optional[str] = None,
    constant_stop_loss_pct: Optional[float] = None,
    atr_stop_k: Optional[float] = None,
    atr_stop_min_pct: Optional[float] = None,
    atr_stop_max_pct: Optional[float] = None,
    swing_lookback_days: Optional[int] = None,
    swing_atr_buffer_k: Optional[float] = None,
    return_threshold: Optional[float] = None,
    legacy_stop_loss: Optional[float] = None
) -> StopLossConfig:
    """
    Create StopLossConfig from command-line arguments or legacy parameters.
    
    This function handles backward compatibility with the old stop_loss parameter
    and the 2:1 risk-reward default behavior.
    
    Args:
        stop_loss_mode: "constant" or "adaptive_atr" (default: "constant")
        constant_stop_loss_pct: Fixed stop percentage (overrides legacy_stop_loss)
        atr_stop_k: Multiplier for ATR-based stops (default: 1.8)
        atr_stop_min_pct: Minimum stop distance (default: 0.04)
        atr_stop_max_pct: Maximum stop distance (default: 0.10)
        return_threshold: Return threshold for 2:1 risk-reward calculation
        legacy_stop_loss: Legacy stop_loss parameter (negative value, e.g., -0.075)
    
    Returns:
        StopLossConfig object
    """
    # Determine mode
    mode = stop_loss_mode or "constant"
    
    # Determine constant stop loss percentage
    if constant_stop_loss_pct is not None:
        const_pct = constant_stop_loss_pct
    elif legacy_stop_loss is not None:
        # Convert legacy negative value to positive
        const_pct = abs(legacy_stop_loss)
    elif return_threshold is not None:
        # Default 2:1 risk-reward ratio
        const_pct = abs(return_threshold) / 2
    else:
        const_pct = 0.075  # Default 7.5%
    
    return StopLossConfig(
        mode=mode,
        constant_stop_loss_pct=const_pct,
        atr_stop_k=atr_stop_k or 1.8,
        atr_stop_min_pct=atr_stop_min_pct or 0.04,
        atr_stop_max_pct=atr_stop_max_pct or 0.10,
        swing_lookback_days=swing_lookback_days or 10,
        swing_atr_buffer_k=swing_atr_buffer_k or 0.75
    )


def summarize_adaptive_stops(trades: pd.DataFrame, mode: str = "adaptive_atr") -> Dict:
    """
    Generate summary statistics for adaptive stop losses.
    
    This function analyzes the stop_loss_pct column in trades to provide
    insights into how adaptive stops behaved during backtesting.
    
    Args:
        trades: DataFrame of trades with 'stop_loss_pct' column (positive fractions)
        mode: Stop-loss mode ("adaptive_atr" or "swing_atr") for mode-specific stats
    
    Returns:
        Dictionary with summary statistics:
        - avg_stop_pct: Average stop distance
        - min_stop_pct: Minimum stop distance used
        - max_stop_pct: Maximum stop distance used
        - stop_distribution: Dict with bucket counts
        - swing_atr_stats: Additional stats for swing_atr mode (if applicable)
    """
    if trades.empty or "stop_loss_pct" not in trades.columns:
        return {
            "avg_stop_pct": None,
            "min_stop_pct": None,
            "max_stop_pct": None,
            "stop_distribution": {}
        }
    
    stop_pcts = trades["stop_loss_pct"]
    valid_stops = stop_pcts.dropna()
    
    if len(valid_stops) == 0:
        return {
            "avg_stop_pct": None,
            "min_stop_pct": None,
            "max_stop_pct": None,
            "stop_distribution": {}
        }
    
    # Calculate basic statistics
    avg_stop_pct = valid_stops.mean()
    min_stop_pct = valid_stops.min()
    max_stop_pct = valid_stops.max()
    
    # Create distribution buckets
    buckets = [
        (0.0, 0.04, "0-4%"),
        (0.04, 0.05, "4-5%"),
        (0.05, 0.07, "5-7%"),
        (0.07, 0.10, "7-10%"),
        (0.10, float('inf'), ">10%")
    ]
    
    stop_distribution = {}
    for low, high, label in buckets:
        count = ((valid_stops >= low) & (valid_stops < high)).sum()
        if count > 0:
            pct = (count / len(valid_stops)) * 100
            stop_distribution[label] = {"count": count, "pct": pct}
    
    result = {
        "avg_stop_pct": avg_stop_pct,
        "min_stop_pct": min_stop_pct,
        "max_stop_pct": max_stop_pct,
        "stop_distribution": stop_distribution
    }
    
    # Add swing_atr-specific statistics if mode is swing_atr
    if mode == "swing_atr" and "used_swing_low" in trades.columns:
        swing_stats = {}
        used_swing = trades["used_swing_low"].fillna(False)
        swing_stats['trades_using_swing_low'] = used_swing.sum()
        swing_stats['trades_using_fallback'] = (~used_swing).sum()
        swing_stats['swing_low_usage_pct'] = (used_swing.sum() / len(used_swing) * 100) if len(used_swing) > 0 else 0
        
        if 'swing_distance_pct' in trades.columns:
            swing_distances = trades[used_swing]['swing_distance_pct'].dropna()
            if len(swing_distances) > 0:
                swing_stats['avg_swing_distance_pct'] = swing_distances.mean()
                swing_stats['min_swing_distance_pct'] = swing_distances.min()
                swing_stats['max_swing_distance_pct'] = swing_distances.max()
        
        if 'buffer_pct' in trades.columns:
            buffers = trades[used_swing]['buffer_pct'].dropna()
            if len(buffers) > 0:
                swing_stats['avg_buffer_pct'] = buffers.mean()
        
        result['swing_atr_stats'] = swing_stats
    
    return result

