"""Evaluation metrics for instability detection."""

from typing import Tuple
import pandas as pd
import numpy as np


def compute_lead_time(warnings: pd.Series, crash_date: str) -> int:
    """
    Compute lead time (days of advance warning) before crash.
    
    Args:
        warnings: Boolean Series indicating warning days (True = warning)
        crash_date: Crash date in 'YYYY-MM-DD' format
    
    Returns:
        Number of days of advance warning (0 if no warning before crash)
    """
    crash_idx = pd.to_datetime(crash_date)
    
    # Find last warning before crash
    warnings_before = warnings[warnings.index < crash_idx]
    
    if warnings_before.empty or not warnings_before.any():
        return 0
    
    # Get last warning date
    last_warning_date = warnings_before[warnings_before].index.max()
    
    # Compute days between last warning and crash
    lead_time = (crash_idx - last_warning_date).days
    
    return max(0, lead_time)


def compute_precision(
    warnings: pd.Series, 
    crash_window: Tuple[str, str]
) -> float:
    """
    Compute precision: proportion of warnings that occur within crash window.
    
    Args:
        warnings: Boolean Series indicating warning days
        crash_window: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    
    Returns:
        Precision score (0.0 to 1.0)
    """
    start_date = pd.to_datetime(crash_window[0])
    end_date = pd.to_datetime(crash_window[1])
    
    # Total warnings
    total_warnings = warnings.sum()
    
    if total_warnings == 0:
        return 0.0
    
    # Warnings within crash window
    window_mask = (warnings.index >= start_date) & (warnings.index <= end_date)
    warnings_in_window = warnings[window_mask].sum()
    
    precision = warnings_in_window / total_warnings
    
    return precision


def stability_score(warnings: pd.Series) -> float:
    """
    Compute stability score: measures consistency of warnings.
    
    Lower score = more stable (fewer fluctuations).
    Higher score = less stable (more fluctuations).
    
    Args:
        warnings: Boolean Series indicating warning days
    
    Returns:
        Stability score (0.0 to 1.0, lower is better)
    """
    if len(warnings) < 2:
        return 0.0
    
    # Count transitions (True->False or False->True)
    transitions = (warnings != warnings.shift(1)).sum() - 1  # Subtract 1 for first row
    
    # Normalize by length
    max_possible_transitions = len(warnings) - 1
    stability = 1.0 - (transitions / max_possible_transitions) if max_possible_transitions > 0 else 0.0
    
    return stability


def compute_recall(
    warnings: pd.Series,
    crash_window: Tuple[str, str]
) -> float:
    """
    Compute recall: proportion of crash window days with warnings.
    
    Args:
        warnings: Boolean Series indicating warning days
        crash_window: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    
    Returns:
        Recall score (0.0 to 1.0)
    """
    start_date = pd.to_datetime(crash_window[0])
    end_date = pd.to_datetime(crash_window[1])
    
    # Days in crash window
    window_mask = (warnings.index >= start_date) & (warnings.index <= end_date)
    window_days = window_mask.sum()
    
    if window_days == 0:
        return 0.0
    
    # Warnings within crash window
    warnings_in_window = warnings[window_mask].sum()
    
    recall = warnings_in_window / window_days
    
    return recall
