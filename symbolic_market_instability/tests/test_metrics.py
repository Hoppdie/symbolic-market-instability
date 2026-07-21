"""Tests for evaluation metric functions."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.metrics import (
    compute_lead_time,
    compute_precision,
    compute_recall,
    stability_score,
)


def _bool_series(values, start='2020-01-01'):
    """Helper: create a boolean Series with a daily DatetimeIndex."""
    dates = pd.date_range(start, periods=len(values), freq='D')
    return pd.Series(values, index=dates, dtype=bool)


# ---------------------------------------------------------------------------
# compute_lead_time
# ---------------------------------------------------------------------------

def test_lead_time_warns_before_crash():
    """Last warning 8 days before crash should return 8."""
    warnings = _bool_series([False] * 5 + [True] + [False] * 10)
    crash_date = str(warnings.index[5 + 8].date())  # 8 days after the warning
    assert compute_lead_time(warnings, crash_date) == 8


def test_lead_time_no_warnings():
    """No warnings at all should return 0."""
    warnings = _bool_series([False] * 20)
    crash_date = str(warnings.index[10].date())
    assert compute_lead_time(warnings, crash_date) == 0


def test_lead_time_warnings_only_after_crash():
    """Warnings that all occur after the crash should return 0."""
    warnings = _bool_series([False] * 10 + [True] * 5)
    crash_date = str(warnings.index[5].date())  # crash before any warning
    assert compute_lead_time(warnings, crash_date) == 0


def test_lead_time_warning_on_crash_day():
    """A warning on the crash day itself counts as 0 days advance notice."""
    warnings = _bool_series([False] * 9 + [True])
    crash_date = str(warnings.index[9].date())
    assert compute_lead_time(warnings, crash_date) == 0


# ---------------------------------------------------------------------------
# compute_precision
# ---------------------------------------------------------------------------

def test_precision_all_warnings_in_window():
    """All warnings inside the crash window → precision = 1.0."""
    warnings = _bool_series([False] * 5 + [True] * 5 + [False] * 5)
    window = (str(warnings.index[5].date()), str(warnings.index[9].date()))
    assert compute_precision(warnings, window) == 1.0


def test_precision_no_warnings_in_window():
    """All warnings outside the crash window → precision = 0.0."""
    warnings = _bool_series([True] * 5 + [False] * 10)
    # Window is after all warnings
    window = (str(warnings.index[6].date()), str(warnings.index[10].date()))
    assert compute_precision(warnings, window) == 0.0


def test_precision_no_warnings_at_all():
    """Zero total warnings → precision = 0.0 (not a division-by-zero error)."""
    warnings = _bool_series([False] * 10)
    window = (str(warnings.index[3].date()), str(warnings.index[7].date()))
    assert compute_precision(warnings, window) == 0.0


def test_precision_partial():
    """Half the warnings inside the window → precision ≈ 0.5."""
    warnings = _bool_series([True] * 5 + [False] * 5 + [True] * 5)
    # Window covers only the last 5 True values
    window = (str(warnings.index[10].date()), str(warnings.index[14].date()))
    precision = compute_precision(warnings, window)
    assert abs(precision - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# compute_recall
# ---------------------------------------------------------------------------

def test_recall_full():
    """Warning on every crash window day → recall = 1.0."""
    warnings = _bool_series([False] * 5 + [True] * 10 + [False] * 5)
    window = (str(warnings.index[5].date()), str(warnings.index[14].date()))
    assert compute_recall(warnings, window) == 1.0


def test_recall_partial():
    """Warnings on half the crash window days → recall = 0.5."""
    values = [False] * 5 + [True, False] * 5 + [False] * 5  # alternating in window
    warnings = _bool_series(values)
    window = (str(warnings.index[5].date()), str(warnings.index[14].date()))
    recall = compute_recall(warnings, window)
    assert abs(recall - 0.5) < 1e-9


def test_recall_no_window_days():
    """Window dates that don't match any index entries → recall = 0.0."""
    warnings = _bool_series([True] * 10)
    # Window is entirely outside the series range
    assert compute_recall(warnings, ('2025-01-01', '2025-01-10')) == 0.0


def test_recall_no_warnings_in_window():
    """No warnings at all in window → recall = 0.0."""
    warnings = _bool_series([False] * 20)
    window = (str(warnings.index[5].date()), str(warnings.index[14].date()))
    assert compute_recall(warnings, window) == 0.0


# ---------------------------------------------------------------------------
# stability_score
# ---------------------------------------------------------------------------

def test_stability_all_true():
    """All-True series has no transitions → maximum stability score (1.0)."""
    warnings = _bool_series([True] * 10)
    assert stability_score(warnings) == 1.0


def test_stability_all_false():
    """All-False series has no transitions → maximum stability score (1.0)."""
    warnings = _bool_series([False] * 10)
    assert stability_score(warnings) == 1.0


def test_stability_alternating():
    """Perfectly alternating series has many transitions → low stability score."""
    warnings = _bool_series([True, False] * 10)
    score = stability_score(warnings)
    assert score < 0.5


def test_stability_single_element():
    """Single-element series returns 0.0 (edge case defined in docstring)."""
    warnings = _bool_series([True])
    assert stability_score(warnings) == 0.0


def test_stability_score_in_range():
    """stability_score must always return a value in [0.0, 1.0]."""
    np.random.seed(42)
    for _ in range(20):
        values = np.random.choice([True, False], size=np.random.randint(2, 30))
        score = stability_score(_bool_series(values.tolist()))
        assert 0.0 <= score <= 1.0
