"""Tests for FuzzySymbolizer (triangular membership functions)."""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.symbolization.fuzzy_logic import FuzzySymbolizer


@pytest.fixture
def fuzzy():
    return FuzzySymbolizer()


# ---------------------------------------------------------------------------
# compute_membership() – boundary conditions
# ---------------------------------------------------------------------------

def test_membership_below_low_is_zero(fuzzy):
    assert fuzzy.compute_membership(0.0, low=1.0, mid=2.0, high=3.0) == 0.0


def test_membership_at_low_is_zero(fuzzy):
    assert fuzzy.compute_membership(1.0, low=1.0, mid=2.0, high=3.0) == 0.0


def test_membership_at_mid_is_one(fuzzy):
    assert fuzzy.compute_membership(2.0, low=1.0, mid=2.0, high=3.0) == 1.0


def test_membership_at_high_is_zero(fuzzy):
    assert fuzzy.compute_membership(3.0, low=1.0, mid=2.0, high=3.0) == 0.0


def test_membership_above_high_is_zero(fuzzy):
    assert fuzzy.compute_membership(4.0, low=1.0, mid=2.0, high=3.0) == 0.0


def test_membership_left_interpolation(fuzzy):
    """Halfway between low and mid → 0.5."""
    assert fuzzy.compute_membership(1.5, low=1.0, mid=2.0, high=3.0) == pytest.approx(0.5)


def test_membership_right_interpolation(fuzzy):
    """Halfway between mid and high → 0.5."""
    assert fuzzy.compute_membership(2.5, low=1.0, mid=2.0, high=3.0) == pytest.approx(0.5)


def test_membership_output_always_in_range(fuzzy):
    """Membership must be in [0, 1] for any input across the full range."""
    for v in np.linspace(-5, 10, 100):
        m = fuzzy.compute_membership(v, low=1.0, mid=3.0, high=5.0)
        assert 0.0 <= m <= 1.0, f"Membership {m} out of range for value {v}"


# ---------------------------------------------------------------------------
# compute_fuzzy_symbols()
# ---------------------------------------------------------------------------

@pytest.fixture
def vol_thresholds():
    return {
        'p10': 0.008, 'p25': 0.012, 'p40': 0.016,
        'p70': 0.022, 'p75': 0.025, 'p80': 0.030, 'p90': 0.045,
    }


def test_fuzzy_symbols_returns_dict(fuzzy, vol_thresholds):
    result = fuzzy.compute_fuzzy_symbols(0.02, vol_thresholds)
    assert isinstance(result, dict)


def test_fuzzy_symbols_high_key_present(fuzzy, vol_thresholds):
    """With p70/p80/p90 in thresholds, 'High' key must appear."""
    result = fuzzy.compute_fuzzy_symbols(0.035, vol_thresholds)
    assert 'High' in result


def test_fuzzy_symbols_low_key_present(fuzzy, vol_thresholds):
    """With p10/p25/p40 in thresholds, 'Low' key must appear."""
    result = fuzzy.compute_fuzzy_symbols(0.012, vol_thresholds)
    assert 'Low' in result


def test_fuzzy_symbols_values_in_range(fuzzy, vol_thresholds):
    """All membership values must be in [0.0, 1.0]."""
    result = fuzzy.compute_fuzzy_symbols(0.02, vol_thresholds)
    for symbol, val in result.items():
        assert 0.0 <= val <= 1.0, f"Symbol '{symbol}' membership {val} out of range"


def test_fuzzy_symbols_peak_value_at_threshold(fuzzy, vol_thresholds):
    """Value at p80 (triangle peak for High) should have High membership = 1.0."""
    result = fuzzy.compute_fuzzy_symbols(vol_thresholds['p80'], vol_thresholds)
    assert result.get('High', 0.0) == pytest.approx(1.0)
