"""Tests for ThresholdCalibrator."""

import json
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.calibrator import ThresholdCalibrator


@pytest.fixture
def feature_df():
    """100-row DataFrame with all 5 required feature columns."""
    np.random.seed(7)
    dates = pd.date_range('2015-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'rolling_return':     np.random.normal(0.01, 0.04, 100),
        'rolling_volatility': np.abs(np.random.normal(0.02, 0.008, 100)),
        'price_slope':        np.random.normal(0.001, 0.002, 100),
        'volume_trend':       np.random.normal(0.0, 0.15, 100),
        'vix':                np.random.uniform(13, 30, 100),
    }, index=dates)


@pytest.fixture
def calibrator(tmp_path):
    """Calibrator that writes to a temp directory."""
    return ThresholdCalibrator(config_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------

def test_calibrate_returns_all_features(calibrator, feature_df):
    """Returned dict must contain all 5 feature keys."""
    thresholds = calibrator.calibrate(feature_df)
    for key in ['rolling_return', 'rolling_volatility', 'price_slope', 'volume_trend', 'vix']:
        assert key in thresholds


def test_calibrate_correct_percentiles(calibrator, feature_df):
    """p10/p90 values must match numpy's percentile computation."""
    thresholds = calibrator.calibrate(feature_df)
    for feature in ['rolling_return', 'rolling_volatility', 'vix']:
        expected_p10 = float(np.percentile(feature_df[feature].dropna(), 10))
        expected_p90 = float(np.percentile(feature_df[feature].dropna(), 90))
        assert abs(thresholds[feature]['p10'] - expected_p10) < 1e-9
        assert abs(thresholds[feature]['p90'] - expected_p90) < 1e-9


def test_calibrate_all_percentile_keys_present(calibrator, feature_df):
    """Each feature must have p10, p25, p40, p70, p75, p80, p90."""
    thresholds = calibrator.calibrate(feature_df)
    expected_keys = {'p10', 'p25', 'p40', 'p70', 'p75', 'p80', 'p90'}
    for feature, values in thresholds.items():
        assert expected_keys.issubset(set(values.keys())), \
            f"Feature '{feature}' missing percentile keys"


def test_calibrate_missing_feature_raises(calibrator, feature_df):
    """DataFrame missing a required feature column must raise ValueError."""
    incomplete = feature_df.drop(columns=['vix'])
    with pytest.raises(ValueError, match="vix"):
        calibrator.calibrate(incomplete)


def test_calibrate_all_nan_feature_raises(calibrator, feature_df):
    """Feature column with all NaN values must raise ValueError."""
    feature_df['vix'] = np.nan
    with pytest.raises(ValueError):
        calibrator.calibrate(feature_df)


def test_calibrate_saves_file(calibrator, feature_df, tmp_path):
    """calibrate() must write thresholds.json to the config directory."""
    calibrator.calibrate(feature_df)
    assert (tmp_path / 'thresholds.json').exists()


# ---------------------------------------------------------------------------
# load_thresholds()
# ---------------------------------------------------------------------------

def test_load_thresholds_round_trip(calibrator, feature_df):
    """Values loaded from disk must match those returned by calibrate()."""
    saved = calibrator.calibrate(feature_df)
    loaded = calibrator.load_thresholds()
    assert saved == loaded


def test_load_thresholds_missing_file_raises(tmp_path):
    """load_thresholds() must raise FileNotFoundError when no file exists."""
    calibrator = ThresholdCalibrator(config_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        calibrator.load_thresholds()


# ---------------------------------------------------------------------------
# validate_economic_plausibility()
# ---------------------------------------------------------------------------

def test_validate_vix_p75_too_high_raises(calibrator):
    """VIX p75 >= 35 must raise ValueError."""
    bad_thresholds = {
        'vix': {'p10': 12.0, 'p25': 15.0, 'p75': 40.0}
    }
    with pytest.raises(ValueError, match="p75"):
        calibrator.validate_economic_plausibility(bad_thresholds)


def test_validate_vix_p25_too_low_raises(calibrator):
    """VIX p25 <= 10 must raise ValueError."""
    bad_thresholds = {
        'vix': {'p10': 8.0, 'p25': 9.0, 'p75': 25.0}
    }
    with pytest.raises(ValueError, match="p25"):
        calibrator.validate_economic_plausibility(bad_thresholds)


def test_validate_plausible_thresholds_pass(calibrator):
    """Valid VIX thresholds must not raise any exception."""
    good_thresholds = {
        'vix': {'p10': 12.0, 'p25': 14.5, 'p75': 26.0}
    }
    calibrator.validate_economic_plausibility(good_thresholds)  # should not raise


def test_validate_missing_vix_raises(calibrator):
    """Thresholds dict without 'vix' key must raise ValueError."""
    with pytest.raises(ValueError, match="VIX"):
        calibrator.validate_economic_plausibility({})
