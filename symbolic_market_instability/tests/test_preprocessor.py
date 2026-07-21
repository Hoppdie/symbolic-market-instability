"""Tests for DataPreprocessor."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_acquisition.preprocessor import DataPreprocessor


@pytest.fixture
def clean_market_data():
    """A valid, clean market DataFrame with no NaNs or outliers."""
    dates = pd.date_range('2020-01-01', periods=50, freq='D')
    np.random.seed(1)
    prices = 3000 + np.random.normal(0, 20, 50)
    return pd.DataFrame({
        'Open':   prices * 0.99,
        'High':   prices * 1.01,
        'Low':    prices * 0.98,
        'Close':  prices,
        'Volume': np.random.randint(1_000_000, 5_000_000, 50),
        'VIX':    np.random.uniform(13, 25, 50),
    }, index=dates)


# ---------------------------------------------------------------------------
# clean() – validation
# ---------------------------------------------------------------------------

def test_clean_negative_price_raises(clean_market_data):
    """Negative Close price must raise ValueError."""
    preprocessor = DataPreprocessor()
    clean_market_data.iloc[5, clean_market_data.columns.get_loc('Close')] = -10.0
    with pytest.raises(ValueError, match="Negative prices"):
        preprocessor.clean(clean_market_data)


def test_clean_negative_volume_raises(clean_market_data):
    """Negative Volume must raise ValueError."""
    preprocessor = DataPreprocessor()
    clean_market_data.iloc[3, clean_market_data.columns.get_loc('Volume')] = -1
    with pytest.raises(ValueError, match="Negative volumes"):
        preprocessor.clean(clean_market_data)


# ---------------------------------------------------------------------------
# clean() – forward-fill
# ---------------------------------------------------------------------------

def test_clean_forward_fills_small_gaps(clean_market_data):
    """1-3 consecutive NaNs should be forward-filled, not dropped."""
    preprocessor = DataPreprocessor()
    # Introduce 2 consecutive NaNs in Close
    clean_market_data.iloc[10, clean_market_data.columns.get_loc('Close')] = np.nan
    clean_market_data.iloc[11, clean_market_data.columns.get_loc('Close')] = np.nan

    result = preprocessor.clean(clean_market_data)

    assert len(result) == len(clean_market_data), "Rows should not be dropped for small gaps"
    assert not result['Close'].isna().any(), "Forward-fill should remove NaNs"


def test_clean_drops_large_gap_rows(clean_market_data):
    """4+ consecutive NaNs must cause those rows to be dropped."""
    preprocessor = DataPreprocessor()
    for i in range(10, 15):  # 5 consecutive NaNs
        clean_market_data.iloc[i, clean_market_data.columns.get_loc('Close')] = np.nan

    result = preprocessor.clean(clean_market_data)

    assert len(result) < len(clean_market_data), "Rows with large NaN gaps should be dropped"
    assert not result['Close'].isna().any(), "No NaNs should remain after cleaning"


# ---------------------------------------------------------------------------
# clean() – outlier replacement
# ---------------------------------------------------------------------------

def test_clean_replaces_outlier_with_median(clean_market_data):
    """A value 6 std-devs from the mean should be replaced with the median."""
    preprocessor = DataPreprocessor()
    mean = clean_market_data['Close'].mean()
    std = clean_market_data['Close'].std()
    original_median = clean_market_data['Close'].median()

    # Inject extreme outlier
    clean_market_data.iloc[20, clean_market_data.columns.get_loc('Close')] = mean + 20 * std

    result = preprocessor.clean(clean_market_data)

    # The outlier index should now be close to the pre-injection median
    assert abs(result['Close'].iloc[20] - original_median) < 1.0


def test_clean_preserves_valid_data(clean_market_data):
    """A DataFrame with no issues should be returned intact (same row count)."""
    preprocessor = DataPreprocessor()
    result = preprocessor.clean(clean_market_data)
    assert len(result) == len(clean_market_data)
    assert not result.isna().any().any()


# ---------------------------------------------------------------------------
# split_temporal()
# ---------------------------------------------------------------------------

def test_split_temporal_correct_sizes(clean_market_data):
    """Default 60/20/20 split should produce correct proportions."""
    preprocessor = DataPreprocessor()
    n = len(clean_market_data)
    calib, valid, test = preprocessor.split_temporal(clean_market_data)

    assert len(calib) == int(n * 0.6)
    assert len(valid) == int(n * 0.2)
    # Test set gets the remainder
    assert len(test) == n - len(calib) - len(valid)


def test_split_temporal_no_overlap(clean_market_data):
    """The three splits must not share any index values."""
    preprocessor = DataPreprocessor()
    calib, valid, test = preprocessor.split_temporal(clean_market_data)

    assert set(calib.index).isdisjoint(set(valid.index))
    assert set(calib.index).isdisjoint(set(test.index))
    assert set(valid.index).isdisjoint(set(test.index))


def test_split_temporal_chronological_order(clean_market_data):
    """calib dates < valid dates < test dates."""
    preprocessor = DataPreprocessor()
    calib, valid, test = preprocessor.split_temporal(clean_market_data)

    assert calib.index.max() < valid.index.min()
    assert valid.index.max() < test.index.min()


def test_split_temporal_invalid_ratios_raises(clean_market_data):
    """Ratios that exceed 1.0 should raise ValueError."""
    preprocessor = DataPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.split_temporal(clean_market_data, calib_ratio=0.7, valid_ratio=0.5)


def test_split_temporal_empty_data_raises():
    """Empty DataFrame should raise ValueError."""
    preprocessor = DataPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.split_temporal(pd.DataFrame())
