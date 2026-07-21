"""Tests for feature computation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path so we can import the src package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_engineering.technical_features import TechnicalFeatureComputer


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    
    # Create synthetic price data (trending upward with some noise)
    np.random.seed(42)
    base_price = 3000
    trend = np.linspace(0, 200, 100)
    noise = np.random.normal(0, 20, 100)
    prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100),
        'VIX': np.random.uniform(12, 30, 100)
    }, index=dates)
    
    return data


def test_compute_rolling_return(sample_data):
    """Test rolling return computation."""
    computer = TechnicalFeatureComputer()
    features = computer.compute_all_features(sample_data)
    
    assert 'rolling_return' in features.columns
    # All rows kept after feature computation should have valid (non-NaN) values
    assert not features['rolling_return'].isna().any()


def test_compute_rolling_volatility(sample_data):
    """Test rolling volatility computation."""
    computer = TechnicalFeatureComputer()
    features = computer.compute_all_features(sample_data)
    
    assert 'rolling_volatility' in features.columns
    # All rows kept after feature computation should have valid (non-NaN) values
    assert not features['rolling_volatility'].isna().any()
    # Volatility should be non‑negative
    assert (features['rolling_volatility'] >= 0).all()


def test_compute_price_slope(sample_data):
    """Test price slope computation."""
    computer = TechnicalFeatureComputer()
    features = computer.compute_all_features(sample_data)
    
    assert 'price_slope' in features.columns
    # All rows kept after feature computation should have valid (non-NaN) values
    assert not features['price_slope'].isna().any()
    # For upward trending data, at least one slope value should be positive
    assert (features['price_slope'] > 0).any()


def test_compute_volume_trend(sample_data):
    """Test volume trend computation."""
    computer = TechnicalFeatureComputer()
    features = computer.compute_all_features(sample_data)
    
    assert 'volume_trend' in features.columns
    # All rows kept after feature computation should have valid (non-NaN) values
    assert not features['volume_trend'].isna().any()


def test_compute_vix(sample_data):
    """Test VIX feature."""
    computer = TechnicalFeatureComputer()
    features = computer.compute_all_features(sample_data)
    
    assert 'vix' in features.columns
    # VIX should match original VIX values on aligned index (after dropping NaN rows)
    original_vix_aligned = sample_data['VIX'].loc[features.index]
    # Ignore the name difference between 'vix' and 'VIX'
    pd.testing.assert_series_equal(features['vix'], original_vix_aligned, check_names=False)


def test_drop_na_rows(sample_data):
    """Test that NaN rows are dropped."""
    computer = TechnicalFeatureComputer()
    features = computer.compute_all_features(sample_data)
    
    # Should have fewer rows than original (first 30 dropped)
    assert len(features) < len(sample_data)
    # Should have no NaN values in feature columns
    feature_cols = ['rolling_return', 'rolling_volatility', 'price_slope', 
                    'volume_trend', 'vix']
    for col in feature_cols:
        assert not features[col].isna().any()


if __name__ == "__main__":
    pytest.main([__file__])
