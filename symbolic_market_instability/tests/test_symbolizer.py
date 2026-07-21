"""Tests for symbolization module."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path so we can import the src package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.symbolization.symbolizer import FeatureSymbolizer


@pytest.fixture
def sample_thresholds():
    """Sample thresholds for testing."""
    return {
        'rolling_return': {'p10': -0.08, 'p25': -0.03, 'p40': -0.01, 
                          'p70': 0.02, 'p75': 0.03, 'p80': 0.04, 'p90': 0.12},
        'rolling_volatility': {'p10': 0.008, 'p25': 0.012, 'p40': 0.016,
                             'p70': 0.022, 'p75': 0.025, 'p80': 0.030, 'p90': 0.045},
        'price_slope': {'p10': -0.002, 'p25': -0.0005, 'p40': 0.0005,
                       'p70': 0.0015, 'p75': 0.002, 'p80': 0.0025, 'p90': 0.004},
        'volume_trend': {'p10': -0.25, 'p25': -0.10, 'p30': -0.08, 'p40': -0.05,
                        'p70': 0.15, 'p75': 0.20, 'p80': 0.25, 'p90': 0.40},
        'vix': {'p10': 12.0, 'p25': 14.0, 'p40': 17.0,
               'p70': 22.0, 'p75': 25.0, 'p80': 28.0, 'p90': 40.0}
    }


def test_symbolize_rising_prices(sample_thresholds):
    """Test symbolization of rising prices."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    
    # Price slope above p75 should yield RisingPrices
    row = pd.Series({
        'price_slope': 0.003,  # Above p75 (0.002)
        'rolling_volatility': 0.020,
        'volume_trend': 0.10,
        'vix': 20.0,
        'rolling_return': 0.01
    })
    
    symbols = symbolizer.symbolize_row(row)
    assert 'RisingPrices' in symbols


def test_symbolize_falling_prices(sample_thresholds):
    """Test symbolization of falling prices."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    
    # Price slope below p25 should yield FallingPrices
    row = pd.Series({
        'price_slope': -0.001,  # Below p25 (-0.0005)
        'rolling_volatility': 0.020,
        'volume_trend': 0.10,
        'vix': 20.0,
        'rolling_return': 0.01
    })
    
    symbols = symbolizer.symbolize_row(row)
    assert 'FallingPrices' in symbols


def test_symbolize_high_volatility(sample_thresholds):
    """Test symbolization of high volatility."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    
    # Volatility above p80 should yield HighVolatility
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.035,  # Above p80 (0.030)
        'volume_trend': 0.10,
        'vix': 20.0,
        'rolling_return': 0.01
    })
    
    symbols = symbolizer.symbolize_row(row)
    assert 'HighVolatility' in symbols


def test_temporal_persistence(sample_thresholds):
    """Test temporal persistence detection."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    
    # Create a series with symbols
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    symbols_series = pd.Series([
        {'HighVolatility'},
        {'HighVolatility'},
        {'HighVolatility'},
        {'HighVolatility'},
        {'LowVolatility'},
        {'HighVolatility'},
        {'HighVolatility'},
        set(),
        {'HighVolatility'},
        {'HighVolatility'}
    ], index=dates)
    
    # Check persistence of HighVolatility (min 3 days)
    persistence = symbolizer.check_temporal_persistence(
        symbols_series, 'HighVolatility', min_days=3
    )
    
    # First 4 days should be persistent
    assert persistence.iloc[0:4].all()
    # Days 5-7 should not be (only 2 consecutive)
    assert not persistence.iloc[5:7].all()


if __name__ == "__main__":
    pytest.main([__file__])
