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


def test_symbolize_high_volume(sample_thresholds):
    """volume_trend above p70 should yield HighVolume."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.020,
        'volume_trend': 0.20,   # Above p70 (0.15)
        'vix': 20.0,
        'rolling_return': 0.01,
    })
    assert 'HighVolume' in symbolizer.symbolize_row(row)


def test_symbolize_low_volume(sample_thresholds):
    """volume_trend below p30 should yield LowVolume."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.020,
        'volume_trend': -0.15,  # Below p30 (-0.08)
        'vix': 20.0,
        'rolling_return': 0.01,
    })
    assert 'LowVolume' in symbolizer.symbolize_row(row)


def test_symbolize_normal_volume(sample_thresholds):
    """volume_trend in middle range should yield NormalVolume."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.020,
        'volume_trend': 0.05,   # Between p30 and p70
        'vix': 20.0,
        'rolling_return': 0.01,
    })
    assert 'NormalVolume' in symbolizer.symbolize_row(row)


def test_symbolize_tight_liquidity(sample_thresholds):
    """VIX above p75 should yield TightLiquidity."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.020,
        'volume_trend': 0.05,
        'vix': 30.0,    # Above p75 (25.0)
        'rolling_return': 0.01,
    })
    assert 'TightLiquidity' in symbolizer.symbolize_row(row)


def test_symbolize_loose_liquidity(sample_thresholds):
    """VIX below p25 should yield LooseLiquidity."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.020,
        'volume_trend': 0.05,
        'vix': 13.0,    # Below p25 (14.0)
        'rolling_return': 0.01,
    })
    assert 'LooseLiquidity' in symbolizer.symbolize_row(row)


def test_symbolize_excessive_gains(sample_thresholds):
    """rolling_return above p90 should yield ExcessiveGains."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.020,
        'volume_trend': 0.05,
        'vix': 20.0,
        'rolling_return': 0.15,  # Above p90 (0.12)
    })
    assert 'ExcessiveGains' in symbolizer.symbolize_row(row)


def test_symbolize_excessive_losses(sample_thresholds):
    """rolling_return below p10 should yield ExcessiveLosses."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,
        'rolling_volatility': 0.020,
        'volume_trend': 0.05,
        'vix': 20.0,
        'rolling_return': -0.10,  # Below p10 (-0.08)
    })
    assert 'ExcessiveLosses' in symbolizer.symbolize_row(row)


def test_symbolize_stable_prices(sample_thresholds):
    """price_slope between p25 and p75 should yield StablePrices."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': 0.001,   # Between p25 (-0.0005) and p75 (0.002)
        'rolling_volatility': 0.020,
        'volume_trend': 0.05,
        'vix': 20.0,
        'rolling_return': 0.01,
    })
    assert 'StablePrices' in symbolizer.symbolize_row(row)


def test_symbolize_nan_returns_empty_set(sample_thresholds):
    """Row with all-NaN features should return an empty set (no crash)."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    row = pd.Series({
        'price_slope': float('nan'),
        'rolling_volatility': float('nan'),
        'volume_trend': float('nan'),
        'vix': float('nan'),
        'rolling_return': float('nan'),
    })
    assert symbolizer.symbolize_row(row) == set()


def test_symbolize_dataframe_output_shape(sample_thresholds):
    """symbolize_dataframe() must preserve the index and add a 'symbols' column."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'price_slope':        [0.003, -0.001, 0.001, 0.001, 0.001],
        'rolling_volatility': [0.035, 0.035, 0.020, 0.020, 0.020],
        'volume_trend':       [0.20,  0.20,  0.05,  0.05,  0.05],
        'vix':                [30.0,  30.0,  20.0,  20.0,  20.0],
        'rolling_return':     [0.15,  0.15,  0.01,  0.01,  0.01],
    }, index=dates)
    result = symbolizer.symbolize_dataframe(df)
    assert list(result.index) == list(df.index)
    assert 'symbols' in result.columns


def test_symbolize_dataframe_each_row_is_set(sample_thresholds):
    """Every value in the 'symbols' column must be a set instance."""
    symbolizer = FeatureSymbolizer(sample_thresholds)
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'price_slope':        [0.003] * 5,
        'rolling_volatility': [0.035] * 5,
        'volume_trend':       [0.20]  * 5,
        'vix':                [30.0]  * 5,
        'rolling_return':     [0.15]  * 5,
    }, index=dates)
    result = symbolizer.symbolize_dataframe(df)
    for item in result['symbols']:
        assert isinstance(item, set)


if __name__ == "__main__":
    pytest.main([__file__])
