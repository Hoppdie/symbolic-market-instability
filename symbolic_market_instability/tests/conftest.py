"""Shared pytest fixtures for all test modules."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_base.rules import Rule


@pytest.fixture
def sample_thresholds():
    """Standard percentile thresholds used across symbolization tests."""
    return {
        'rolling_return': {
            'p10': -0.08, 'p25': -0.03, 'p40': -0.01,
            'p70': 0.02, 'p75': 0.03, 'p80': 0.04, 'p90': 0.12,
        },
        'rolling_volatility': {
            'p10': 0.008, 'p25': 0.012, 'p40': 0.016,
            'p70': 0.022, 'p75': 0.025, 'p80': 0.030, 'p90': 0.045,
        },
        'price_slope': {
            'p10': -0.002, 'p25': -0.0005, 'p40': 0.0005,
            'p70': 0.0015, 'p75': 0.002, 'p80': 0.0025, 'p90': 0.004,
        },
        'volume_trend': {
            'p10': -0.25, 'p25': -0.10, 'p30': -0.08, 'p40': -0.05,
            'p70': 0.15, 'p75': 0.20, 'p80': 0.25, 'p90': 0.40,
        },
        'vix': {
            'p10': 12.0, 'p25': 14.0, 'p40': 17.0,
            'p70': 22.0, 'p75': 25.0, 'p80': 28.0, 'p90': 40.0,
        },
    }


@pytest.fixture
def sample_features_df():
    """100-row DataFrame with all 5 feature columns and no NaNs."""
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'rolling_return': np.random.normal(0.01, 0.04, 100),
        'rolling_volatility': np.abs(np.random.normal(0.02, 0.01, 100)),
        'price_slope': np.random.normal(0.001, 0.002, 100),
        'volume_trend': np.random.normal(0.0, 0.15, 100),
        'vix': np.random.uniform(13, 30, 100),
    }, index=dates)


@pytest.fixture
def minimal_rules():
    """Two simple in-memory rules for lightweight inference tests."""
    return [
        Rule(
            id='R1',
            antecedents=[{'RisingPrices', 'HighVolume'}],
            consequent='SpeculativeActivity',
            confidence=0.8,
            description='Speculative activity rule',
        ),
        Rule(
            id='R2',
            antecedents=[{'SpeculativeActivity', 'HighVolatility'}],
            consequent='BubbleState',
            confidence=0.9,
            description='Bubble state rule',
        ),
    ]
