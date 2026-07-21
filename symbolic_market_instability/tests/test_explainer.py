"""Tests for ExplanationGenerator."""

import pytest
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.reasoning_engine.explainer import ExplanationGenerator


@pytest.fixture
def thresholds():
    return {
        'rolling_return':     {'p10': -0.08, 'p25': -0.03, 'p90': 0.12},
        'rolling_volatility': {'p10': 0.008, 'p40': 0.016, 'p80': 0.030},
        'price_slope':        {'p25': -0.0005, 'p75': 0.002},
        'volume_trend':       {'p25': -0.10, 'p30': -0.08, 'p70': 0.15},
        'vix':                {'p25': 14.0, 'p75': 25.0},
    }


@pytest.fixture
def features():
    return pd.Series({
        'rolling_return':     0.15,
        'rolling_volatility': 0.035,
        'price_slope':        0.003,
        'volume_trend':       0.20,
        'vix':                30.0,
    })


def _make_result(level, all_facts, fired_rules=None, chain=None):
    return {
        'instability_level': level,
        'all_facts': all_facts,
        'fired_rules': fired_rules or set(),
        'inference_chain': chain or [],
    }


# ---------------------------------------------------------------------------
# generate_explanation() – output structure
# ---------------------------------------------------------------------------

def test_explanation_contains_instability_level_header(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result('Low', set())
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'INSTABILITY LEVEL: Low' in explanation


def test_explanation_contains_active_symbols_section(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result('High', {'RisingPrices', 'ImminentCrash'})
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'ACTIVE SYMBOLS:' in explanation
    assert 'RisingPrices' in explanation


def test_explanation_inference_chain_listed(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result(
        'High',
        {'SpeculativeActivity', 'ImminentCrash'},
        fired_rules={'R1'},
        chain=[('R1', 'SpeculativeActivity')],
    )
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'INFERENCE CHAIN:' in explanation
    assert 'R1' in explanation
    assert 'SpeculativeActivity' in explanation


def test_explanation_no_chain_section_when_empty(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result('Low', set())
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'INFERENCE CHAIN:' not in explanation


# ---------------------------------------------------------------------------
# _generate_conclusion()
# ---------------------------------------------------------------------------

def test_conclusion_high_imminent_crash(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result('High', {'ImminentCrash'})
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'crash' in explanation.lower()


def test_conclusion_medium_high_crash_risk(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result('Medium', {'HighCrashRisk'})
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'bubble' in explanation.lower() or 'crash risk' in explanation.lower()


def test_conclusion_medium_liquidity_crisis(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result('Medium', {'LiquidityCrisis'})
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'liquidity' in explanation.lower()


def test_conclusion_low_stable(thresholds, features):
    gen = ExplanationGenerator()
    result = _make_result('Low', set())
    explanation = gen.generate_explanation(result, features, thresholds)
    assert 'stable' in explanation.lower()


# ---------------------------------------------------------------------------
# _get_symbol_context()
# ---------------------------------------------------------------------------

def test_get_symbol_context_known_symbol_returns_info(thresholds, features):
    gen = ExplanationGenerator()
    value, info = gen._get_symbol_context('RisingPrices', features, thresholds)
    assert info != ""
    assert 'price_slope' in info


def test_get_symbol_context_unknown_symbol_no_crash(thresholds, features):
    gen = ExplanationGenerator()
    value, info = gen._get_symbol_context('UnknownSymbol', features, thresholds)
    # Should return empty strings without raising
    assert value == ""
    assert info == ""
