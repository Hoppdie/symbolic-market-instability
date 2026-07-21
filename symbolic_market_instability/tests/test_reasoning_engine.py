"""Tests for reasoning engine."""

import pytest
import sys
from pathlib import Path

# Add project root to path so we can import the src package
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_base.rules import Rule, EconomicTheoryRules
from src.reasoning_engine.forward_chainer import ForwardChainer


def test_rule_matching():
    """Test rule matching logic."""
    rule = Rule(
        id='R1',
        antecedents=[{'RisingPrices', 'HighVolume'}],
        consequent='SpeculativeActivity',
        confidence=0.8,
        description='Test rule'
    )
    
    # Should match when both antecedents present
    assert rule.matches({'RisingPrices', 'HighVolume', 'OtherSymbol'})
    
    # Should not match when one missing
    assert not rule.matches({'RisingPrices'})
    assert not rule.matches({'HighVolume'})
    
    # Should not match when neither present
    assert not rule.matches({'OtherSymbol'})


def test_forward_chaining_basic():
    """Test basic forward chaining."""
    rules = EconomicTheoryRules.get_instability_rules()
    chainer = ForwardChainer(rules)
    
    # Test with RisingPrices and HighVolume (should trigger R1)
    facts = {'RisingPrices', 'HighVolume'}
    result = chainer.infer(facts)
    
    assert 'SpeculativeActivity' in result['all_facts']
    assert 'R1' in result['fired_rules']


def test_forward_chaining_chain():
    """Test forward chaining with multiple rule applications."""
    rules = EconomicTheoryRules.get_instability_rules()
    chainer = ForwardChainer(rules)
    
    # Test full chain: RisingPrices + HighVolume + HighVolatility + TightLiquidity
    facts = {'RisingPrices', 'HighVolume', 'HighVolatility', 'TightLiquidity'}
    result = chainer.infer(facts)
    
    # Should derive: SpeculativeActivity -> BubbleState -> HighCrashRisk
    assert 'SpeculativeActivity' in result['all_facts']
    assert 'BubbleState' in result['all_facts']
    assert 'HighCrashRisk' in result['all_facts']
    
    # Should fire R1, R3, R4
    assert 'R1' in result['fired_rules']
    assert 'R3' in result['fired_rules']
    assert 'R4' in result['fired_rules']


def test_instability_level_high():
    """Test instability level assignment for high risk."""
    rules = EconomicTheoryRules.get_instability_rules()
    chainer = ForwardChainer(rules)
    
    # Create scenario leading to ImminentCrash
    facts = {
        'RisingPrices', 'HighVolume', 'HighVolatility', 'TightLiquidity',
        'ExcessiveLosses'  # This with HighCrashRisk triggers R7 -> ImminentCrash
    }
    result = chainer.infer(facts)
    
    assert result['instability_level'] == 'High'
    assert 'ImminentCrash' in result['all_facts']


def test_instability_level_medium():
    """Test instability level assignment for medium risk."""
    rules = EconomicTheoryRules.get_instability_rules()
    chainer = ForwardChainer(rules)
    
    # Create scenario leading to HighCrashRisk (but not ImminentCrash)
    facts = {'RisingPrices', 'HighVolume', 'HighVolatility', 'TightLiquidity'}
    result = chainer.infer(facts)
    
    assert result['instability_level'] == 'Medium'
    assert 'HighCrashRisk' in result['all_facts']


def test_instability_level_low():
    """Test instability level assignment for low risk."""
    rules = EconomicTheoryRules.get_instability_rules()
    chainer = ForwardChainer(rules)
    
    # Create scenario with no derived high-risk states
    facts = {'StablePrices', 'MediumVolatility', 'NormalVolume', 'NormalLiquidity'}
    result = chainer.infer(facts)
    
    assert result['instability_level'] == 'Low'


if __name__ == "__main__":
    pytest.main([__file__])
