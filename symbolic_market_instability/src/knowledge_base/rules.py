"""Economic theory rules for market instability inference."""

from dataclasses import dataclass
from typing import List, Set


@dataclass
class Rule:
    """Represents an inference rule."""
    id: str
    antecedents: List[Set[str]]  # List of sets (conjunctions) that must all be present
    consequent: str
    confidence: float
    description: str
    
    def matches(self, active_symbols: Set[str]) -> bool:
        """
        Check if rule matches given active symbols.
        
        All antecedents (conjunctions) must be present in active_symbols.
        
        Args:
            active_symbols: Set of currently active symbols
        
        Returns:
            True if all antecedents are present
        """
        # All antecedent sets must be subsets of active_symbols
        for antecedent_set in self.antecedents:
            if not antecedent_set.issubset(active_symbols):
                return False
        return True


class EconomicTheoryRules:
    """Collection of rules based on Minsky's Financial Instability Hypothesis."""
    
    @staticmethod
    def get_instability_rules() -> List[Rule]:
        """
        Get all instability detection rules.
        
        Rules based on Minsky's Financial Instability Hypothesis:
        - R1: RisingPrices ∧ HighVolume → SpeculativeActivity
        - R2: ExcessiveGains ∧ HighVolume → SpeculativeActivity
        - R3: SpeculativeActivity ∧ HighVolatility → BubbleState
        - R4: BubbleState ∧ TightLiquidity → HighCrashRisk
        - R5: FallingPrices ∧ HighVolatility ∧ TightLiquidity → PanicState
        - R6: PanicState ∧ LowVolume → LiquidityCrisis
        - R7: HighCrashRisk ∧ ExcessiveLosses → ImminentCrash
        - R8: LiquidityCrisis ∧ FallingPrices → ImminentCrash
        
        Returns:
            List of Rule objects
        """
        rules = [
            Rule(
                id='R1',
                antecedents=[{'RisingPrices', 'HighVolume'}],
                consequent='SpeculativeActivity',
                confidence=0.8,
                description='Rising prices with high volume indicates speculation'
            ),
            Rule(
                id='R2',
                antecedents=[{'ExcessiveGains', 'HighVolume'}],
                consequent='SpeculativeActivity',
                confidence=0.75,
                description='Excessive gains with volume suggests irrational exuberance'
            ),
            Rule(
                id='R3',
                antecedents=[{'SpeculativeActivity', 'HighVolatility'}],
                consequent='BubbleState',
                confidence=0.85,
                description='Speculation with high volatility indicates bubble formation'
            ),
            Rule(
                id='R4',
                antecedents=[{'BubbleState', 'TightLiquidity'}],
                consequent='HighCrashRisk',
                confidence=0.9,
                description='Bubble with tight liquidity creates high crash risk'
            ),
            Rule(
                id='R5',
                antecedents=[{'FallingPrices', 'HighVolatility', 'TightLiquidity'}],
                consequent='PanicState',
                confidence=0.85,
                description='Falling prices with high volatility and tight liquidity indicates panic'
            ),
            Rule(
                id='R6',
                antecedents=[{'PanicState', 'LowVolume'}],
                consequent='LiquidityCrisis',
                confidence=0.8,
                description='Panic with low volume indicates liquidity crisis'
            ),
            Rule(
                id='R7',
                antecedents=[{'HighCrashRisk', 'ExcessiveLosses'}],
                consequent='ImminentCrash',
                confidence=0.95,
                description='High crash risk with excessive losses indicates imminent crash'
            ),
            Rule(
                id='R8',
                antecedents=[{'LiquidityCrisis', 'FallingPrices'}],
                consequent='ImminentCrash',
                confidence=0.9,
                description='Liquidity crisis with falling prices indicates imminent crash'
            )
        ]
        
        return rules
