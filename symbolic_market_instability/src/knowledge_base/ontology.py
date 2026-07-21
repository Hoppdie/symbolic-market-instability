"""Ontology definitions for market entities and relationships."""

from typing import Dict, List, Set
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Market:
    """Represents a financial market."""
    name: str
    ticker: str
    region: str


@dataclass
class Indicator:
    """Represents a market indicator."""
    name: str
    symbol: str
    category: str  # 'price', 'volatility', 'volume', 'liquidity', 'returns'


@dataclass
class State:
    """Represents a market state."""
    name: str
    description: str
    category: str  # 'observed', 'derived'


@dataclass
class Event:
    """Represents a market event."""
    name: str
    date: datetime
    description: str


class MarketOntology:
    """Defines market entities and relationships."""
    
    def __init__(self):
        """Initialize ontology with market entities."""
        self.markets: Dict[str, Market] = {}
        self.indicators: Dict[str, Indicator] = {}
        self.states: Dict[str, State] = {}
        self.events: Dict[str, Event] = {}
        
        self._initialize_entities()
    
    def _initialize_entities(self) -> None:
        """Initialize standard market entities."""
        # Markets
        self.markets['SP500'] = Market('S&P 500', '^GSPC', 'US')
        
        # Indicators
        self.indicators['VIX'] = Indicator('Volatility Index', 'VIX', 'liquidity')
        self.indicators['Price'] = Indicator('Price', 'Close', 'price')
        self.indicators['Volume'] = Indicator('Volume', 'Volume', 'volume')
        
        # Observed States
        self.states['RisingPrices'] = State('RisingPrices', 'Prices trending upward', 'observed')
        self.states['FallingPrices'] = State('FallingPrices', 'Prices trending downward', 'observed')
        self.states['StablePrices'] = State('StablePrices', 'Prices relatively stable', 'observed')
        self.states['HighVolatility'] = State('HighVolatility', 'High price volatility', 'observed')
        self.states['MediumVolatility'] = State('MediumVolatility', 'Moderate volatility', 'observed')
        self.states['LowVolatility'] = State('LowVolatility', 'Low volatility', 'observed')
        self.states['HighVolume'] = State('HighVolume', 'High trading volume', 'observed')
        self.states['LowVolume'] = State('LowVolume', 'Low trading volume', 'observed')
        self.states['NormalVolume'] = State('NormalVolume', 'Normal trading volume', 'observed')
        self.states['TightLiquidity'] = State('TightLiquidity', 'Tight liquidity (high VIX)', 'observed')
        self.states['LooseLiquidity'] = State('LooseLiquidity', 'Loose liquidity (low VIX)', 'observed')
        self.states['NormalLiquidity'] = State('NormalLiquidity', 'Normal liquidity', 'observed')
        self.states['ExcessiveGains'] = State('ExcessiveGains', 'Excessive positive returns', 'observed')
        self.states['ExcessiveLosses'] = State('ExcessiveLosses', 'Excessive negative returns', 'observed')
        self.states['NormalReturns'] = State('NormalReturns', 'Normal returns', 'observed')
        
        # Derived States
        self.states['SpeculativeActivity'] = State('SpeculativeActivity', 'Market speculation detected', 'derived')
        self.states['BubbleState'] = State('BubbleState', 'Market bubble conditions', 'derived')
        self.states['HighCrashRisk'] = State('HighCrashRisk', 'High risk of market crash', 'derived')
        self.states['PanicState'] = State('PanicState', 'Market panic conditions', 'derived')
        self.states['LiquidityCrisis'] = State('LiquidityCrisis', 'Liquidity crisis detected', 'derived')
        self.states['ImminentCrash'] = State('ImminentCrash', 'Imminent market crash', 'derived')
    
    def has_state(self, market: str, state: str, timestamp: datetime) -> bool:
        """
        Check if market has a state at a given time.
        
        Args:
            market: Market identifier
            state: State name
            timestamp: Time point
        
        Returns:
            True if market has state at timestamp
        """
        # This is a placeholder - actual implementation would query state history
        return False
    
    def causes(self, state1: str, state2: str) -> bool:
        """
        Check if state1 causes state2 (based on rules).
        
        Args:
            state1: Causal state
            state2: Resulting state
        
        Returns:
            True if state1 causes state2
        """
        # This would be determined by rule antecedents/consequents
        return False
    
    def increases_risk_of(self, state: str, event: str) -> bool:
        """
        Check if state increases risk of event.
        
        Args:
            state: Market state
            event: Event name
        
        Returns:
            True if state increases risk
        """
        # High-level states increase crash risk
        high_risk_states = ['HighCrashRisk', 'LiquidityCrisis', 'ImminentCrash']
        return state in high_risk_states
