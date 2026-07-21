"""Fuzzy logic for smooth symbolization transitions (optional enhancement)."""

from typing import Dict
import numpy as np


class FuzzySymbolizer:
    """Implements fuzzy membership functions for smooth symbolization."""
    
    @staticmethod
    def compute_membership(value: float, low: float, mid: float, high: float) -> float:
        """
        Compute triangular membership function value.
        
        Args:
            value: Input value
            low: Lower bound of triangle
            mid: Peak of triangle (membership = 1.0)
            high: Upper bound of triangle
        
        Returns:
            Membership value in [0, 1]
        """
        if value <= low or value >= high:
            return 0.0
        elif value == mid:
            return 1.0
        elif low < value < mid:
            return (value - low) / (mid - low)
        else:  # mid < value < high
            return (high - value) / (high - mid)
    
    def compute_fuzzy_symbols(
        self, 
        value: float, 
        thresholds: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute fuzzy membership for all symbols related to a feature.
        
        Args:
            value: Feature value
            thresholds: Dictionary with percentile thresholds
        
        Returns:
            Dictionary mapping symbol names to membership values
        """
        memberships = {}
        
        # Example: For volatility, create fuzzy sets
        # Low: [p10, p25, p40]
        # Medium: [p25, p50, p75]
        # High: [p70, p80, p90]
        
        if 'p10' in thresholds and 'p25' in thresholds and 'p40' in thresholds:
            memberships['Low'] = self.compute_membership(
                value, thresholds['p10'], thresholds['p25'], thresholds['p40']
            )
        
        if 'p25' in thresholds and 'p50' in thresholds and 'p75' in thresholds:
            # Use p40 as proxy for p50 if not available
            p50 = thresholds.get('p50', (thresholds['p25'] + thresholds['p75']) / 2)
            memberships['Medium'] = self.compute_membership(
                value, thresholds['p25'], p50, thresholds['p75']
            )
        
        if 'p70' in thresholds and 'p80' in thresholds and 'p90' in thresholds:
            memberships['High'] = self.compute_membership(
                value, thresholds['p70'], thresholds['p80'], thresholds['p90']
            )
        
        return memberships
