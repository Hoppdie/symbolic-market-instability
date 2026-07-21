"""Pattern matching for rule antecedents."""

from typing import Set
from ..knowledge_base.rules import Rule


class PatternMatcher:
    """Matches rule patterns against active facts."""
    
    @staticmethod
    def match_rule(rule: Rule, facts: Set[str]) -> bool:
        """
        Check if a rule matches given facts.
        
        Args:
            rule: Rule to check
            facts: Set of active facts/symbols
        
        Returns:
            True if rule matches (all antecedents present)
        """
        return rule.matches(facts)
