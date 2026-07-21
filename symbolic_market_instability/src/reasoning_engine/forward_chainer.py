"""Forward-chaining inference engine."""

from typing import Set, List, Tuple, Dict
from ..knowledge_base.rules import Rule


class ForwardChainer:
    """Forward-chaining inference engine for market instability detection."""
    
    def __init__(self, rules: List[Rule]):
        """
        Initialize forward chainer with rules.
        
        Args:
            rules: List of inference rules
        """
        self.rules = rules
    
    def infer(self, facts: Set[str]) -> Dict:
        """
        Perform forward-chaining inference from initial facts.
        
        Algorithm:
        1. Start with initial_facts
        2. working_memory = initial_facts.copy()
        3. fired_rules = []
        4. WHILE True:
             new_facts = set()
             FOR each rule in rules:
                 IF rule.matches(working_memory) AND rule.id not in fired_rules:
                     new_facts.add(rule.consequent)
                     fired_rules.append(rule.id)
                     working_memory.add(rule.consequent)
             IF new_facts is empty: BREAK
        5. Determine instability level
        
        Args:
            facts: Initial set of active symbols/facts
        
        Returns:
            Dictionary with keys:
            - 'derived_facts': Set[str] of all derived facts
            - 'fired_rules': List[str] of rule IDs that fired
            - 'inference_chain': List[Tuple[str, str]] of (rule_id, consequent)
            - 'instability_level': 'Low' | 'Medium' | 'High'
        """
        working_memory = facts.copy()
        fired_rules = []
        inference_chain = []
        
        # Forward-chaining loop
        while True:
            new_facts = set()
            
            for rule in self.rules:
                # Check if rule matches and hasn't fired yet
                if rule.matches(working_memory) and rule.id not in fired_rules:
                    consequent = rule.consequent
                    new_facts.add(consequent)
                    fired_rules.append(rule.id)
                    working_memory.add(consequent)
                    inference_chain.append((rule.id, consequent))
            
            # If no new facts were derived, stop
            if not new_facts:
                break
        
        # Determine instability level
        instability_level = self._determine_instability_level(working_memory)
        
        return {
            'derived_facts': working_memory - facts,  # Only newly derived facts
            'all_facts': working_memory,  # All facts (initial + derived)
            'fired_rules': fired_rules,
            'inference_chain': inference_chain,
            'instability_level': instability_level
        }
    
    def _determine_instability_level(self, facts: Set[str]) -> str:
        """
        Determine instability level from derived facts.
        
        Rules:
        - IF 'ImminentCrash' in facts → 'High'
        - ELIF 'HighCrashRisk' or 'LiquidityCrisis' in facts → 'Medium'
        - ELSE → 'Low'
        
        Args:
            facts: Set of all facts (initial + derived)
        
        Returns:
            Instability level: 'Low', 'Medium', or 'High'
        """
        if 'ImminentCrash' in facts:
            return 'High'
        elif 'HighCrashRisk' in facts or 'LiquidityCrisis' in facts:
            return 'Medium'
        else:
            return 'Low'
