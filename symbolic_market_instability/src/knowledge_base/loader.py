"""Load rules from YAML configuration files."""

from pathlib import Path
from typing import List, Optional
import yaml
from .rules import Rule


class RuleLoader:
    """Loads rules from YAML configuration."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize rule loader.
        
        Args:
            config_dir: Directory containing rules.yaml. Defaults to config/
        """
        if config_dir is None:
            # Go up from src/knowledge_base/loader.py -> src/knowledge_base -> src -> symbolic_market_instability
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self.rules_file = self.config_dir / "rules.yaml"
    
    def load_rules(self) -> List[Rule]:
        """
        Load rules from YAML file.
        
        Returns:
            List of Rule objects
        
        Raises:
            FileNotFoundError: If rules.yaml doesn't exist
            ValueError: If YAML structure is invalid
        """
        if not self.rules_file.exists():
            raise FileNotFoundError(
                f"Rules file not found: {self.rules_file}"
            )
        
        with open(self.rules_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'rules' not in config:
            raise ValueError("YAML file must contain 'rules' key")
        
        rules = []
        for rule_dict in config['rules']:
            # Convert antecedents from list of lists to list of sets
            antecedents = [set(ant) for ant in rule_dict['antecedents']]
            
            rule = Rule(
                id=rule_dict['id'],
                antecedents=antecedents,
                consequent=rule_dict['consequent'],
                confidence=rule_dict.get('confidence', 1.0),
                description=rule_dict.get('description', '')
            )
            rules.append(rule)
        
        return rules
