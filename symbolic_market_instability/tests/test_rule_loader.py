"""Tests for RuleLoader."""

import pytest
import yaml
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.knowledge_base.loader import RuleLoader
from src.knowledge_base.rules import Rule

REAL_CONFIG_DIR = project_root / "config"


# ---------------------------------------------------------------------------
# Load from the real rules.yaml
# ---------------------------------------------------------------------------

def test_load_rules_returns_list():
    """load_rules() must return a list."""
    loader = RuleLoader(config_dir=str(REAL_CONFIG_DIR))
    rules = loader.load_rules()
    assert isinstance(rules, list)


def test_load_rules_count():
    """The project's rules.yaml must contain exactly 8 rules."""
    loader = RuleLoader(config_dir=str(REAL_CONFIG_DIR))
    rules = loader.load_rules()
    assert len(rules) == 8


def test_load_rules_are_rule_instances():
    """Every item returned must be a Rule dataclass instance."""
    loader = RuleLoader(config_dir=str(REAL_CONFIG_DIR))
    rules = loader.load_rules()
    for rule in rules:
        assert isinstance(rule, Rule)


def test_load_rules_antecedents_are_sets():
    """Each rule's antecedents must be a list of sets."""
    loader = RuleLoader(config_dir=str(REAL_CONFIG_DIR))
    rules = loader.load_rules()
    for rule in rules:
        assert isinstance(rule.antecedents, list)
        for ant in rule.antecedents:
            assert isinstance(ant, set), f"Rule {rule.id}: antecedent is not a set"


def test_load_rules_ids_unique():
    """All rule IDs must be unique."""
    loader = RuleLoader(config_dir=str(REAL_CONFIG_DIR))
    rules = loader.load_rules()
    ids = [r.id for r in rules]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Error cases using temp directories
# ---------------------------------------------------------------------------

def test_load_rules_missing_file_raises(tmp_path):
    """Non-existent config directory must raise FileNotFoundError."""
    loader = RuleLoader(config_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load_rules()


def test_load_rules_missing_rules_key_raises(tmp_path):
    """YAML file without a 'rules' key must raise ValueError."""
    bad_yaml = tmp_path / "rules.yaml"
    bad_yaml.write_text(yaml.dump({'not_rules': []}))
    loader = RuleLoader(config_dir=str(tmp_path))
    with pytest.raises(ValueError, match="rules"):
        loader.load_rules()
