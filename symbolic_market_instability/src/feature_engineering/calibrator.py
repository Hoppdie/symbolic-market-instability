"""Threshold calibration for feature symbolization."""

import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np


class ThresholdCalibrator:
    """Calibrates percentile thresholds for feature symbolization."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize calibrator.
        
        Args:
            config_dir: Directory containing thresholds.json. Defaults to config/
        """
        if config_dir is None:
            # Go up from src/feature_engineering/calibrator.py -> src/feature_engineering -> src -> symbolic_market_instability
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds_file = self.config_dir / "thresholds.json"
    
    def calibrate(self, features: pd.DataFrame) -> Dict:
        """
        Calibrate thresholds by computing percentiles for each feature.
        
        Percentiles computed: P10, P25, P40, P70, P75, P80, P90
        
        Args:
            features: DataFrame with feature columns:
                rolling_return, rolling_volatility, price_slope, volume_trend, vix
        
        Returns:
            Dictionary of thresholds in format:
            {
                "rolling_return": {"p10": -0.08, "p25": -0.03, ..., "p90": 0.12},
                ...
            }
        """
        feature_names = ['rolling_return', 'rolling_volatility', 'price_slope', 
                        'volume_trend', 'vix']
        
        thresholds = {}
        percentiles = [10, 25, 40, 70, 75, 80, 90]
        
        for feature_name in feature_names:
            if feature_name not in features.columns:
                raise ValueError(f"Feature {feature_name} not found in data")
            
            feature_values = features[feature_name].dropna()
            
            if len(feature_values) == 0:
                raise ValueError(f"No valid values for feature {feature_name}")
            
            # Compute percentiles
            feature_thresholds = {}
            for p in percentiles:
                feature_thresholds[f"p{p}"] = float(np.percentile(feature_values, p))
            
            thresholds[feature_name] = feature_thresholds
        
        # Save to file
        self._save_thresholds(thresholds)
        
        # Validate economic plausibility
        self.validate_economic_plausibility(thresholds)
        
        return thresholds
    
    def _save_thresholds(self, thresholds: Dict) -> None:
        """Save thresholds to JSON file."""
        with open(self.thresholds_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
    
    def load_thresholds(self) -> Dict:
        """
        Load thresholds from JSON file.
        
        Returns:
            Dictionary of thresholds
        
        Raises:
            FileNotFoundError: If thresholds.json doesn't exist
        """
        if not self.thresholds_file.exists():
            raise FileNotFoundError(
                f"Thresholds file not found: {self.thresholds_file}. "
                "Run calibration first."
            )
        
        with open(self.thresholds_file, 'r') as f:
            thresholds = json.load(f)
        
        return thresholds
    
    def validate_economic_plausibility(self, thresholds: Dict) -> None:
        """
        Validate that thresholds are economically plausible.
        
        Checks:
        - VIX p75 < 35 (VIX rarely exceeds 35 in normal times)
        - VIX p25 > 10 (VIX rarely below 10)
        
        Args:
            thresholds: Dictionary of thresholds
        
        Raises:
            ValueError: If thresholds fail validation
        """
        if 'vix' not in thresholds:
            raise ValueError("VIX thresholds not found")
        
        vix_thresholds = thresholds['vix']
        
        # Check VIX p75 < 35
        if vix_thresholds.get('p75', float('inf')) >= 35:
            raise ValueError(
                f"VIX p75 ({vix_thresholds.get('p75')}) should be < 35. "
                "This suggests unusual market conditions or data issues."
            )
        
        # Check VIX p25 > 10
        if vix_thresholds.get('p25', 0) <= 10:
            raise ValueError(
                f"VIX p25 ({vix_thresholds.get('p25')}) should be > 10. "
                "This suggests unusually low volatility or data issues."
            )
