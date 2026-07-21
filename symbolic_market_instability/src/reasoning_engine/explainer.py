"""Explanation generation for inference results."""

from typing import Dict, Set
import pandas as pd


class ExplanationGenerator:
    """Generates human-readable explanations of inference results."""
    
    def generate_explanation(
        self, 
        inference_result: Dict, 
        original_features: pd.Series,
        thresholds: Dict
    ) -> str:
        """
        Generate human-readable explanation of inference.
        
        Args:
            inference_result: Result from ForwardChainer.infer()
            original_features: Original feature values for the day
            thresholds: Threshold dictionary for context
        
        Returns:
            Multi-line explanation string
        """
        lines = []
        
        # Header
        instability_level = inference_result['instability_level']
        lines.append(f"INSTABILITY LEVEL: {instability_level}")
        lines.append("")
        
        # Active symbols with feature values
        all_facts = inference_result['all_facts']
        observed_symbols = [s for s in all_facts if s in [
            'RisingPrices', 'FallingPrices', 'StablePrices',
            'HighVolatility', 'MediumVolatility', 'LowVolatility',
            'HighVolume', 'LowVolume', 'NormalVolume',
            'TightLiquidity', 'LooseLiquidity', 'NormalLiquidity',
            'ExcessiveGains', 'ExcessiveLosses', 'NormalReturns'
        ]]
        
        if observed_symbols:
            lines.append("ACTIVE SYMBOLS:")
            for symbol in sorted(observed_symbols):
                feature_value, percentile_info = self._get_symbol_context(
                    symbol, original_features, thresholds
                )
                lines.append(f"  - {symbol} {percentile_info}")
            lines.append("")
        
        # Inference chain
        inference_chain = inference_result['inference_chain']
        if inference_chain:
            lines.append("INFERENCE CHAIN:")
            for i, (rule_id, consequent) in enumerate(inference_chain, 1):
                lines.append(f"  {i}. [{rule_id}] → {consequent}")
            lines.append("")
        
        # Conclusion
        conclusion = self._generate_conclusion(inference_result, all_facts)
        lines.append(f"CONCLUSION: {conclusion}")
        
        return "\n".join(lines)
    
    def _get_symbol_context(
        self, 
        symbol: str, 
        features: pd.Series, 
        thresholds: Dict
    ) -> tuple:
        """
        Get feature value and percentile context for a symbol.
        
        Args:
            symbol: Symbol name
            features: Feature values
            thresholds: Threshold dictionary
        
        Returns:
            Tuple of (feature_value_str, percentile_info_str)
        """
        # Map symbols to features
        symbol_to_feature = {
            'RisingPrices': 'price_slope',
            'FallingPrices': 'price_slope',
            'StablePrices': 'price_slope',
            'HighVolatility': 'rolling_volatility',
            'MediumVolatility': 'rolling_volatility',
            'LowVolatility': 'rolling_volatility',
            'HighVolume': 'volume_trend',
            'LowVolume': 'volume_trend',
            'NormalVolume': 'volume_trend',
            'TightLiquidity': 'vix',
            'LooseLiquidity': 'vix',
            'NormalLiquidity': 'vix',
            'ExcessiveGains': 'rolling_return',
            'ExcessiveLosses': 'rolling_return',
            'NormalReturns': 'rolling_return'
        }
        
        feature_name = symbol_to_feature.get(symbol)
        if not feature_name or feature_name not in features:
            return ("", "")
        
        value = features[feature_name]
        
        # Determine which percentile threshold was crossed
        percentile_info = ""
        if feature_name in thresholds:
            thresh = thresholds[feature_name]
            
            if symbol in ['RisingPrices', 'HighVolatility', 'HighVolume', 
                         'TightLiquidity', 'ExcessiveGains']:
                # High-end symbols
                if symbol == 'RisingPrices' and value > thresh.get('p75', 0):
                    percentile_info = f"(price_slope={value:.4f}, above 75th percentile)"
                elif symbol == 'HighVolatility' and value > thresh.get('p80', 0):
                    percentile_info = f"(volatility={value:.4f}, above 80th percentile)"
                elif symbol == 'HighVolume' and value > thresh.get('p70', 0):
                    percentile_info = f"(volume_trend={value:.2f}, above 70th percentile)"
                elif symbol == 'TightLiquidity' and value > thresh.get('p75', 0):
                    percentile_info = f"(VIX={value:.1f}, above 75th percentile)"
                elif symbol == 'ExcessiveGains' and value > thresh.get('p90', 0):
                    percentile_info = f"(rolling_return={value:.4f}, above 90th percentile)"
            elif symbol in ['FallingPrices', 'LowVolatility', 'LowVolume',
                           'LooseLiquidity', 'ExcessiveLosses']:
                # Low-end symbols
                if symbol == 'FallingPrices' and value < thresh.get('p25', 0):
                    percentile_info = f"(price_slope={value:.4f}, below 25th percentile)"
                elif symbol == 'LowVolatility' and value < thresh.get('p40', 0):
                    percentile_info = f"(volatility={value:.4f}, below 40th percentile)"
                elif symbol == 'LowVolume':
                    p30 = thresh.get('p30', thresh.get('p25', 0))
                    if value < p30:
                        percentile_info = f"(volume_trend={value:.2f}, below 30th percentile)"
                elif symbol == 'LooseLiquidity' and value < thresh.get('p25', 0):
                    percentile_info = f"(VIX={value:.1f}, below 25th percentile)"
                elif symbol == 'ExcessiveLosses' and value < thresh.get('p10', 0):
                    percentile_info = f"(rolling_return={value:.4f}, below 10th percentile)"
        
        if not percentile_info:
            percentile_info = f"({feature_name}={value:.4f})"
        
        return (value, percentile_info)
    
    def _generate_conclusion(
        self, 
        inference_result: Dict, 
        all_facts: Set[str]
    ) -> str:
        """
        Generate conclusion text based on inference results.
        
        Args:
            inference_result: Inference result dictionary
            all_facts: All active facts
        
        Returns:
            Conclusion string
        """
        level = inference_result['instability_level']
        
        if level == 'High':
            if 'ImminentCrash' in all_facts:
                return ("Imminent crash detected due to combination of high crash risk "
                       "or liquidity crisis with falling prices and excessive losses.")
            else:
                return "High instability detected."
        elif level == 'Medium':
            if 'HighCrashRisk' in all_facts:
                return ("High crash risk detected due to bubble conditions with tight liquidity.")
            elif 'LiquidityCrisis' in all_facts:
                return ("Liquidity crisis detected: panic selling with low trading volume.")
            else:
                return "Medium instability detected."
        else:
            return ("Market conditions appear stable with no significant instability indicators.")
