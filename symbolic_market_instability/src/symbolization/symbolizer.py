"""Feature symbolization: converting numeric features to symbolic states."""

from typing import Set, Dict
import pandas as pd
import numpy as np


class FeatureSymbolizer:
    """Converts numeric features to symbolic representations."""
    
    def __init__(self, thresholds: Dict):
        """
        Initialize symbolizer with thresholds.
        
        Args:
            thresholds: Dictionary of thresholds from ThresholdCalibrator
        """
        self.thresholds = thresholds
    
    def symbolize_row(self, row: pd.Series) -> Set[str]:
        """
        Convert a single row of features to a set of symbols.
        
        Symbolization rules:
        - Price: RisingPrices, FallingPrices, StablePrices
        - Volatility: HighVolatility, MediumVolatility, LowVolatility
        - Volume: HighVolume, LowVolume, NormalVolume
        - Liquidity: TightLiquidity, LooseLiquidity, NormalLiquidity
        - Returns: ExcessiveGains, ExcessiveLosses, NormalReturns
        
        Args:
            row: Series with feature values (price_slope, rolling_volatility, 
                 volume_trend, vix, rolling_return)
        
        Returns:
            Set of active symbols
        """
        symbols = set()
        
        # Price Symbols
        price_slope = row.get('price_slope', np.nan)
        if not np.isnan(price_slope):
            if price_slope > self.thresholds['price_slope']['p75']:
                symbols.add('RisingPrices')
            elif price_slope < self.thresholds['price_slope']['p25']:
                symbols.add('FallingPrices')
            else:
                symbols.add('StablePrices')
        
        # Volatility Symbols
        rolling_volatility = row.get('rolling_volatility', np.nan)
        if not np.isnan(rolling_volatility):
            if rolling_volatility > self.thresholds['rolling_volatility']['p80']:
                symbols.add('HighVolatility')
            elif rolling_volatility < self.thresholds['rolling_volatility']['p40']:
                symbols.add('LowVolatility')
            else:
                symbols.add('MediumVolatility')
        
        # Volume Symbols
        volume_trend = row.get('volume_trend', np.nan)
        if not np.isnan(volume_trend):
            if volume_trend > self.thresholds['volume_trend']['p70']:
                symbols.add('HighVolume')
            elif volume_trend < self.thresholds['volume_trend'].get('p30', 
                    self.thresholds['volume_trend']['p25']):  # Use p25 if p30 not available
                symbols.add('LowVolume')
            else:
                symbols.add('NormalVolume')
        
        # Liquidity Symbols (VIX-based)
        vix = row.get('vix', np.nan)
        if not np.isnan(vix):
            if vix > self.thresholds['vix']['p75']:
                symbols.add('TightLiquidity')
            elif vix < self.thresholds['vix']['p25']:
                symbols.add('LooseLiquidity')
            else:
                symbols.add('NormalLiquidity')
        
        # Return Symbols
        rolling_return = row.get('rolling_return', np.nan)
        if not np.isnan(rolling_return):
            if rolling_return > self.thresholds['rolling_return']['p90']:
                symbols.add('ExcessiveGains')
            elif rolling_return < self.thresholds['rolling_return']['p10']:
                symbols.add('ExcessiveLosses')
            else:
                symbols.add('NormalReturns')
        
        return symbols
    
    def symbolize_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Convert entire DataFrame of features to symbols.
        
        Args:
            features: DataFrame with feature columns
        
        Returns:
            DataFrame with original columns plus 'symbols' column (Set[str] per row)
        """
        symbolized = features.copy()
        
        # Apply symbolization to each row
        symbolized['symbols'] = symbolized.apply(
            self.symbolize_row, axis=1
        )
        
        return symbolized
    
    def check_temporal_persistence(
        self, 
        symbols_series: pd.Series, 
        state: str, 
        min_days: int = 3
    ) -> pd.Series:
        """
        Check if a symbol persists for at least min_days consecutive days.
        
        Args:
            symbols_series: Series where each element is a Set[str] of symbols
            state: Symbol to check for persistence (e.g., 'HighVolatility')
            min_days: Minimum consecutive days required (default: 3)
        
        Returns:
            Boolean Series indicating where state persists for >= min_days
        """
        # Create boolean series: True if state is present
        has_state = symbols_series.apply(lambda s: state in s if isinstance(s, set) else False)
        
        # Find consecutive runs
        persistence = pd.Series(False, index=symbols_series.index)
        
        # Group consecutive True values
        groups = (has_state != has_state.shift()).cumsum()
        
        for group_id in groups.unique():
            group_mask = groups == group_id
            if has_state[group_mask].iloc[0]:  # If this group is True
                group_length = group_mask.sum()
                if group_length >= min_days:
                    persistence[group_mask] = True
        
        return persistence
