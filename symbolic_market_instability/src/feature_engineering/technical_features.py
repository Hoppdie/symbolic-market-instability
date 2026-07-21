"""Technical feature computation for market data."""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict


class TechnicalFeatureComputer:
    """Computes technical features from market data."""
    
    def compute_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features from market data.
        
        Features computed:
        - F1: rolling_return = (Close_t - Close_t-30) / Close_t-30
        - F2: rolling_volatility = std(daily_returns over 30 days)
        - F3: price_slope = linear regression slope of Close over 30 days
        - F4: volume_trend = (Volume_t - MA_30(Volume)) / MA_30(Volume)
        - F5: vix = VIX_t (already in data)
        
        Args:
            data: DataFrame with columns: Open, High, Low, Close, Volume, VIX
        
        Returns:
            DataFrame with original columns plus 5 new feature columns
        """
        features = data.copy()
        
        # F1: Rolling return (30-day)
        features['rolling_return'] = (
            (features['Close'] - features['Close'].shift(30)) / 
            features['Close'].shift(30)
        )
        
        # F2: Rolling volatility (30-day standard deviation of daily returns)
        daily_returns = features['Close'].pct_change()
        features['rolling_volatility'] = daily_returns.rolling(window=30).std()
        
        # F3: Price slope (linear regression slope over 30 days)
        features['price_slope'] = features['Close'].rolling(window=30).apply(
            self._compute_slope, raw=True
        )
        
        # F4: Volume trend (deviation from 30-day moving average)
        volume_ma_30 = features['Volume'].rolling(window=30).mean()
        features['volume_trend'] = (
            (features['Volume'] - volume_ma_30) / volume_ma_30
        )
        
        # F5: VIX (already in data, just rename for clarity)
        if 'VIX' in features.columns:
            features['vix'] = features['VIX']
        elif 'vix' not in features.columns:
            raise ValueError("VIX column not found in data")
        
        # Drop NaN rows after rolling calculations
        features = features.dropna()
        
        return features
    
    def _compute_slope(self, values: np.ndarray) -> float:
        """
        Compute linear regression slope for a window of values.
        
        Args:
            values: Array of values (typically 30 days of prices)
        
        Returns:
            Slope of linear regression line
        """
        if len(values) < 2 or np.isnan(values).any():
            return np.nan
        
        # Create x-axis (days: 0, 1, 2, ..., n-1)
        x = np.arange(len(values))
        
        # Remove NaN values
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan
        
        x_clean = x[mask]
        y_clean = values[mask]
        
        # Compute slope using linear regression
        slope, _, _, _, _ = linregress(x_clean, y_clean)
        
        return slope
