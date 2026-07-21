"""Data preprocessing and cleaning utilities."""

from typing import Tuple
import pandas as pd
import numpy as np


class DataPreprocessor:
    """Preprocesses market data: cleaning, outlier detection, temporal splitting."""
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean market data: forward-fill missing values and detect outliers.
        
        Args:
            data: DataFrame with market data (Open, High, Low, Close, Volume, VIX)
        
        Returns:
            Cleaned DataFrame
        
        Raises:
            ValueError: If data contains invalid values (negative prices/volumes)
        """
        data = data.copy()
        
        # Validate: no negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in data.columns:
                if (data[col] < 0).any():
                    raise ValueError(f"Negative prices found in {col}")
        
        # Validate: no negative volumes
        if 'Volume' in data.columns:
            if (data['Volume'] < 0).any():
                raise ValueError("Negative volumes found")
        
        # Forward-fill missing values (max 3 consecutive days)
        data_cleaned = data.copy()
        for col in data.columns:
            # Count consecutive NaNs
            is_nan = data_cleaned[col].isna()
            groups = (is_nan != is_nan.shift()).cumsum()
            nan_counts = is_nan.groupby(groups).transform('sum')
            
            # Only forward-fill if <= 3 consecutive NaNs
            mask_to_fill = (is_nan) & (nan_counts <= 3)
            data_cleaned.loc[mask_to_fill, col] = data_cleaned[col].ffill()
            
            # Drop rows with more than 3 consecutive NaNs
            mask_to_drop = (is_nan) & (nan_counts > 3)
            data_cleaned = data_cleaned[~mask_to_drop]
        
        # Detect outliers using 5-sigma rule
        numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = data_cleaned[col].mean()
            std = data_cleaned[col].std()
            
            # Mark outliers (beyond 5 standard deviations)
            outliers = np.abs(data_cleaned[col] - mean) > 5 * std
            if outliers.any():
                # Replace outliers with median (more robust than mean)
                median = data_cleaned[col].median()
                data_cleaned.loc[outliers, col] = median
        
        return data_cleaned
    
    def split_temporal(
        self, 
        data: pd.DataFrame, 
        calib_ratio: float = 0.6, 
        valid_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally into calibration, validation, and test sets.
        
        Args:
            data: DataFrame indexed by date
            calib_ratio: Proportion for calibration (default: 0.6)
            valid_ratio: Proportion for validation (default: 0.2)
        
        Returns:
            Tuple of (calibration_data, validation_data, test_data)
        
        Raises:
            ValueError: If ratios don't sum to 1.0 or data is empty
        """
        if abs(calib_ratio + valid_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {calib_ratio + valid_ratio}")
        
        if data.empty:
            raise ValueError("Cannot split empty data")
        
        # Sort by date to ensure temporal order
        data_sorted = data.sort_index()
        n = len(data_sorted)
        
        calib_end = int(n * calib_ratio)
        valid_end = int(n * (calib_ratio + valid_ratio))
        
        calibration_data = data_sorted.iloc[:calib_end].copy()
        validation_data = data_sorted.iloc[calib_end:valid_end].copy()
        test_data = data_sorted.iloc[valid_end:].copy()
        
        return calibration_data, validation_data, test_data
