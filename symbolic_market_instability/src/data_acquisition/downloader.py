"""Market data downloader using yfinance."""

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf
from datetime import datetime


class MarketDataDownloader:
    """Downloads market data from Yahoo Finance and caches locally."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the downloader.
        
        Args:
            cache_dir: Directory to cache downloaded data. Defaults to data/raw/
        """
        if cache_dir is None:
            # Get project root: src/data_acquisition/downloader.py -> src/data_acquisition -> src -> symbolic_market_instability
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "data" / "raw"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_index(data):
        """
        Coerce a date index to tz-naive midnight timestamps.

        yfinance returns tz-aware timestamps (America/New_York). Once cached
        to CSV, ranges that span a DST change contain mixed UTC offsets,
        which pandas re-reads as strings — an inner join between a cached
        frame and a fresh one then matches nothing. Stripping the tz on both
        paths keeps every index comparable.
        """
        idx = pd.to_datetime(data.index, utc=True, errors='coerce')
        data = data[~idx.isna()]
        data.index = idx[~idx.isna()].tz_localize(None).normalize()
        data.index.name = 'Date'
        return data

    def download_index(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Download index data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol (e.g., '^GSPC' for S&P 500)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        
        Raises:
            ValueError: If download fails or returns empty data
        """
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}.csv"
        
        # Check cache first
        if cache_file.exists():
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return self._normalize_index(data)

        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}")

            # Select required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in downloaded data: {missing_cols}")

            data = self._normalize_index(data[required_cols].copy())
            
            # Cache the data
            data.to_csv(cache_file)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to download data for {ticker}: {str(e)}")
    
    def download_vix(
        self, 
        start_date: str, 
        end_date: str, 
        market: str = "US"
    ) -> pd.Series:
        """
        Download VIX (Volatility Index) data.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            market: Market identifier (default: "US" for ^VIX)
        
        Returns:
            Series with VIX values indexed by date
        
        Raises:
            ValueError: If download fails
        """
        cache_file = self.cache_dir / f"VIX_{market}_{start_date}_{end_date}.csv"
        
        # Check cache first
        if cache_file.exists():
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            data = self._normalize_index(data)
            return data.squeeze() if len(data.columns) == 1 else data.iloc[:, 0]

        try:
            vix_ticker = "^VIX" if market == "US" else f"VIX{market}"
            ticker_obj = yf.Ticker(vix_ticker)
            data = ticker_obj.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No VIX data returned from {start_date} to {end_date}")

            # Extract Close price as VIX value
            vix_series = self._normalize_index(data)['Close'].copy()
            vix_series.name = 'VIX'
            
            # Cache the data
            vix_series.to_frame().to_csv(cache_file)
            
            return vix_series
            
        except Exception as e:
            raise ValueError(f"Failed to download VIX data: {str(e)}")
    
    def get_combined_data(
        self, 
        index_ticker: str, 
        start_date: str, 
        end_date: str, 
        market: str = "US"
    ) -> pd.DataFrame:
        """
        Download and combine index data with VIX.
        
        Args:
            index_ticker: Index ticker (e.g., '^GSPC')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            market: Market identifier for VIX
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, VIX
        """
        # Download index data
        index_data = self.download_index(index_ticker, start_date, end_date)
        
        # Download VIX data
        vix_data = self.download_vix(start_date, end_date, market)
        
        # Merge on date index
        combined = index_data.join(vix_data, how='inner')
        
        if combined.empty:
            raise ValueError("Combined data is empty after merging index and VIX")
        
        return combined
