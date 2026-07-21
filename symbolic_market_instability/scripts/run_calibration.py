"""Download data, compute features, calibrate thresholds, save to config/"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data_acquisition.downloader import MarketDataDownloader
from src.data_acquisition.preprocessor import DataPreprocessor
from src.feature_engineering.technical_features import TechnicalFeatureComputer
from src.feature_engineering.calibrator import ThresholdCalibrator
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    """Run calibration pipeline."""
    logger.info("Starting calibration pipeline...")
    
    # Configuration
    index_ticker = "^GSPC"  # S&P 500
    start_date = "2000-01-01"
    end_date = "2015-12-31"  # Calibration period
    market = "US"
    
    # Step 1: Download data
    logger.info(f"Downloading {index_ticker} data from {start_date} to {end_date}...")
    downloader = MarketDataDownloader()
    data = downloader.get_combined_data(index_ticker, start_date, end_date, market)
    logger.info(f"Downloaded {len(data)} days of data")
    
    # Step 2: Preprocess
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    data_clean = preprocessor.clean(data)
    logger.info(f"Cleaned data: {len(data_clean)} days")
    
    # Step 3: Compute features
    logger.info("Computing technical features...")
    feature_computer = TechnicalFeatureComputer()
    features = feature_computer.compute_all_features(data_clean)
    logger.info(f"Computed features for {len(features)} days")
    
    # Step 4: Calibrate thresholds
    logger.info("Calibrating thresholds...")
    calibrator = ThresholdCalibrator()
    thresholds = calibrator.calibrate(features)
    logger.info("Thresholds calibrated and saved to config/thresholds.json")
    
    # Log summary
    logger.info("Calibration Summary:")
    for feature_name, thresh_dict in thresholds.items():
        logger.info(f"  {feature_name}: p10={thresh_dict['p10']:.4f}, "
                   f"p90={thresh_dict['p90']:.4f}")
    
    logger.info("Calibration complete!")


if __name__ == "__main__":
    main()
