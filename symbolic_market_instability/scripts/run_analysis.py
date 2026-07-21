"""Run full pipeline on test data"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.data_acquisition.downloader import MarketDataDownloader
from src.data_acquisition.preprocessor import DataPreprocessor
from src.feature_engineering.technical_features import TechnicalFeatureComputer
from src.feature_engineering.calibrator import ThresholdCalibrator
from src.symbolization.symbolizer import FeatureSymbolizer
from src.knowledge_base.rules import EconomicTheoryRules
from src.knowledge_base.loader import RuleLoader
from src.reasoning_engine.forward_chainer import ForwardChainer
from src.reasoning_engine.explainer import ExplanationGenerator
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    """Run full analysis pipeline."""
    logger.info("Starting analysis pipeline...")
    
    # Configuration
    index_ticker = "^GSPC"
    start_date = "2000-01-01"  # Extended to include 2008 crisis
    end_date = "2024-12-31"  # Test period
    market = "US"
    
    # Step 1: Load thresholds
    logger.info("Loading thresholds...")
    calibrator = ThresholdCalibrator()
    thresholds = calibrator.load_thresholds()
    logger.info("Thresholds loaded")
    
    # Step 2: Download test data
    logger.info(f"Downloading {index_ticker} data from {start_date} to {end_date}...")
    downloader = MarketDataDownloader()
    data = downloader.get_combined_data(index_ticker, start_date, end_date, market)
    logger.info(f"Downloaded {len(data)} days of data")
    
    # Step 3: Preprocess
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    data_clean = preprocessor.clean(data)
    logger.info(f"Cleaned data: {len(data_clean)} days")
    
    # Step 4: Compute features
    logger.info("Computing technical features...")
    feature_computer = TechnicalFeatureComputer()
    features = feature_computer.compute_all_features(data_clean)
    logger.info(f"Computed features for {len(features)} days")
    
    # Step 5: Symbolize
    logger.info("Symbolizing features...")
    symbolizer = FeatureSymbolizer(thresholds)
    symbolized = symbolizer.symbolize_dataframe(features)
    logger.info("Features symbolized")
    
    # Step 6: Load rules
    logger.info("Loading rules...")
    rule_loader = RuleLoader()
    rules = rule_loader.load_rules()
    logger.info(f"Loaded {len(rules)} rules")
    
    # Step 7: Initialize reasoning engine
    logger.info("Initializing reasoning engine...")
    chainer = ForwardChainer(rules)
    explainer = ExplanationGenerator()
    
    # Step 8: Run inference for each day
    logger.info("Running inference for each day...")
    results = []
    
    for date, row in symbolized.iterrows():
        # Normalize date index (remove timezone if present) for clean CSV output
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            date_value = date.tz_convert(None) if hasattr(date, "tz_convert") else date.tz_localize(None)
        else:
            date_value = date

        # Get active symbols
        active_symbols = row['symbols']
        
        # Run inference
        inference_result = chainer.infer(active_symbols)
        
        # Generate explanation
        explanation = explainer.generate_explanation(
            inference_result, 
            row,
            thresholds
        )
        
        # Store results
        results.append({
            'date': date_value,
            # Include a close price column for downstream analysis / evaluation
            'close_price': float(row['Close']) if 'Close' in row else None,
            'instability_level': inference_result['instability_level'],
            'fired_rules': ','.join(inference_result['fired_rules']),
            'symbols': ','.join(sorted(active_symbols)),
            'derived_facts': ','.join(sorted(inference_result['derived_facts'])),
            'explanation': explanation
        })
        
        if len(results) % 100 == 0:
            logger.info(f"Processed {len(results)} days...")
    
    # Step 9: Save results
    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    
    results_file = project_root / "results" / "analysis_results.csv"
    results_df.to_csv(results_file)
    logger.info(f"Results saved to {results_file}")
    
    # Save explanations separately
    explanations_dir = project_root / "results" / "explanations"
    explanations_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a sample explanation (3 days before a high instability day)
    high_instability_days = results_df[results_df['instability_level'] == 'High']
    if not high_instability_days.empty:
        sample_date = high_instability_days.index[0]
        sample_explanation = results_df.loc[sample_date, 'explanation']
        
        # Convert date to string safely

        date_str = sample_date.strftime('%Y%m%d') if hasattr(sample_date, 'strftime') else str(sample_date).replace('-', '')
        explanation_file = explanations_dir / f"explanation_{date_str}.txt"
        with open(explanation_file, 'w', encoding='utf-8') as f:
            f.write(f"Date: {sample_date}\n\n")
            f.write(sample_explanation)
        logger.info(f"Sample explanation saved to {explanation_file}")
    
    # Summary statistics
    logger.info("\nAnalysis Summary:")
    logger.info(f"Total days analyzed: {len(results_df)}")
    logger.info(f"High instability days: {(results_df['instability_level'] == 'High').sum()}")
    logger.info(f"Medium instability days: {(results_df['instability_level'] == 'Medium').sum()}")
    logger.info(f"Low instability days: {(results_df['instability_level'] == 'Low').sum()}")
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
