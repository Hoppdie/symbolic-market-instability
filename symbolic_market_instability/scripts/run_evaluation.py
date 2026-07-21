"""Evaluate on historical crashes"""

import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.evaluation.case_studies import CaseStudyEvaluator
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    """Run evaluation on historical crashes."""
    logger.info("Starting evaluation pipeline...")
    
    # Load analysis results
    results_file = project_root / "results" / "analysis_results.csv"
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Please run run_analysis.py first")
        return
    
    logger.info(f"Loading results from {results_file}...")
    results = pd.read_csv(results_file, index_col=0, parse_dates=True)
    # Normalize timezone if needed
    if isinstance(results.index, pd.DatetimeIndex) and results.index.tz is not None:
        results.index = results.index.tz_localize(None)
    logger.info(f"Loaded {len(results)} days of results")
    
    # Initialize evaluator
    evaluator = CaseStudyEvaluator()
    
    # Evaluate 2008 crisis
    logger.info("\nEvaluating 2008 financial crisis...")
    metrics_2008 = evaluator.evaluate_2008_crisis(results)
    
    if metrics_2008.get('note'):
        logger.warning(metrics_2008['note'])
    else:
        logger.info(f"  Lead time: {metrics_2008.get('lead_time_days')} days")
        logger.info(f"  Precision: {metrics_2008.get('precision'):.3f}")
        logger.info(f"  Recall: {metrics_2008.get('recall'):.3f}")
        logger.info(f"  Warnings before crash: {metrics_2008.get('warnings_before_crash')}")
    
    # Evaluate 2020 COVID crash
    logger.info("\nEvaluating 2020 COVID-19 crash...")
    metrics_2020 = evaluator.evaluate_covid_crash(results)
    
    if metrics_2020.get('note'):
        logger.warning(metrics_2020['note'])
    else:
        logger.info(f"  Lead time: {metrics_2020.get('lead_time_days')} days")
        logger.info(f"  Precision: {metrics_2020.get('precision'):.3f}")
        logger.info(f"  Recall: {metrics_2020.get('recall'):.3f}")
        logger.info(f"  Warnings before crash: {metrics_2020.get('warnings_before_crash')}")
    
    # Save metrics
    metrics_dir = project_root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = metrics_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            '2008_crisis': metrics_2008,
            '2020_covid_crash': metrics_2020
        }, f, indent=2, default=str)
    logger.info(f"\nMetrics saved to {metrics_file}")
    
    # Generate timeline plot
    logger.info("\nGenerating timeline plot...")
    crash_dates = []
    if metrics_2008.get('crash_date'):
        crash_dates.append(metrics_2008['crash_date'])
    if metrics_2020.get('crash_date'):
        crash_dates.append(metrics_2020['crash_date'])
    
    if crash_dates:
        evaluator.plot_timeline(results, crash_dates)
        logger.info("Timeline plot saved to results/figures/instability_timeline.png")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
