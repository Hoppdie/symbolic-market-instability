"""Executable analysis pipeline: download -> features -> symbols -> inference.

Used both by scripts/run_analysis.py (CLI) and webapp/app.py (browser-triggered
runs). Every execution is persisted as a run via RunManager, including chained
evaluation on every registered historical crash (2000, 2008, 2020, 2022).
"""

import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data_acquisition.downloader import MarketDataDownloader
from src.data_acquisition.preprocessor import DataPreprocessor
from src.feature_engineering.technical_features import TechnicalFeatureComputer
from src.feature_engineering.calibrator import ThresholdCalibrator
from src.symbolization.symbolizer import FeatureSymbolizer
from src.knowledge_base.loader import RuleLoader
from src.reasoning_engine.forward_chainer import ForwardChainer
from src.reasoning_engine.explainer import ExplanationGenerator
from src.evaluation.case_studies import CaseStudyEvaluator
from src.utils.run_manager import RunManager
from src.utils.logger import setup_logger

logger = setup_logger()

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_analysis(ticker: str, start_date: str, end_date: str,
                 market: str = "US") -> pd.DataFrame:
    """
    Run the analysis pipeline and return the per-day results DataFrame.

    Raises on any failure (missing thresholds, download error, ...).
    """
    calibrator = ThresholdCalibrator()
    thresholds = calibrator.load_thresholds()

    logger.info(f"Downloading {ticker} data from {start_date} to {end_date}...")
    downloader = MarketDataDownloader()
    data = downloader.get_combined_data(ticker, start_date, end_date, market)
    logger.info(f"Downloaded {len(data)} days of data")

    preprocessor = DataPreprocessor()
    data_clean = preprocessor.clean(data)

    feature_computer = TechnicalFeatureComputer()
    features = feature_computer.compute_all_features(data_clean)

    symbolizer = FeatureSymbolizer(thresholds)
    symbolized = symbolizer.symbolize_dataframe(features)

    rules = RuleLoader().load_rules()
    chainer = ForwardChainer(rules)
    explainer = ExplanationGenerator()

    results = []
    for date, row in symbolized.iterrows():
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            date_value = date.tz_convert(None) if hasattr(date, "tz_convert") else date.tz_localize(None)
        else:
            date_value = date

        active_symbols = row['symbols']
        inference_result = chainer.infer(active_symbols)
        explanation = explainer.generate_explanation(inference_result, row, thresholds)

        results.append({
            'date': date_value,
            'close_price': float(row['Close']) if 'Close' in row else None,
            'instability_level': inference_result['instability_level'],
            'fired_rules': ','.join(inference_result['fired_rules']),
            'symbols': ','.join(sorted(active_symbols)),
            'derived_facts': ','.join(sorted(inference_result['derived_facts'])),
            'explanation': explanation,
        })

    results_df = pd.DataFrame(results)
    results_df.set_index('date', inplace=True)
    return results_df


def run_evaluation(results_df: pd.DataFrame) -> Dict:
    """
    Evaluate results against every registered historical crash whose
    window overlaps the run's date range (2000, 2008, 2020, 2022).
    """
    evaluator = CaseStudyEvaluator()
    return evaluator.evaluate_all(results_df)


def execute_run(
    ticker: str,
    start_date: str,
    end_date: str,
    run_manager: Optional[RunManager] = None,
    market: str = "US",
    run_id: Optional[str] = None,
) -> str:
    """
    Run the full pipeline as a persisted run: analysis, then chained
    evaluation. On failure the run is finalized as 'failed' with the error.

    Args:
        run_id: Use an already-created run (e.g. announced by the web API)
                instead of creating a new one.

    Returns:
        The run_id (whether the run completed or failed).
    """
    if run_manager is None:
        run_manager = RunManager()

    if run_id is None:
        run_id, _run_dir = run_manager.create_run({
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date,
            'market': market,
        })
    started = time.time()

    try:
        results_df = run_analysis(ticker, start_date, end_date, market)
        run_manager.save_results(run_id, results_df)

        # Mirror latest results for backward compatibility with
        # scripts/run_evaluation.py
        latest = PROJECT_ROOT / "results" / "analysis_results.csv"
        latest.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(latest)

        metrics = run_evaluation(results_df)
        run_manager.save_metrics(run_id, metrics)

        lead_times = [m.get('lead_time_days') for m in metrics.values()
                      if isinstance(m, dict) and m.get('lead_time_days') is not None]

        level_counts = results_df['instability_level'].value_counts()
        run_manager.finalize_run(run_id, summary={
            'days_analyzed': int(len(results_df)),
            'high_days': int(level_counts.get('High', 0)),
            'medium_days': int(level_counts.get('Medium', 0)),
            'low_days': int(level_counts.get('Low', 0)),
            'best_lead_time_days': max(lead_times) if lead_times else None,
            'duration_seconds': round(time.time() - started, 1),
        })
        logger.info(f"Run {run_id} completed in {time.time() - started:.1f}s")
    except Exception as exc:
        logger.exception(f"Run {run_id} failed")
        run_manager.finalize_run(
            run_id,
            summary={'duration_seconds': round(time.time() - started, 1)},
            status='failed',
            error=str(exc),
        )

    return run_id
