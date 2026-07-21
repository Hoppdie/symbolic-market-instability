"""Run full pipeline on test data (thin CLI wrapper around src.pipeline.runner)."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.pipeline.runner import execute_run
from src.utils.run_manager import RunManager
from src.utils.logger import setup_logger

logger = setup_logger()


def main():
    """Run the full analysis pipeline as a persisted run."""
    logger.info("Starting analysis pipeline...")

    run_manager = RunManager()
    run_id = execute_run(
        ticker="^GSPC",
        start_date="2000-01-01",  # Extended to include 2008 crisis
        end_date="2024-12-31",
        run_manager=run_manager,
    )

    metadata = run_manager.get_run(run_id)
    if metadata and metadata.get('status') == 'completed':
        summary = metadata.get('summary', {})
        logger.info("\nAnalysis Summary:")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Total days analyzed: {summary.get('days_analyzed')}")
        logger.info(f"High instability days: {summary.get('high_days')}")
        logger.info(f"Medium instability days: {summary.get('medium_days')}")
        logger.info(f"Low instability days: {summary.get('low_days')}")
        logger.info("Analysis complete!")
    else:
        error = (metadata or {}).get('error', 'unknown error')
        logger.error(f"Run {run_id} failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
