"""Case study evaluation for historical crashes."""

from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Registry of historical crash events the system is evaluated against.
# window_start is ~30 days before the crash date (the "early warning" span);
# window_end covers the acute phase of the drawdown.
CRASH_EVENTS = [
    {
        'key': '2000_dotcom_crash',
        'name': '2000 DOT-COM CRASH',
        'crash_date': '2000-03-10',       # Nasdaq peak / bust begins
        'window_start': '2000-02-09',
        'window_end': '2000-12-31',
    },
    {
        'key': '2008_crisis',
        'name': '2008 FINANCIAL CRISIS',
        'crash_date': '2008-09-15',       # Lehman Brothers bankruptcy
        'window_start': '2008-08-15',
        'window_end': '2008-12-31',
    },
    {
        'key': '2020_covid_crash',
        'name': '2020 COVID-19 CRASH',
        'crash_date': '2020-03-12',       # Black Thursday
        'window_start': '2020-02-11',
        'window_end': '2020-04-30',
    },
    {
        'key': '2022_rate_hike_drawdown',
        'name': '2022 RATE-HIKE DRAWDOWN',
        'crash_date': '2022-01-03',       # S&P 500 peak before the bear year
        'window_start': '2021-12-02',
        'window_end': '2022-12-31',
    },
]


class CaseStudyEvaluator:
    """Evaluates system performance on historical crash events."""
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory to save results. Defaults to results/
        """
        if results_dir is None:
            # Go up from src/evaluation/case_studies.py -> src/evaluation -> src -> symbolic_market_instability
            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / "results"
        
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_crash(
        self,
        results: pd.DataFrame,
        crash_date: str,
        window_start: str,
        window_end: str,
        name: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate performance on one crash event.

        Args:
            results: DataFrame with an instability_level column, date-indexed
            crash_date: The event date ('YYYY-MM-DD') lead time is measured to
            window_start: Start of the evaluation window
            window_end: End of the evaluation window
            name: Human-readable event name, echoed into the result

        Returns:
            Dictionary with evaluation metrics
        """
        # Work with a copy and normalize index timezone
        results = results.copy()
        if not isinstance(results.index, pd.DatetimeIndex):
            results.index = pd.to_datetime(results.index, utc=True)
        if results.index.tz is not None:
            results.index = results.index.tz_localize(None)

        window_start = pd.Timestamp(window_start)
        window_end = pd.Timestamp(window_end)
        results_filtered = results[
            (results.index >= window_start) &
            (results.index <= window_end)
        ].copy()

        if results_filtered.empty:
            return {
                'name': name,
                'crash_date': crash_date,
                'lead_time': None,
                'precision': None,
                'recall': None,
                'warnings_before_crash': 0,
                'total_warnings': 0,
                'note': f'No data available for this period'
                        f' ({window_start.date()} to {window_end.date()})',
            }

        # Create warning series (High or Medium instability)
        warnings = (results_filtered['instability_level'].isin(['High', 'Medium']))

        from .metrics import compute_lead_time, compute_precision, compute_recall

        lead_time = compute_lead_time(warnings, crash_date)
        precision = compute_precision(warnings, (window_start, window_end))
        recall = compute_recall(warnings, (window_start, window_end))

        crash_date_ts = pd.to_datetime(crash_date)
        warnings_before = warnings[warnings.index < crash_date_ts].sum()
        total_warnings = warnings.sum()

        return {
            'name': name,
            'crash_date': crash_date,
            'lead_time_days': lead_time,
            'precision': precision,
            'recall': recall,
            'warnings_before_crash': int(warnings_before),
            'total_warnings': int(total_warnings),
            'false_positive_rate': 1.0 - precision if precision > 0 else None
        }

    def evaluate_all(self, results: pd.DataFrame) -> Dict:
        """
        Evaluate every registered crash event whose window overlaps the
        results' date range. Returns a dict keyed by event key.
        """
        idx = pd.to_datetime(results.index, utc=True).tz_localize(None)
        data_start, data_end = idx.min(), idx.max()

        metrics = {}
        for event in CRASH_EVENTS:
            if (pd.Timestamp(event['window_end']) < data_start or
                    pd.Timestamp(event['window_start']) > data_end):
                continue  # event entirely outside this run's range
            metrics[event['key']] = self.evaluate_crash(
                results,
                crash_date=event['crash_date'],
                window_start=event['window_start'],
                window_end=event['window_end'],
                name=event['name'],
            )
        return metrics

    def evaluate_2008_crisis(self, results: pd.DataFrame) -> Dict:
        """Evaluate the 2008 financial crisis (Lehman, 2008-09-15)."""
        event = next(e for e in CRASH_EVENTS if e['key'] == '2008_crisis')
        return self.evaluate_crash(
            results, event['crash_date'], event['window_start'],
            event['window_end'], event['name'])

    def evaluate_covid_crash(self, results: pd.DataFrame) -> Dict:
        """Evaluate the 2020 COVID-19 crash (Black Thursday, 2020-03-12)."""
        event = next(e for e in CRASH_EVENTS if e['key'] == '2020_covid_crash')
        return self.evaluate_crash(
            results, event['crash_date'], event['window_start'],
            event['window_end'], event['name'])


    def plot_timeline(
        self, 
        results: pd.DataFrame, 
        crash_dates: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot instability timeline with crash dates marked.
        
        Args:
            results: DataFrame with instability_level column
            crash_dates: List of crash dates in 'YYYY-MM-DD' format
            save_path: Path to save figure. If None, saves to results/figures/
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Map instability levels to numeric values
        level_map = {'Low': 1, 'Medium': 2, 'High': 3}
        results['instability_numeric'] = results['instability_level'].map(level_map)
        
        # Plot instability level over time
        ax.plot(results.index, results['instability_numeric'], 
               linewidth=1.5, alpha=0.7, label='Instability Level')
        
        # Mark crash dates
        for crash_date in crash_dates:
            crash_idx = pd.to_datetime(crash_date)
            if crash_idx in results.index:
                ax.axvline(x=crash_idx, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Crash: {crash_date}')
        
        ax.set_ylabel('Instability Level', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Market Instability Timeline', fontsize=14, fontweight='bold')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Low', 'Medium', 'High'])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.figures_dir / "instability_timeline.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
